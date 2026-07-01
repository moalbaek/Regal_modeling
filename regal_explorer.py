"""REGAL Scenario Explorer — Python engine (port of regal_explorer.html).

The blinded milestones (60/72/78 deaths) stay fixed, so the pooled survival is
always re-calibrated and only the *split between arms* moves. Two survival shapes
are fit to the same milestones and each yields its own P(success):

  * a PLATEAU shape  — cure-mixture (Weibull per component, optional shape k),
  * a NO-PLATEAU shape — log-logistic tail.

The gap between the two headline numbers is the irreducible "is the plateau real?"
uncertainty that the blinded data cannot resolve. This file mirrors, function for
function, the JavaScript in regal_explorer.html:

  enroll · common · buildCure · buildLL · mc · median · chart(figure)

Research/analysis tool, not investment advice.
"""
import numpy as np
from datetime import date, timedelta
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# ---------------------------------------------------------------- primitives
DPM = 30.4375
BASE = date(2020, 9, 1)
mo = lambda y, m, d: (date(y, m, d) - BASE).days / DPM          # mfb() in the html

def _to_date(t):
    """Month-from-BASE -> calendar date (e.g. for the accrual-timeline x-axis)."""
    return BASE + timedelta(days=t * DPM)

Acoef = lambda cure: -np.log((0.5 - cure) / (1 - cure))          # S(med)=0.5 coefficient
lam   = lambda med, cure, k: med / Acoef(cure) ** (1.0 / k)       # Weibull scale
def Sc(t, med, cure, k):                                          # cure-mixture Weibull survival
    return cure + (1 - cure) * np.exp(-(np.clip(t, 0, None) / lam(med, cure, k)) ** k)
def sampNC(med, cure, k, u):                                     # sample a NON-cured Weibull time
    return lam(med, cure, k) * (-np.log(u)) ** (1.0 / k)
Sll   = lambda t, al, be: 1.0 / (1.0 + (np.clip(t, 1e-9, None) / al) ** be)   # log-logistic survival
sampLL = lambda al, be, u: al * (1.0 / u - 1.0) ** (1.0 / be)                 # log-logistic sample
# natural (non-disease) all-cause mortality as an independent competing risk.
# ndr is an annual death fraction; convert to a constant monthly hazard.
natH  = lambda p: (-np.log(1.0 - p) / 12.0) if p > 0 else 0.0                 # monthly hazard from annual fraction
Snat  = lambda t, h: np.exp(-h * np.clip(t, 0, None))                         # background survival factor

def obs_frac(S, tau, hd, n=10):
    """Fraction of a cohort *observed* dead by tau under independent exponential censoring
    (loss-to-follow-up hazard hd). With hd=0 this is just the death CDF 1-S(tau)."""
    if tau <= 0: return 0.0
    if hd <= 0: return 1.0 - S(tau)
    ts = np.linspace(0.0, tau, n + 1)
    f = (1.0 - S(ts)) * np.exp(-hd * ts)
    integ = (tau / n) * (f.sum() - 0.5 * (f[0] + f[-1]))          # trapezoid ∫ (1-S) e^{-hd t} dt
    return float(np.exp(-hd * tau) * (1.0 - S(tau)) + hd * integ)

# ---------------------------------------------------------------- defaults
DEFAULT_COMP = [
    {"name": "Observation",  "w": 15, "med": 6.0,  "cure": 8,  "k": 1},
    {"name": "Hydroxyurea",  "w": 5,  "med": 6.0,  "cure": 5,  "k": 1},
    {"name": "HMA",          "w": 30, "med": 10.0, "cure": 13, "k": 1},
    {"name": "Venetoclax",   "w": 35, "med": 13.0, "cure": 22, "k": 1},
    {"name": "LDAC",         "w": 15, "med": 8.0,  "cure": 9,  "k": 1},
]
DEFAULT_EV = [
    {"label": "60 events", "y": 2024, "m": 12, "d": 10, "n": 60},
    {"label": "72 events", "y": 2025, "m": 12, "d": 26, "n": 72},
    {"label": "78 events", "y": 2026, "m": 5,  "d": 11, "n": 78},
]
PRESETS = {
    "base": {"w": [15, 5, 30, 35, 15], "vc": 22},
    "low":  {"w": [25, 15, 35, 10, 15], "vc": 22},
    "dom":  {"w": [5, 2, 23, 60, 10],  "vc": 22},
    "bear": {"w": [5, 2, 13, 70, 10],  "vc": 36},
}

def default_cfg(**over):
    cfg = dict(N=126, FINAL=80, HRC=0.636, fnr=0.20, bl=0.50, beta=1.20,
               ndr=0.02, IA=60, futHR=1.0, drop=0.0, esel=0.25, unweighted=False,
               comp=[dict(c) for c in DEFAULT_COMP],
               ev=[dict(e) for e in DEFAULT_EV])
    cfg.update(over)
    return cfg

def apply_preset(cfg, name):
    p = PRESETS[name]
    for i, w in enumerate(p["w"]):
        cfg["comp"][i]["w"] = w
    cfg["comp"][3]["cure"] = p["vc"]
    return cfg

# ---------------------------------------------------------------- enrollment
def enroll(bl, N):
    """Monthly cohorts summing to N; bl interpolates flat(0) <-> back-loaded(1)."""
    win = [(2020, 9, 2020, 12, 1.2, 0.8), (2021, 1, 2021, 12, 1.8, 0.9),
           (2022, 1, 2022, 12, 2.8, 2.2), (2023, 1, 2023, 11, 4.2, 5.6),
           (2023, 12, 2024, 3, 6.2, 6.2)]
    coh = []
    for ys, ms, ye, me, rf, rb in win:
        y, m = ys, ms
        rate = (1 - bl) * rf + bl * rb
        while (y < ye) or (y == ye and m <= me):
            coh.append([mo(y, m, 15), rate]); m += 1
            if m > 12: m = 1; y += 1
    coh = np.array(coh, float)
    coh[:, 1] *= N / coh[:, 1].sum()
    return coh

def common(cfg):
    comp = cfg["comp"]
    tot = sum(c["w"] for c in comp) or 1.0
    w = np.array([c["w"] / tot for c in comp])
    cm = [dict(med=c["med"], cure=min(max(c["cure"] / 100.0, 0.0), 0.49),
               k=max(c.get("k", 1) or 1, 0.3)) for c in comp]
    coh = enroll(cfg["bl"], cfg["N"])
    MT = np.array([mo(e["y"], e["m"], e["d"]) for e in cfg["ev"]])
    MOBS = np.array([e["n"] for e in cfg["ev"]], float)
    WT = np.array([1.0, 1.0, 1.0]) if cfg.get("unweighted") else np.array([1.0, 2.0, 4.0])
    return w, cm, coh, MT, MOBS, WT

def median(S):
    if S(900.0) >= 0.5: return np.inf
    lo, hi = 0.01, 900.0
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if S(m) > 0.5: lo = m
        else: hi = m
    return 0.5 * (lo + hi)

# ---------------------------------------------------------------- enrollment readouts
def med_enroll(coh):
    """Month-from-BASE at which cumulative enrollment crosses N/2 (the median enrollment)."""
    c = np.cumsum(coh[:, 1]); half = c[-1] / 2.0
    i = int(np.searchsorted(c, half))
    return coh[min(i, len(coh) - 1), 0]

def month_label(m):
    return _to_date(m).strftime("%b %Y")

def cum_enroll(coh, y, m, d=28):
    """Cumulative patients enrolled by a calendar date (for the sourced PR anchors)."""
    return float(coh[coh[:, 0] <= mo(y, m, d), 1].sum())

# ---------------------------------------------------------------- cure-mixture model
def build_cure(cfg):
    w, cm, coh, MT, MOBS, WT = common(cfg)
    fnr = cfg["fnr"]
    h = natH(cfg.get("ndr", 0.0)); hd = natH(cfg.get("drop", 0.0))
    F = min(max(cfg.get("esel", 0.0), 0.0), 0.5)   # enrollment selection: drop weakest fraction F
    def Ssel(t, c):                                 # survival of the healthiest (1-F) of a component
        return np.minimum(1.0, Sc(t, c["med"], c["cure"], c["k"]) / (1 - F))
    pibat = sum(w[i] * cm[i]["cure"] / (1 - F) for i in range(len(cm)))
    obs = cm[0]
    def Sbat(t): return sum(w[i] * Ssel(t, cm[i]) for i in range(len(cm)))
    def Snc(t):  return (Sbat(t) - pibat) / (1 - pibat)
    def Spool(t, pr):
        return 0.5 * Sbat(t) + 0.5 * ((1 - fnr) * (pr + (1 - pr) * Snc(t)) + fnr * Ssel(t, obs))
    # observed deaths are all-cause (disease x background mortality) and net of loss-to-follow-up.
    def ed(T, pr):
        Sf = lambda t: Spool(t, pr) * Snat(t, h)
        return sum(c[1] * obs_frac(Sf, T - c[0], hd) for c in coh if c[0] <= T)

    # The GPS responder cure (presp) is the only free parameter; the BAT arm is fixed by the
    # component medians and enrollment selection.
    best, bs = 0.6, 1e18
    for ki in range(91):
        pr = ki / 90.0
        e = sum(WT[j] * (ed(MT[j], pr) - MOBS[j]) ** 2 for j in range(3))
        if e < bs: bs, best = e, pr
    for it in range(3):
        st = 0.05 / (it + 1)
        for dp in range(-4, 5):
            pr = min(0.985, max(0.0, best + dp * st))
            e = sum(WT[j] * (ed(MT[j], pr) - MOBS[j]) ** 2 for j in range(3))
            if e < bs: bs, best = e, pr
    presp = best
    pgps = (1 - fnr) * presp + fnr * obs["cure"] / (1 - F)
    # returned curves are all-cause (disease x background mortality).
    Sb = lambda t: Sbat(t) * Snat(t, h)
    Sg = lambda t: ((1 - fnr) * (presp + (1 - presp) * Snc(t)) + fnr * Ssel(t, obs)) * Snat(t, h)
    Sp = lambda t: Spool(t, presp) * Snat(t, h)
    return dict(kind="cure", cfg=cfg, w=w, cm=cm, coh=coh, MT=MT, MOBS=MOBS, WT=WT,
                presp=presp, h=h, pibat=pibat, pgps=pgps, obs=obs,
                Sbat=Sb, Sgps=Sg, Spool=Sp, ed_raw=ed,
                batMed=median(Sb), gpsMed=median(Sg), poolMed=median(Sp),
                poolCure=0.5 * (pibat + pgps), ed=lambda t: ed(t, presp))

# ---------------------------------------------------------------- log-logistic (no plateau)
def build_ll(cfg):
    w, cm, coh, MT, MOBS, WT = common(cfg)
    fnr, beta = cfg["fnr"], cfg["beta"]
    h = natH(cfg.get("ndr", 0.0)); hd = natH(cfg.get("drop", 0.0))
    F = min(max(cfg.get("esel", 0.0), 0.0), 0.5)          # enrollment selection: drop weakest fraction F
    def Ssl(t, al): return np.minimum(1.0, Sll(t, al, beta) / (1 - F))   # healthiest (1-F) of a log-logistic subgroup
    mC = sum(w[i] * cm[i]["med"] for i in range(len(cm)))
    mObs = cm[0]["med"]
    def Spool(t, k, r):
        mB = k * mC
        return 0.5 * Ssl(t, mB) + 0.5 * ((1 - fnr) * Ssl(t, r * mB) + fnr * Ssl(t, k * mObs))
    def ed(T, k, r):
        Sf = lambda t: Spool(t, k, r) * Snat(t, h)
        return sum(c[1] * obs_frac(Sf, T - c[0], hd) for c in coh if c[0] <= T)

    best, bs = (1.4, 2.0), 1e18
    for ki in range(49):
        for ri in range(61):
            k = 0.6 + ki / 48.0 * 2.6; r = 1 + ri / 60.0 * 6
            e = sum(WT[j] * (ed(MT[j], k, r) - MOBS[j]) ** 2 for j in range(3))
            if e < bs: bs, best = e, (k, r)
    for it in range(3):
        k0, r0 = best; st = 0.04 / (it + 1)
        for dk in range(-3, 4):
            for dr in range(-3, 4):
                k = max(0.4, k0 + dk * st * 2); r = max(1.0, r0 + dr * st * 4)
                e = sum(WT[j] * (ed(MT[j], k, r) - MOBS[j]) ** 2 for j in range(3))
                if e < bs: bs, best = e, (k, r)
    k, r = best; mB = k * mC
    Sb = lambda t: Ssl(t, mB) * Snat(t, h)
    Sg = lambda t: ((1 - fnr) * Ssl(t, r * mB) + fnr * Ssl(t, k * mObs)) * Snat(t, h)
    Sp = lambda t: 0.5 * Sb(t) + 0.5 * Sg(t)
    return dict(kind="ll", cfg=cfg, w=w, cm=cm, coh=coh, MT=MT, MOBS=MOBS, WT=WT,
                beta=beta, k=k, r=r, mB=mB, h=h, mObs=k * mObs, ed_raw=ed,
                Sbat=Sb, Sgps=Sg, Spool=Sp,
                batMed=median(Sb), gpsMed=median(Sg), ratio=r, poolMed=median(Sp),
                ed=lambda t: ed(t, k, r))

# ---------------------------------------------------------------- fit uncertainty
def fit_ci(cfg, builder):
    """Poisson ~68% interval on the GPS median from the +/-sqrt(n) sampling noise of the event counts.
    Returns (med_more, med_fewer): more deaths -> shorter GPS median; fewer deaths -> longer (often NR)."""
    def refit(sign):
        c = dict(cfg); c["ev"] = [dict(e) for e in cfg["ev"]]
        for e in c["ev"]:
            e["n"] = max(1.0, e["n"] + sign * np.sqrt(e["n"]))
        return builder(c)["gpsMed"]
    return refit(+1), refit(-1)

# ---------------------------------------------------------------- shared Monte-Carlo
def mc(M, nsim=1500, seed=987654321):
    """Enrollment -> per-arm death draws -> censor at FINAL-th event -> log-rank test.
    Returns dict(ps, reach, medHR, medHR_IA, futOK, aliveG, aliveB): P(significant), fraction
    reaching the trigger, median final HR, median implied HR at the interim (feature 1), whether
    that clears the futility threshold, and the mean per-arm patients alive at the 80th (feature 3)."""
    cfg = M["cfg"]; N, FINAL, HRC, fnr = cfg["N"], cfg["FINAL"], cfg["HRC"], cfg["fnr"]
    h = natH(cfg.get("ndr", 0.0))                                  # background mortality competing risk (an event)
    hdrop = natH(cfg.get("drop", 0.0))                             # loss-to-follow-up (censoring, not an event)
    ZC = abs(np.log(HRC)) * np.sqrt(FINAL) / 2.0
    rng = np.random.default_rng(seed)
    coh, w, cm = M["coh"], M["w"], M["cm"]
    cohp = coh[:, 1] / coh[:, 1].sum()                              # cohort enrollment probs
    F = min(max(cfg.get("esel", 0.0), 0.0), 0.5)                    # enrollment selection: drop weakest fraction F
    snq = np.array([(1 - F - cm[i]["cure"]) / (1 - cm[i]["cure"]) for i in range(len(cm))])  # non-cured survival at the F-quantile (caps the conditional draw)
    ncw = np.array([w[i] * (1 - F - cm[i]["cure"]) for i in range(len(cm))])
    ncw = ncw / ncw.sum()                                          # BAT non-cured component mix (selected)
    n1 = N // 2
    IA = min(int(cfg.get("IA", 60)), FINAL - 1)                    # interim-analysis event count
    futHR = cfg.get("futHR", 1.0)                                  # interim futility HR threshold
    sig = reached = 0; hrs = []; hrsIA = []; aliveG = aliveB = 0.0

    def score(time, ev):                                          # log-rank/Cox score test (num, var)
        idx = np.argsort(time, kind="mergesort")
        totX = arm.sum(); prefX = 0; num = 0.0; varr = 0.0
        for p in range(N):
            i = idx[p]; nAt = N - p; sx = totX - prefX
            if ev[i] == 1:
                pb = sx / nAt; num += arm[i] - pb; varr += pb * (1 - pb)
            prefX += arm[i]
        return num, varr

    def draw_cure_bat(n):
        out = np.empty(n)
        pick = rng.choice(len(cm), size=n, p=w)
        for j, c in enumerate(cm):
            idx = np.where(pick == j)[0]
            if idx.size:
                cured = rng.random(idx.size) < c["cure"] / (1 - F)
                s = sampNC(c["med"], c["cure"], c["k"], rng.random(idx.size) * snq[j])
                out[idx] = np.where(cured, 1e9, s)
        return out

    def draw_cure_gps(n):
        out = np.empty(n)
        isnr = rng.random(n) < fnr
        nr = np.where(isnr)[0]; rs = np.where(~isnr)[0]
        obs = M["obs"]
        if nr.size:
            cured = rng.random(nr.size) < obs["cure"] / (1 - F)
            s = sampNC(obs["med"], obs["cure"], obs["k"], rng.random(nr.size) * snq[0])
            out[nr] = np.where(cured, 1e9, s)
        if rs.size:
            cured = rng.random(rs.size) < M["presp"]
            pick = rng.choice(len(cm), size=rs.size, p=ncw)
            s = np.empty(rs.size)
            for j, c in enumerate(cm):
                jj = np.where(pick == j)[0]
                if jj.size: s[jj] = sampNC(c["med"], c["cure"], c["k"], rng.random(jj.size) * snq[j])
            out[rs] = np.where(cured, 1e9, s)
        return out

    for _ in range(nsim):
        arm = rng.permutation(np.r_[np.ones(n1), np.zeros(N - n1)]).astype(int)
        en = coh[rng.choice(len(coh), size=N, p=cohp), 0]
        surv = np.empty(N)
        a1 = arm == 1; a0 = ~a1
        if M["kind"] == "cure":
            surv[a1] = draw_cure_gps(a1.sum())
            surv[a0] = draw_cure_bat(a0.sum())
        else:
            mB, beta, r, mObs = M["mB"], M["beta"], M["r"], M["mObs"]
            u = rng.random(N) * (1 - F)                              # conditional on outliving the F-quantile (enrollment selection)
            nr = a1 & (rng.random(N) < fnr)
            surv[a0] = sampLL(mB, beta, u[a0])
            surv[a1 & ~nr] = sampLL(r * mB, beta, u[a1 & ~nr])
            surv[nr] = sampLL(mObs, beta, u[nr])
        if h > 0:                                                   # natural death may preempt disease death
            surv = np.minimum(surv, -np.log(rng.random(N)) / h)
        # loss-to-follow-up: an independent censoring time; if it precedes death the patient is censored
        td = (-np.log(rng.random(N)) / hdrop) if hdrop > 0 else np.full(N, np.inf)
        isdeath = surv <= td                                        # a death is observed only if it precedes dropout
        obsT = np.minimum(surv, td)                                 # follow-up time (death or censoring)
        rawcal = en + surv                                          # death calendar ignoring dropout (alive-count basis)
        dcal = np.where(isdeath, en + surv, 1e9)                    # event calendar feeding the trigger
        fin = np.sort(dcal[dcal < 1e8])
        if fin.size < FINAL: continue
        reached += 1
        t80 = fin[FINAL - 1]
        ev = (isdeath & (dcal <= t80)).astype(int)
        time = np.minimum(obsT, np.clip(t80 - en, 0, None))
        num, varr = score(time, ev)                              # final-analysis test (the trial's)
        if varr > 0:
            z = -num / np.sqrt(varr)
            if z > ZC: sig += 1
            hrs.append(np.exp(num / varr))
        # implied HR at the interim (the futility read-through, feature 1)
        tIA = fin[IA - 1]
        evIA = (isdeath & (dcal <= tIA)).astype(int)
        timeIA = np.minimum(obsT, np.clip(tIA - en, 0, None))
        numIA, varrIA = score(timeIA, evIA)
        if varrIA > 0: hrsIA.append(np.exp(numIA / varrIA))
        # per-arm patients still alive at the 80th event (feature 3, before censoring)
        aliveG += np.sum((arm == 1) & (rawcal > t80))
        aliveB += np.sum((arm == 0) & (rawcal > t80))
    hrs.sort(); hrsIA.sort()
    medHR_IA = hrsIA[len(hrsIA) // 2] if hrsIA else np.nan
    return dict(ps=(sig / reached if reached else 0.0),
                reach=reached / nsim,
                medHR=(hrs[len(hrs) // 2] if hrs else np.nan),
                hrsAll=np.array(hrs),                          # full final-HR distribution (for the histogram)
                medHR_IA=medHR_IA, futHR=futHR, futOK=bool(medHR_IA <= futHR),
                aliveG=(aliveG / reached if reached else np.nan),
                aliveB=(aliveB / reached if reached else np.nan))

# ---------------------------------------------------------------- figure
NAVY = "#0b2545"; RED = "#9e2b25"; TEAL = "#197278"; GREY = "#6b6f72"; ORANGE = "#e8910b"

def proj_cross(ed_fn, target, t0, t1):
    """First month-from-BASE where cumulative events ed_fn(t) reaches target, or None if it never does
    within [t0,t1] (a plateau curve can asymptote below the trigger -> the 80th event stalls)."""
    if ed_fn(t1) < target:
        return None
    lo, hi = t0, t1
    for _ in range(50):
        m = 0.5 * (lo + hi)
        if ed_fn(m) < target: lo = m
        else: hi = m
    return 0.5 * (lo + hi)

def figure(path, nsim=1500):
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140})
    fig, ax = plt.subplots(3, 3, figsize=(16.5, 15.4)); tg = np.linspace(0, 48, 300)

    # base preset is reused by (a),(d),(e),(f); fit both shapes once
    cfg = apply_preset(default_cfg(), "base")
    Mc = build_cure(cfg); Ml = build_ll(cfg)
    rc = mc(Mc, nsim); rl = mc(Ml, nsim)

    # (a) survival curves for the base preset, plateau (cure) shape
    a = ax[0, 0]
    a.plot(tg, 100 * Mc["Sbat"](tg), color=RED, lw=2.4, label=f"BAT (cure {100*Mc['pibat']:.0f}%)")
    a.plot(tg, 100 * Mc["Sgps"](tg), color=NAVY, lw=2.4, label=f"GPS (cure {100*Mc['pgps']:.0f}%)")
    a.plot(tg, 100 * Mc["Spool"](tg), color="#111", lw=1.6, ls="--", alpha=.75, label="Pooled (blinded)")
    for e in cfg["ev"]:
        a.axhline(100 * (1 - e["n"] / cfg["N"]), color=ORANGE, ls=":", lw=1, alpha=.7)
    a.axhline(100 * (1 - cfg["FINAL"] / cfg["N"]), color=RED, ls="--", lw=1, alpha=.5)
    a.set_title("(a) Blinded data pin the pooled curve; the arm split is an assumption",
                fontweight="bold", fontsize=9)
    a.set_xlabel("months from randomization"); a.set_ylabel("% alive")
    a.set_xlim(0, 48); a.set_ylim(0, 101); a.legend(fontsize=7.4, loc="upper right")

    # (b) dual P(success) across the non-responder sweep (base preset)
    fr = [0, 10, 20, 30, 40]; pc = []; pll = []
    for f in fr:
        c = apply_preset(default_cfg(fnr=f / 100.0), "base")
        pc.append(100 * mc(build_cure(c), nsim)["ps"])
        pll.append(100 * mc(build_ll(c), nsim)["ps"])
    b = ax[0, 1]
    b.plot(fr, pc, color=NAVY, lw=2.4, marker="o", label="Plateau (cure-mixture)")
    b.plot(fr, pll, color=ORANGE, lw=2.2, ls="-.", marker="s", label="No-plateau (log-logistic)")
    b.axhline(50, color=GREY, ls=":", lw=1)
    b.set_ylim(0, 103); b.set_xlabel("% GPS non-responders"); b.set_ylabel("P(success) %")
    b.set_title("(b) Non-responders barely move either shape's P(success)",
                fontweight="bold", fontsize=9); b.legend(fontsize=7.6)

    # (c) dual P(success) across the four BAT-composition presets
    names = ["base", "low", "dom", "bear"]; labels = ["Base", "Low-ven", "Ven-dom", "Bear"]
    gc = []; gl = []
    for nm in names:
        c = apply_preset(default_cfg(), nm)
        gc.append(100 * mc(build_cure(c), nsim)["ps"])
        gl.append(100 * mc(build_ll(c), nsim)["ps"])
    c = ax[0, 2]; x = np.arange(len(names))
    c.bar(x - 0.19, gc, 0.36, color=NAVY, label="Plateau")
    c.bar(x + 0.19, gl, 0.36, color=ORANGE, label="No-plateau")
    c.set_xticks(x); c.set_xticklabels(labels); c.set_ylim(0, 103)
    c.set_ylabel("P(success) %")
    c.set_title("(c) The plateau-vs-tail gap is the irreducible uncertainty",
                fontweight="bold", fontsize=9); c.legend(fontsize=7.6)

    # (d) event-accrual timeline — modeled cumulative deaths vs calendar, milestone anchors, 80-event trigger
    d = ax[1, 0]; N, FINAL = cfg["N"], cfg["FINAL"]
    t0 = float(Mc["coh"][0, 0]); t1 = 90.0
    ts = np.linspace(t0, t1, 220); dts = [_to_date(t) for t in ts]
    edc = np.array([Mc["ed"](t) for t in ts]); edl = np.array([Ml["ed"](t) for t in ts])
    d.plot(dts, edc, color=NAVY, lw=2.2, label="Plateau accrual")
    d.plot(dts, edl, color=ORANGE, lw=2.0, ls="-.", label="No-plateau accrual")
    d.scatter([_to_date(t) for t in Mc["MT"]], Mc["MOBS"], color=RED, s=34, zorder=5,
              label="Blinded milestones (60/72/78)")
    d.axhline(FINAL, color=RED, ls="--", lw=1, alpha=.6)
    d.text(dts[2], FINAL + 1.5, f"{FINAL}-event trigger", color=RED, fontsize=7.5)
    for ed_fn, col in [(Mc["ed"], NAVY), (Ml["ed"], ORANGE)]:
        tc = proj_cross(ed_fn, FINAL, float(Mc["MT"][-1]), t1)
        if tc is not None:
            d.axvline(_to_date(tc), color=col, ls=":", lw=1, alpha=.65)
            d.text(_to_date(tc), 6, _to_date(tc).strftime("%b %Y"), color=col,
                   fontsize=7, rotation=90, va="bottom", ha="right")
    d.set_ylim(0, max(95.0, float(edc.max()), float(edl.max())) * 1.03); d.set_ylabel("cumulative deaths")
    d.set_title("(d) When does the 80th event fire? Plateau accrual can stall",
                fontweight="bold", fontsize=9)
    d.legend(fontsize=7.2, loc="lower right")
    for lab in d.get_xticklabels(): lab.set_rotation(25); lab.set_ha("right"); lab.set_fontsize(7.5)

    # (e) distribution of simulated final HRs — P(success) is the mass left of the threshold
    e = ax[1, 1]; HRC = cfg["HRC"]
    bins = np.linspace(0.0, 1.6, 41)
    hc = rc["hrsAll"]; hl = rl["hrsAll"]
    hc = np.clip(hc[np.isfinite(hc)], 0, 1.59); hl = np.clip(hl[np.isfinite(hl)], 0, 1.59)
    e.hist(hc, bins=bins, density=True, color=NAVY, alpha=.55, label=f"Plateau  (P={100*rc['ps']:.0f}%)")
    e.hist(hl, bins=bins, density=True, color=ORANGE, alpha=.45, label=f"No-plateau  (P={100*rl['ps']:.0f}%)")
    e.axvspan(0, HRC, color=TEAL, alpha=.07)
    e.axvline(HRC, color=RED, lw=1.4, ls="--")
    e.axvline(1.0, color=GREY, lw=1, ls=":")
    ytop = e.get_ylim()[1]
    e.text(HRC - 0.02, ytop * 0.92, f"significant\nHR ≤ {HRC:.3f}", color=RED, fontsize=7.2, ha="right", va="top")
    e.set_xlim(0, 1.6); e.set_xlabel("simulated final hazard ratio (GPS / BAT)"); e.set_ylabel("density")
    e.set_title("(e) Each trial's HR is a draw; success = mass below the line",
                fontweight="bold", fontsize=9); e.legend(fontsize=7.2, loc="upper right")

    # (f) plateau-vs-tail divergence — both pooled curves agree at the milestones, fan apart in the tail
    f = ax[1, 2]
    sc = 100 * Mc["Spool"](tg); sl = 100 * Ml["Spool"](tg)
    f.fill_between(tg, sc, sl, color=GREY, alpha=.18, label="shape disagreement")
    f.plot(tg, sc, color=NAVY, lw=2.2, label="Plateau pooled")
    f.plot(tg, sl, color=ORANGE, lw=2.0, ls="-.", label="No-plateau pooled")
    for ev in cfg["ev"]:                                       # event-fraction levels both shapes are pinned to
        f.axhline(100 * (1 - ev["n"] / N), color=RED, ls=":", lw=.9, alpha=.5)
    f.set_xlim(0, 48); f.set_ylim(0, 101)
    f.set_xlabel("months from randomization"); f.set_ylabel("% alive (pooled)")
    f.set_title("(f) Both shapes pinned at the milestones; the gap is irreducible",
                fontweight="bold", fontsize=9); f.legend(fontsize=7.2, loc="upper right")

    # (g) enrollment validation — modeled cumulative enrollment vs the sourced public anchors
    gx = ax[2, 0]; coh = Mc["coh"]; N = cfg["N"]
    cdate = [_to_date(t) for t in coh[:, 0]]; cum = np.cumsum(coh[:, 1])
    gx.plot(cdate, cum, color=TEAL, lw=2.4, label="modeled cumulative enrollment")
    me = med_enroll(coh)
    gx.axvline(_to_date(me), color=GREY, ls=":", lw=1)
    gx.text(_to_date(me), 4, f"median {month_label(me)}", color=GREY, fontsize=7.5,
            rotation=90, va="bottom", ha="right")
    anchors = [(2022, 4, 20), (2023, 11, 104), (2024, 4, 126)]      # sourced PR cumulative counts
    gx.scatter([_to_date(mo(y, m, 28)) for (y, m, _) in anchors], [n for (_, _, n) in anchors],
               color=RED, s=42, zorder=5, label="sourced PR anchors (~20/104/126)")
    gx.set_ylabel("patients enrolled"); gx.set_ylim(0, N * 1.05)
    gx.set_title("(g) Modeled enrollment tracks the sourced public milestones",
                 fontweight="bold", fontsize=9); gx.legend(fontsize=7.2, loc="lower right")
    for lab in gx.get_xticklabels(): lab.set_rotation(25); lab.set_ha("right"); lab.set_fontsize(7.5)

    # (h) P(success) as a power curve vs the implied treatment effect; shaded = the effect the data allow
    hx = ax[2, 1]; nsim_h = max(250, nsim // 3)

    def power_sweep(M, key, vals, fixed):
        """Sweep one effect knob (presp for plateau, r for no-plateau); return implied HR, P(success),
        and milestone misfit at each point. The pooled curve is only data-consistent near the fit."""
        base = M[key]; hr, ps, E = [], [], []
        for v in vals:
            M[key] = v; r = mc(M, nsim_h)
            hr.append(r["medHR"]); ps.append(100 * r["ps"])
            E.append(sum(M["WT"][k] * (M["ed_raw"](M["MT"][k], *fixed(v)) - M["MOBS"][k]) ** 2 for k in range(3)))
        M[key] = base
        hr, ps, E = np.array(hr), np.array(ps), np.array(E)
        m = np.isfinite(hr) & np.isfinite(ps); hr, ps, E = hr[m], ps[m], E[m]
        o = np.argsort(hr); return hr[o], ps[o], E[o]

    p_hr, p_ps, p_E = power_sweep(Mc, "presp", np.linspace(0.0, 0.97, 13), lambda pv: (pv,))
    l_hr, l_ps, l_E = power_sweep(Ml, "r", np.linspace(1.0, 7.0, 13), lambda rv: (Ml["k"], rv))

    def band(hr, E):                                            # HR span of the data-consistent (low-misfit) points
        if not len(E): return None
        ok = hr[E <= E.min() + 0.08 * (E.max() - E.min())]
        return (float(ok.min()), float(ok.max())) if len(ok) else None
    for hr, E, col in [(p_hr, p_E, NAVY), (l_hr, l_E, ORANGE)]:
        bd = band(hr, E)
        if bd: hx.axvspan(bd[0], bd[1], color=col, alpha=.10)
    hx.plot(p_hr, p_ps, color=NAVY, lw=2.4, marker="o", ms=3, label="Plateau")
    hx.plot(l_hr, l_ps, color=ORANGE, lw=2.2, ls="-.", marker="s", ms=3, label="No-plateau")
    hx.scatter([rc["medHR"]], [100 * rc["ps"]], color=NAVY, s=130, marker="*", zorder=6, edgecolor="#fff", linewidth=.8)
    hx.scatter([rl["medHR"]], [100 * rl["ps"]], color=ORANGE, s=130, marker="*", zorder=6, edgecolor="#fff", linewidth=.8)
    hx.axvline(HRC, color=RED, ls="--", lw=1.2)
    hx.text(HRC + 0.012, 6, f"success threshold HR={HRC:.3f}", color=RED, fontsize=7, rotation=90, va="bottom")
    hx.set_xlim(0, 1.15); hx.set_ylim(0, 103)
    hx.set_xlabel("implied trial hazard ratio (GPS / BAT)"); hx.set_ylabel("P(success) %")
    hx.set_title("(h) P(success) vs effect size; ★ = current fit, shaded = effect the data allow",
                 fontweight="bold", fontsize=9); hx.legend(fontsize=7.2, loc="lower left")

    # (i) how enrollment selection q lifts the BAT arm — median OS and cure fraction, the two
    #     quantities the comparator assumption turns on, shown explicitly.
    ix = ax[2, 2]; w0, cm0, h0 = Mc["w"], Mc["cm"], Mc["h"]
    qs = np.linspace(0.0, 0.5, 26)
    bat_med, bat_cure = [], []
    for q in qs:
        pib = sum(w0[i] * cm0[i]["cure"] / (1 - q) for i in range(len(cm0)))
        Sbq = lambda t, q=q: (sum(w0[i] * np.minimum(1.0, Sc(t, cm0[i]["med"], cm0[i]["cure"], cm0[i]["k"]) / (1 - q))
                                  for i in range(len(cm0)))) * Snat(t, h0)
        bat_cure.append(100 * pib); bat_med.append(median(Sbq))
    bat_med = np.array(bat_med); mcap = 60.0                       # clip a "not reached" median for display
    med_plot = np.where(np.isfinite(bat_med), np.minimum(bat_med, mcap), mcap)
    ix.axvspan(20, 35, color=GREY, alpha=.10)                      # defensible selection band (~fitness + guarantee-time)
    ix.text(27.5, 3, "defensible\n~20–35%", color=GREY, fontsize=7, ha="center", va="bottom")
    ix.plot(100 * qs, med_plot, color=NAVY, lw=2.4, marker="o", ms=2.5, label="BAT median OS (mo)")
    ix.set_xlabel("enrollment selection q — drop weakest % "); ix.set_ylabel("BAT median OS (months)", color=NAVY)
    ix.tick_params(axis="y", labelcolor=NAVY); ix.set_xlim(0, 50); ix.set_ylim(0, mcap * 1.02)
    ix.axhline(mcap, color=NAVY, ls=":", lw=.8, alpha=.4)
    ix.text(1, mcap - 2, "≥60 / NR", color=NAVY, fontsize=6.8, va="top")
    ix2 = ix.twinx(); ix2.spines["top"].set_visible(False)
    ix2.plot(100 * qs, bat_cure, color=TEAL, lw=2.2, ls="-.", marker="s", ms=2.5, label="BAT cure fraction (%)")
    ix2.set_ylabel("BAT cure fraction (%)", color=TEAL); ix2.tick_params(axis="y", labelcolor=TEAL)
    ix2.set_ylim(0, max(55.0, 1.15 * max(bat_cure)))
    ix.set_title("(i) Enrollment selection q lifts BAT median OS & cure fraction",
                 fontweight="bold", fontsize=9)
    h1, l1 = ix.get_legend_handles_labels(); h2, l2 = ix2.get_legend_handles_labels()
    ix.legend(h1 + h2, l1 + l2, fontsize=7.2, loc="upper left")

    fig.suptitle("REGAL Scenario Explorer — two survival shapes fit the same blinded milestones; "
                 "their P(success) gap is the 'is the plateau real?' uncertainty.",
                 fontweight="bold", fontsize=10.5, y=1.01)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    return path

# ---------------------------------------------------------------- CLI
def fmt_med(m): return "NR" if not np.isfinite(m) else f"{m:.0f}mo"

if __name__ == "__main__":
    NSIM = 800   # matches the html's interactive budget (~600); raise for tighter MC error
    base = apply_preset(default_cfg(), "base")
    Mc, Ml = build_cure(base), build_ll(base)
    rc, rl = mc(Mc, NSIM), mc(Ml, NSIM)
    wmode = "unweighted" if base["unweighted"] else "weighted 1/2/4"
    print(f"REGAL Scenario Explorer (base preset, f_nr=20%, natural death {100*base['ndr']:.1f}%/yr, "
          f"loss-to-FU {100*base['drop']:.0f}%/yr, enrol-selection drop-weakest {100*base['esel']:.0f}%, fit {wmode})")
    print(f"  BAT  : cure {100*Mc['pibat']:.0f}%  median {fmt_med(Mc['batMed'])}  @36mo {100*Mc['Sbat'](36):.0f}%")
    ci_more, ci_few = fit_ci(base, build_cure)
    print(f"  GPS  : cure {100*Mc['pgps']:.0f}%  median {fmt_med(Mc['gpsMed'])}  (cure gap +{100*(Mc['pgps']-Mc['pibat']):.0f}pp)")
    print(f"         GPS median Poisson 68% CI [{fmt_med(ci_more)} .. {fmt_med(ci_few)}] (from 60/72/78 +/- sqrt(n))")
    print(f"  pool : median {fmt_med(Mc['poolMed'])}")
    coh = Mc['coh']
    print(f"  enrol: median {month_label(med_enroll(coh))}  "
          f"cum {cum_enroll(coh,2022,4):.0f}/{cum_enroll(coh,2023,11):.0f}/{cum_enroll(coh,2024,4):.0f} "
          f"by Apr22/Nov23/Apr24 (sourced ~20/104/126)")
    edv = [Mc['ed'](t) for t in Mc['MT']]
    print(f"  fit  : modeled deaths {'/'.join(f'{x:.0f}' for x in edv)}  vs observed {'/'.join(f'{x:.0f}' for x in Mc['MOBS'])}")
    # the interim HR is undefined when no sim reaches the 80th event; only flag a real breach
    if np.isfinite(rc['medHR_IA']):
        fut = "OK" if rc['futOK'] else f"VIOLATED >{base['futHR']:.2f}"
        ia = f"{rc['medHR_IA']:.2f} (futility {fut})"
    else:
        ia = "n/a (80th not reached)"
    print(f"\n  P(success) PLATEAU (cure-mixture) : {100*rc['ps']:.0f}%   medHR {rc['medHR']:.2f}   reached {100*rc['reach']:.0f}%")
    print(f"         interim: implied HR@{base['IA']} {ia}   "
          f"@80th: {rc['aliveG']:.0f} GPS alive / {rc['aliveB']:.0f} BAT alive")
    ia_ll = f"{rl['medHR_IA']:.2f}" if np.isfinite(rl['medHR_IA']) else "n/a"
    print(f"  P(success) NO-PLATEAU (log-logistic β={base['beta']:.2f}): {100*rl['ps']:.0f}%   medHR {rl['medHR']:.2f}   "
          f"interim HR {ia_ll}   @80th: {rl['aliveG']:.0f} GPS / {rl['aliveB']:.0f} BAT")
    print(f"  -> shape gap = {abs(round(100*rc['ps'])-round(100*rl['ps']))} points (irreducible from blinded data)\n")

    print(f"{'preset':>8} | {'f_nr':>5} | {'P(plateau)':>10} {'P(no-plat)':>10} | {'BATmed':>7} {'GPSmed':>7}")
    for nm in ["base", "low", "dom", "bear"]:
        c = apply_preset(default_cfg(), nm)
        mcc, mll = build_cure(c), build_ll(c)
        rcc, rll = mc(mcc, NSIM), mc(mll, NSIM)
        print(f"{nm:>8} | {100*c['fnr']:4.0f}% | {100*rcc['ps']:9.0f}% {100*rll['ps']:9.0f}% | "
              f"{fmt_med(mcc['batMed']):>7} {fmt_med(mcc['gpsMed']):>7}")

    out = figure("regal_explorer_panel.png", NSIM)
    print(f"\nsaved {out}")
