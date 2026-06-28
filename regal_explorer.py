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
from datetime import date
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# ---------------------------------------------------------------- primitives
DPM = 30.4375
BASE = date(2020, 9, 1)
mo = lambda y, m, d: (date(y, m, d) - BASE).days / DPM          # mfb() in the html

Acoef = lambda cure: -np.log((0.5 - cure) / (1 - cure))          # S(med)=0.5 coefficient
lam   = lambda med, cure, k: med / Acoef(cure) ** (1.0 / k)       # Weibull scale
def Sc(t, med, cure, k, L):                                       # cure-mixture Weibull, time x L
    return cure + (1 - cure) * np.exp(-(L * np.clip(t, 0, None) / lam(med, cure, k)) ** k)
def sampNC(med, cure, k, L, u):                                  # sample a NON-cured Weibull time
    return (lam(med, cure, k) / L) * (-np.log(u)) ** (1.0 / k)
Sll   = lambda t, al, be: 1.0 / (1.0 + (np.clip(t, 1e-9, None) / al) ** be)   # log-logistic survival
sampLL = lambda al, be, u: al * (1.0 / u - 1.0) ** (1.0 / be)                 # log-logistic sample

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
               maxStretch=1.5, comp=[dict(c) for c in DEFAULT_COMP],
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
    WT = np.array([1.0, 2.0, 4.0])
    return w, cm, coh, MT, MOBS, WT

def median(S):
    if S(900.0) >= 0.5: return np.inf
    lo, hi = 0.01, 900.0
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if S(m) > 0.5: lo = m
        else: hi = m
    return 0.5 * (lo + hi)

# ---------------------------------------------------------------- cure-mixture model
def build_cure(cfg):
    w, cm, coh, MT, MOBS, WT = common(cfg)
    fnr = cfg["fnr"]
    pibat = sum(w[i] * cm[i]["cure"] for i in range(len(cm)))
    obs = cm[0]
    def Sbat(t, L): return sum(w[i] * Sc(t, cm[i]["med"], cm[i]["cure"], cm[i]["k"], L) for i in range(len(cm)))
    def Snc(t, L):  return (Sbat(t, L) - pibat) / (1 - pibat)
    def Spool(t, pr, L):
        return 0.5 * Sbat(t, L) + 0.5 * ((1 - fnr) * (pr + (1 - pr) * Snc(t, L))
                                         + fnr * Sc(t, obs["med"], obs["cure"], obs["k"], L))
    def ed(T, pr, L):
        return sum(c[1] * (1 - Spool(T - c[0], pr, L)) for c in coh if c[0] <= T)

    Lmin, Lmax = 1.0 / (cfg.get("maxStretch") or 3.0), 2.2
    best, bs = (0.6, min(1.0, Lmax)), 1e18
    for ki in range(45):
        for li in range(31):
            pr = ki / 45.0; L = Lmin + li / 30.0 * (Lmax - Lmin)
            e = sum(WT[j] * (ed(MT[j], pr, L) - MOBS[j]) ** 2 for j in range(3))
            if e < bs: bs, best = e, (pr, L)
    for it in range(3):
        p0, l0 = best; st = 0.06 / (it + 1)
        for dp in range(-3, 4):
            for dl in range(-3, 4):
                pr = min(0.985, max(0.0, p0 + dp * st))
                L = max(Lmin, min(Lmax, l0 + dl * st * 4))
                e = sum(WT[j] * (ed(MT[j], pr, L) - MOBS[j]) ** 2 for j in range(3))
                if e < bs: bs, best = e, (pr, L)
    presp, L = best
    pgps = (1 - fnr) * presp + fnr * obs["cure"]
    Sb = lambda t: Sbat(t, L)
    Sg = lambda t: (1 - fnr) * (presp + (1 - presp) * Snc(t, L)) + fnr * Sc(t, obs["med"], obs["cure"], obs["k"], L)
    Sp = lambda t: Spool(t, presp, L)
    return dict(kind="cure", cfg=cfg, w=w, cm=cm, coh=coh, MT=MT, MOBS=MOBS,
                presp=presp, L=L, pibat=pibat, pgps=pgps, obs=obs,
                Sbat=Sb, Sgps=Sg, Spool=Sp,
                batMed=median(Sb), batMedRaw=median(lambda t: Sbat(t, 1.0)),
                gpsMed=median(Sg), poolMed=median(Sp),
                poolCure=0.5 * (pibat + pgps), ed=lambda t: ed(t, presp, L))

# ---------------------------------------------------------------- log-logistic (no plateau)
def build_ll(cfg):
    w, cm, coh, MT, MOBS, WT = common(cfg)
    fnr, beta = cfg["fnr"], cfg["beta"]
    mC = sum(w[i] * cm[i]["med"] for i in range(len(cm)))
    mObs = cm[0]["med"]
    def Spool(t, k, r):
        mB = k * mC
        return 0.5 * Sll(t, mB, beta) + 0.5 * ((1 - fnr) * Sll(t, r * mB, beta) + fnr * Sll(t, k * mObs, beta))
    def ed(T, k, r):
        return sum(c[1] * (1 - Spool(T - c[0], k, r)) for c in coh if c[0] <= T)

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
    Sb = lambda t: Sll(t, mB, beta)
    Sg = lambda t: (1 - fnr) * Sll(t, r * mB, beta) + fnr * Sll(t, k * mObs, beta)
    Sp = lambda t: 0.5 * Sb(t) + 0.5 * Sg(t)
    return dict(kind="ll", cfg=cfg, w=w, cm=cm, coh=coh, MT=MT, MOBS=MOBS,
                beta=beta, k=k, r=r, mB=mB, mObs=k * mObs,
                Sbat=Sb, Sgps=Sg, Spool=Sp,
                batMed=mB, gpsMed=r * mB, ratio=r, poolMed=median(Sp),
                ed=lambda t: ed(t, k, r))

# ---------------------------------------------------------------- shared Monte-Carlo
def mc(M, nsim=1500, seed=987654321):
    """Enrollment -> per-arm death draws -> censor at FINAL-th event -> log-rank test.
    Returns dict(ps, reach, medHR): P(significant), fraction reaching the trigger, median HR."""
    cfg = M["cfg"]; N, FINAL, HRC, fnr = cfg["N"], cfg["FINAL"], cfg["HRC"], cfg["fnr"]
    ZC = abs(np.log(HRC)) * np.sqrt(FINAL) / 2.0
    rng = np.random.default_rng(seed)
    coh, w, cm = M["coh"], M["w"], M["cm"]
    cohp = coh[:, 1] / coh[:, 1].sum()                              # cohort enrollment probs
    ncw = np.array([w[i] * (1 - cm[i]["cure"]) for i in range(len(cm))])
    ncw = ncw / ncw.sum()                                          # BAT non-cured component mix
    n1 = N // 2
    sig = reached = 0; hrs = []

    def draw_cure_bat(n):
        out = np.empty(n)
        pick = rng.choice(len(cm), size=n, p=w)
        for j, c in enumerate(cm):
            idx = np.where(pick == j)[0]
            if idx.size:
                cured = rng.random(idx.size) < c["cure"]
                s = sampNC(c["med"], c["cure"], c["k"], M["L"], rng.random(idx.size))
                out[idx] = np.where(cured, 1e9, s)
        return out

    def draw_cure_gps(n):
        out = np.empty(n)
        isnr = rng.random(n) < fnr
        nr = np.where(isnr)[0]; rs = np.where(~isnr)[0]
        obs = M["obs"]
        if nr.size:
            cured = rng.random(nr.size) < obs["cure"]
            s = sampNC(obs["med"], obs["cure"], obs["k"], M["L"], rng.random(nr.size))
            out[nr] = np.where(cured, 1e9, s)
        if rs.size:
            cured = rng.random(rs.size) < M["presp"]
            pick = rng.choice(len(cm), size=rs.size, p=ncw)
            s = np.empty(rs.size)
            for j, c in enumerate(cm):
                jj = np.where(pick == j)[0]
                if jj.size: s[jj] = sampNC(c["med"], c["cure"], c["k"], M["L"], rng.random(jj.size))
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
            u = rng.random(N)
            nr = a1 & (rng.random(N) < fnr)
            surv[a0] = sampLL(mB, beta, u[a0])
            surv[a1 & ~nr] = sampLL(r * mB, beta, u[a1 & ~nr])
            surv[nr] = sampLL(mObs, beta, u[nr])
        dcal = en + surv
        fin = np.sort(dcal[dcal < 1e8])
        if fin.size < FINAL: continue
        reached += 1
        t80 = fin[FINAL - 1]
        ev = (dcal <= t80).astype(int)
        time = np.minimum(surv, np.clip(t80 - en, 0, None))
        # log-rank score = Cox score test (the trial's significance test)
        idx = np.argsort(time, kind="mergesort")
        totX = arm.sum(); prefX = 0; num = 0.0; varr = 0.0
        for p in range(N):
            i = idx[p]; nAt = N - p; sx = totX - prefX
            if ev[i] == 1:
                pb = sx / nAt; num += arm[i] - pb; varr += pb * (1 - pb)
            prefX += arm[i]
        if varr > 0:
            z = -num / np.sqrt(varr)
            if z > ZC: sig += 1
            hrs.append(np.exp(num / varr))
    hrs.sort()
    return dict(ps=(sig / reached if reached else 0.0),
                reach=reached / nsim,
                medHR=(hrs[len(hrs) // 2] if hrs else np.nan))

# ---------------------------------------------------------------- figure
NAVY = "#0b2545"; RED = "#9e2b25"; TEAL = "#197278"; GREY = "#6b6f72"; ORANGE = "#e8910b"

def figure(path, nsim=1500):
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140})
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2)); tg = np.linspace(0, 48, 300)

    # (a) survival curves for the base preset, plateau (cure) shape
    cfg = apply_preset(default_cfg(), "base")
    Mc = build_cure(cfg)
    a = ax[0]
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
    b = ax[1]
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
    c = ax[2]; x = np.arange(len(names))
    c.bar(x - 0.19, gc, 0.36, color=NAVY, label="Plateau")
    c.bar(x + 0.19, gl, 0.36, color=ORANGE, label="No-plateau")
    c.set_xticks(x); c.set_xticklabels(labels); c.set_ylim(0, 103)
    c.set_ylabel("P(success) %")
    c.set_title("(c) The plateau-vs-tail gap is the irreducible uncertainty",
                fontweight="bold", fontsize=9); c.legend(fontsize=7.6)

    fig.suptitle("REGAL Scenario Explorer — two survival shapes fit the same blinded milestones; "
                 "their P(success) gap is the 'is the plateau real?' uncertainty.",
                 fontweight="bold", fontsize=10.5, y=1.02)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    return path

# ---------------------------------------------------------------- CLI
def fmt_med(m): return "NR" if not np.isfinite(m) else f"{m:.0f}mo"

if __name__ == "__main__":
    NSIM = 800   # matches the html's interactive budget (~600); raise for tighter MC error
    base = apply_preset(default_cfg(), "base")
    Mc, Ml = build_cure(base), build_ll(base)
    rc, rl = mc(Mc, NSIM), mc(Ml, NSIM)
    print("REGAL Scenario Explorer (base preset, f_nr=20%)")
    print(f"  BAT  : cure {100*Mc['pibat']:.0f}%  median {fmt_med(Mc['batMedRaw'])}->{fmt_med(Mc['batMed'])}  @36mo {100*Mc['Sbat'](36):.0f}%")
    print(f"  GPS  : cure {100*Mc['pgps']:.0f}%  median {fmt_med(Mc['gpsMed'])}  (cure gap +{100*(Mc['pgps']-Mc['pibat']):.0f}pp)")
    print(f"  calib: survival {1/Mc['L']:.2f}x inputs  poolMed {fmt_med(Mc['poolMed'])}")
    edv = [Mc['ed'](t) for t in Mc['MT']]
    print(f"  fit  : modeled deaths {'/'.join(f'{x:.0f}' for x in edv)}  vs observed {'/'.join(f'{x:.0f}' for x in Mc['MOBS'])}")
    print(f"\n  P(success) PLATEAU (cure-mixture) : {100*rc['ps']:.0f}%   medHR {rc['medHR']:.2f}   reached {100*rc['reach']:.0f}%")
    print(f"  P(success) NO-PLATEAU (log-logistic β={base['beta']:.2f}): {100*rl['ps']:.0f}%   medHR {rl['medHR']:.2f}")
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
