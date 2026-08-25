"""REGAL legacy scenario explorer — Python engine (port of regal_explorer.html).

The blinded milestones (60/72/78 deaths) constrain assumed pooled survival families;
the arm split remains assumption-driven. The engine compares two fixed scenarios
using an identical BAT arm and changing only the GPS responder component:

  * plateau (GPS cure)  — GPS responders get a durable-remission plateau (cure-mixture),
  * bounded no-GPS-cure alternative — responders use a fitted no-cure Weibull.

GPS non-responders track Observation in both scenarios. The bounded alternative emits
a fit status: A for a boundary/non-identified fit, B for a residual misfit, and C for
an adequate interior fit. These are diagnostics, not formal hypothesis-test results.
The `ps` output is a fixed-scenario simulated rejection rate. The survival, fitting,
and legacy final-Monte-Carlo outputs mirror the JavaScript in regal_explorer.html:

  enroll · common · bat_arm · build_plateau · build_no_gps_cure · median · chart(figure)

Python `mc()` additionally exposes four audit-only interim-efficacy fields used by
`audit/interim_efficacy_replay.py`. The browser retains the v1 Monte-Carlo outputs and
does not compute that boundary diagnostic; full Python/HTML parity remains v2 WP8 work.

Research/analysis tool, not investment advice.

IMPORTANT: this v1 engine reports a fixed-scenario simulated rejection rate. It does
not condition on the observed interim continuation and is not a posterior forecast
for the ongoing REGAL trial. See V2_IMPLEMENTATION_PLAN.md for the rebuild.
"""
import os
import numpy as np
from datetime import date, timedelta
from concurrent.futures import ProcessPoolExecutor
from trial_design import obrien_fleming_two_look
# matplotlib is imported lazily inside figure() — it's only ever used there (the main
# process), and worker processes spawned for _mc_task() must not pay its import cost.

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
# Shared responder family used by BOTH panels: a Weibull (shape<1 = heavier tail, monotone
# non-increasing hazard). The plateau panel wraps it in a cured fraction (Sc above); the
# bounded no-GPS-cure alternative uses it bare for GPS responders. wscale maps a median to the scale (S(median)=0.5).
def Sweib(t, scale, shape):                                      # bare Weibull survival
    return np.exp(-(np.clip(t, 0, None) / scale) ** shape)
sampWeib = lambda scale, shape, u: scale * (-np.log(u)) ** (1.0 / shape)      # inverse-transform Weibull sample
wscale   = lambda med, shape: med / (np.log(2.0)) ** (1.0 / shape)            # median -> Weibull scale

# ---------------------------------------------------------------- eligibility selection (gamma frailty)
# Trial eligibility enrols a healthier subset than the unselected population the component medians
# describe. It screens on BASELINE covariates that merely CORRELATE with survival, never on the
# realized death time, so the enrolled curve must retain positive hazard from t=0.
#
# Population frailty Z ~ Gamma(mean 1, variance theta) multiplies the UNCURED Weibull hazard.
# Eligibility accepts a patient with probability proportional to exp(-beta Z): frailer patients are
# exponentially less likely to pass screening. Gamma is conjugate to that tilt, so the enrolled
# cohort is again Gamma with the same shape 1/theta and a smaller scale, and the whole model stays
# closed-form. Fixing the overall acceptance rate at 1-q gives beta = ((1-q)^-theta - 1)/theta and
#
#     theta_sel = theta (1-q)^theta,      E[Z | eligible] = (1-q)^theta.
#
# Selection therefore rescales the uncured hazard; it does NOT change the cure fraction (an uncured
# patient cannot be screened into being cured) and it does NOT alter component composition weights.
# theta -> 0 collapses every formula below to v1's original unselected Weibull: with no unobserved
# heterogeneity there is nothing for eligibility to select on, so no enrichment is possible at any q.
def fsel(theta, q):
    """Enrolled-cohort gamma scale parameter theta_sel."""
    if theta <= 0: return 0.0
    return theta * (1.0 - min(max(q, 0.0), 0.999)) ** theta

def lamf(med, cure, k, theta):
    """Weibull scale anchored so the POPULATION (q=0) marginal median is `med`.

    The published component medians are already marginal over each source study's patient
    heterogeneity, so the frailty mixture must be re-anchored to reproduce them at zero selection.
    Anchoring the conditional (frailty=1) curve instead would double-count that heterogeneity."""
    if theta <= 0: return lam(med, cure, k)
    A = (0.5 - cure) / (1.0 - cure)
    return med / (((A ** (-theta)) - 1.0) / theta) ** (1.0 / k)

def Scf(t, med, cure, k, theta, q):
    """Cure-mixture survival for the ENROLLED cohort under gamma-frailty eligibility selection."""
    s = (np.clip(t, 0, None) / lamf(med, cure, k, theta)) ** k
    if theta <= 0: return cure + (1 - cure) * np.exp(-s)
    return cure + (1 - cure) * (1.0 + fsel(theta, q) * s) ** (-1.0 / theta)

def sampf(n, theta, q, rng):
    """Draw enrolled-cohort frailties (all ones when theta = 0)."""
    if theta <= 0: return np.ones(n)
    return rng.gamma(1.0 / theta, fsel(theta, q), size=n)

def sampNCf(med, cure, k, theta, u, z):
    """Uncured event time given frailty z: S(t|z) = exp(-z (t/lam)^k)."""
    return lamf(med, cure, k, theta) * (-np.log(u) / z) ** (1.0 / k)
# natural (non-disease) all-cause mortality as an independent competing risk.
# ndr is an annual death fraction; convert to a constant monthly hazard.
natH  = lambda p: (-np.log(1.0 - p) / 12.0) if p > 0 else 0.0                 # monthly hazard from annual fraction
Snat  = lambda t, h: np.exp(-h * np.clip(t, 0, None))                         # background survival factor

OBS_FRAC_N = 10   # quadrature points for the hd>0 branch below; build_no_gps_cure's vectorized
                  # resid_grid() mirrors this math and must stay in lockstep on this constant

def obs_frac(S, tau, hd, n=OBS_FRAC_N):
    """Fraction of a cohort *observed* dead by tau under independent exponential censoring
    (loss-to-follow-up hazard hd). With hd=0 this is just the death CDF 1-S(tau)."""
    if tau <= 0: return 0.0
    if hd <= 0: return 1.0 - S(tau)
    ts = np.linspace(0.0, tau, n + 1)
    f = (1.0 - S(ts)) * np.exp(-hd * ts)
    integ = (tau / n) * (f.sum() - 0.5 * (f[0] + f[-1]))          # trapezoid ∫ (1-S) e^{-hd t} dt
    return float(np.exp(-hd * tau) * (1.0 - S(tau)) + hd * integ)

# ---------------------------------------------------------------- defaults
# Base-case component parameters re-derived from the CR2 transplant-ineligible
# mixture-cure survival review (VIALE-A / QUAZAR / r-r salvage, discounted to CR2)
# and the US/EU/China BAT-composition review. Recommended base weights: observation
# 35% (split 27/8 obs/hydroxyurea), venetoclax 35%, HMA 22%, LDAC 8%. Per-component
# (median OS, cure, Weibull k): observation 6mo/3%/1.1, HMA 12mo/10%/1.0,
# venetoclax 12mo/15%/k=0.78 (published VEN+AZA decreasing-hazard tail), LDAC 7mo/8%/1.1.
DEFAULT_COMP = [
    {"name": "Observation",  "w": 27, "med": 6.0,  "cure": 3,  "k": 1.1},
    {"name": "Hydroxyurea",  "w": 8,  "med": 5.0,  "cure": 2,  "k": 1.1},
    {"name": "HMA",          "w": 22, "med": 12.0, "cure": 10, "k": 1.0},
    {"name": "Venetoclax",   "w": 35, "med": 12.0, "cure": 15, "k": 0.78},
    {"name": "LDAC",         "w": 8,  "med": 7.0,  "cure": 8,  "k": 1.1},
]
DEFAULT_EV = [
    {"label": "60 events", "y": 2024, "m": 12, "d": 10, "n": 60},
    {"label": "72 events", "y": 2025, "m": 12, "d": 26, "n": 72},
    {"label": "78 events", "y": 2026, "m": 5,  "d": 11, "n": 78},
]
PRESETS = {
    "base": {"w": [27, 8, 22, 35, 8], "vc": 15},
    "low":  {"w": [33, 12, 30, 15, 10], "vc": 15},   # observation-heavy / venetoclax-light (access-constrained)
    "dom":  {"w": [8, 4, 18, 60, 10],  "vc": 15},    # venetoclax-dominant (US-heavy uptake)
    "bear": {"w": [5, 3, 12, 70, 10],  "vc": 25},    # max venetoclax + top-of-range cure (strongest BAT)
    "bull": {"w": [40, 10, 25, 15, 10], "vc": 10},   # observation-heavy weak-BAT corner (optimistic for GPS)
}

def default_cfg(**over):
    cfg = dict(N=126, FINAL=80, HRC=0.636, fnr=0.20, bl=0.50, shape=0.60, shapeOverride=False,
               ndr=0.02, IA=60, futHR=1.0, drop=0.0, esel=0.25, fvar=1.43, unweighted=False,
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

# ---------------------------------------------------------------- shared BAT arm (both panels)
def bat_arm(cfg):
    """The BAT arm construction shared byte-for-byte by BOTH panels. The two panels are literally
    one biological lever apart: same per-component medians, cures, shapes and left-truncation."""
    w, cm, coh, MT, MOBS, WT = common(cfg)
    h = natH(cfg.get("ndr", 0.0)); hd = natH(cfg.get("drop", 0.0))
    F = min(max(cfg.get("esel", 0.0), 0.0), 0.5)   # eligibility screen-out fraction q
    TH = max(cfg.get("fvar", 0.0) or 0.0, 0.0)     # population frailty variance theta
    # Eligibility selects on baseline frailty, never on the realized death time, so the enrolled
    # curve keeps positive hazard at t=0 (no guarantee interval) and the cure fraction is unchanged.
    def Ssel(t, c):
        return Scf(t, c["med"], c["cure"], c["k"], TH, F)
    pibat = sum(w[i] * cm[i]["cure"] for i in range(len(cm)))
    obs = cm[0]
    def Sbat(t): return sum(w[i] * Ssel(t, cm[i]) for i in range(len(cm)))
    def Snc(t):  return (Sbat(t) - pibat) / (1 - pibat)   # non-cured BAT shape (plateau panel only)
    return dict(w=w, cm=cm, coh=coh, MT=MT, MOBS=MOBS, WT=WT, h=h, hd=hd, F=F, TH=TH,
                Ssel=Ssel, pibat=pibat, obs=obs, Sbat=Sbat, Snc=Snc)

# ---------------------------------------------------------------- plateau (GPS-cure) model
def build_plateau(cfg):
    # "plateau"/"cure" here means the GPS-cure model: GPS responders get a durable-remission plateau.
    B = bat_arm(cfg)
    w, cm, coh, MT, MOBS, WT = B["w"], B["cm"], B["coh"], B["MT"], B["MOBS"], B["WT"]
    h, hd, F, TH, Ssel, pibat, obs, Sbat, Snc = (B["h"], B["hd"], B["F"], B["TH"], B["Ssel"],
                                                 B["pibat"], B["obs"], B["Sbat"], B["Snc"])
    fnr = cfg["fnr"]
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
    pgps = (1 - fnr) * presp + fnr * obs["cure"]
    # returned curves are all-cause (disease x background mortality).
    Sb = lambda t: Sbat(t) * Snat(t, h)
    Sg = lambda t: ((1 - fnr) * (presp + (1 - presp) * Snc(t)) + fnr * Ssel(t, obs)) * Snat(t, h)
    Sp = lambda t: Spool(t, presp) * Snat(t, h)
    return dict(kind="plateau", cfg=cfg, w=w, cm=cm, coh=coh, MT=MT, MOBS=MOBS, WT=WT,
                presp=presp, h=h, pibat=pibat, pgps=pgps, obs=obs,
                Sbat=Sb, Sgps=Sg, Spool=Sp, ed_raw=ed,
                batMed=median(Sb), gpsMed=median(Sg), poolMed=median(Sp),
                poolCure=0.5 * (pibat + pgps), ed=lambda t: ed(t, presp))

# ----------------------- bounded no-GPS-cure alternative (shared BAT + no-cure GPS responder)
def build_no_gps_cure(cfg):
    """Identical BAT to the plateau panel; GPS responders swap the cure-mixture for a fitted no-cure
    Weibull (median mG, tail shape sG); GPS non-responders (fnr) still track Observation in BOTH
    panels. Emits a bounded-fit diagnostic: A=boundary/non-identified, B=residual misfit,
    C=adequate interior fit. It does not test or establish a biological cure mechanism."""
    B = bat_arm(cfg)
    w, cm, coh, MT, MOBS, WT = B["w"], B["cm"], B["coh"], B["MT"], B["MOBS"], B["WT"]
    h, hd, F, TH, Ssel, pibat, obs, Sbat = (B["h"], B["hd"], B["F"], B["TH"], B["Ssel"],
                                            B["pibat"], B["obs"], B["Sbat"])
    fnr = cfg["fnr"]
    bat_med = median(lambda t: Sbat(t) * Snat(t, h))
    fit_shape = not cfg.get("shapeOverride", False)       # AUTO fits sG; override holds the slider value fixed
    MGLO = min(bat_med if np.isfinite(bat_med) else 60.0, 110.0); MGHI = 120.0; SGMIN, SGMAX = 0.15, 1.5
    # GPS responder = a single NO-CURE Weibull under the same eligibility selection as BAT.
    def Sresp(t, mG, sG): return Scf(t, mG, 0.0, sG, TH, F)
    # GPS non-responders (fnr) track Observation — unchanged and identical to the plateau panel.
    def Sgps(t, mG, sG): return (1 - fnr) * Sresp(t, mG, sG) + fnr * Ssel(t, obs)
    def Spool(t, mG, sG): return 0.5 * Sbat(t) + 0.5 * Sgps(t, mG, sG)
    def ed(T, mG, sG):
        Sf = lambda t: Spool(t, mG, sG) * Snat(t, h)
        return sum(c[1] * obs_frac(Sf, T - c[0], hd) for c in coh if c[0] <= T)

    def Spool_grid(t, mG, sG):
        """Spool(), but mG/sG are 1D arrays of candidate values (a fit-grid) instead of scalars;
        t may be any-shaped array. Returns t.shape + (len(mG),) via numpy broadcasting on a
        trailing grid axis, so the whole candidate grid is evaluated in one vectorized pass."""
        t = np.asarray(t, dtype=float); te = t[..., None]
        Sr = Scf(te, mG, 0.0, sG, TH, F)
        Sg = (1 - fnr) * Sr + fnr * Ssel(t, obs)[..., None]
        return 0.5 * Sbat(t)[..., None] + 0.5 * Sg

    def resid_grid(mG, sG):
        """The milestone-fit residual (WT-weighted squared miss on ed() at each of the 3
        milestones), evaluated across an entire (mG, sG) candidate grid in one vectorized pass.
        The fit loops below need this at ~600-800 grid points; doing that one scalar call at a
        time (walking ~40 cohort rows through obs_frac()'s tiny numpy ops per call) made the fit
        alone cost seconds — per-call overhead dominates at that array size. This mirrors
        obs_frac()'s fast/quadrature-path math exactly, just batched over the whole grid."""
        ngrid = len(mG); e = np.zeros(ngrid)
        for T, Mo, Wt in zip(MT, MOBS, WT):
            keep = coh[:, 0] < T                        # tau = T - c[0] > 0, same gate as obs_frac's tau<=0 check
            if not np.any(keep):
                e += Wt * Mo ** 2
                continue
            tau = T - coh[keep, 0]; wts = coh[keep, 1]
            if hd <= 0:
                frac = 1.0 - Spool_grid(tau, mG, sG) * Snat(tau, h)[:, None]
            else:
                n = OBS_FRAC_N; qf = np.linspace(0.0, 1.0, n + 1)
                ts = tau[:, None] * qf[None, :]                                    # (K, n+1)
                Sf3 = Spool_grid(ts, mG, sG) * Snat(ts, h)[..., None]              # (K, n+1, ngrid)
                f = (1.0 - Sf3) * np.exp(-hd * ts)[..., None]
                integ = (tau / n)[:, None] * (f.sum(axis=1) - 0.5 * (f[:, 0, :] + f[:, -1, :]))
                frac = np.exp(-hd * tau)[:, None] * (1.0 - Sf3[:, -1, :]) + hd * integ
            e += Wt * ((wts[:, None] * frac).sum(axis=0) - Mo) ** 2
        return e

    # §2/§3: fit the GPS responder median mG and (auto) tail shape sG to the 3 milestones. §3: BAT is
    # FIXED on purpose here while varying the GPS responder family; sG is free to go heavy.
    sgN = 18 if fit_shape else 0
    best, bs = (min(MGHI, (bat_med or 12.0) * 2), 0.6 if fit_shape else cfg["shape"]), 1e18
    mgs = MGLO + (MGHI - MGLO) * np.arange(31) / 30.0
    sgs = (SGMIN + (SGMAX - SGMIN) * np.arange(sgN + 1) / (sgN or 1)) if fit_shape else np.array([cfg["shape"]])
    MGg, SGg = np.meshgrid(mgs, sgs, indexing="ij")                # mG outer, sG inner — matches the old nested loop order
    e_grid = resid_grid(MGg.ravel(), SGg.ravel())
    k = int(np.argmin(e_grid))
    if e_grid[k] < bs: bs, best = float(e_grid[k]), (float(MGg.ravel()[k]), float(SGg.ravel()[k]))
    for it in range(4):
        m0, s0 = best; st = 1.0 / (it + 1)
        dms = np.arange(-3, 4); dss = np.arange(-3, 4) if fit_shape else np.array([0])
        DM, DS = np.meshgrid(dms, dss, indexing="ij")
        mgc = np.clip(m0 + DM.ravel() * 1.2 * st, MGLO, MGHI)
        sgc = np.clip(s0 + DS.ravel() * 0.04 * st, SGMIN, SGMAX) if fit_shape else np.full(DM.size, s0)
        e_ref = resid_grid(mgc, sgc)
        k = int(np.argmin(e_ref))
        if e_ref[k] < bs: bs, best = float(e_ref[k]), (float(mgc[k]), float(sgc[k]))
    mG, sG = best
    edv = [ed(t, mG, sG) for t in MT]
    rms_resid = float(np.sqrt(sum((edv[i] - MOBS[i]) ** 2 for i in range(3)) / 3.0))
    max_off = float(max(abs(edv[i] - MOBS[i]) for i in range(3)))
    # §5 boundary detection on the GPS knobs (mG cap/track, sG edges).
    mg_cap = mG >= MGHI - 0.5; mg_floor = mG <= MGLO + 0.5
    sg_heavy = fit_shape and sG <= SGMIN + 0.01; sg_light = fit_shape and sG >= SGMAX - 0.01
    mg_track = False
    if mg_cap:                                             # raise the mG cap; if the fit tracks it, mG is unidentified
        MGHI2 = MGHI * 1.6
        mms = MGLO + (MGHI2 - MGLO) * np.arange(21) / 20.0
        e2 = resid_grid(mms, np.full(21, sG))
        m2b = float(mms[int(np.argmin(e2))])
        mg_track = m2b > MGHI + 1.0
    # §5 fit status. All non-interior fits are non-identified (State A) with no reported scenario
    # rate. The legacy cureReq field only separates upper/heavy from light-edge boundaries; neither
    # subtype is a formal test of a biological mechanism. RMS-based tolerance ensures the
    # weighted fit's deliberate middle-milestone trade-off does not by itself trip State B.
    RMS_TOL, OFF_TOL = 2.0, 3.0
    cure_bound = mg_cap or mg_track or sg_heavy
    cure_req = False
    if cure_bound:
        state = "A"; cure_req = True
        reason = (("GPS median runs to its %dmo cap%s — the bounded fit is non-identified" % (MGHI, " and tracks a raised cap" if mg_track else ""))
                  if (mg_cap or mg_track) else
                  ("tail shape pinned at the heavy edge (%.2f) — a near-degenerate tail faking the plateau" % SGMIN))
    elif sg_light:
        state = "A"
        reason = ("GPS tail pinned at the light edge (%.2f): the milestones want an even lighter "
                  "(sharper, increasing-hazard) responder tail, so the no-cure fit is unidentified here. This is "
                  "this boundary does not identify whether a GPS-specific cure exists" % SGMAX)
    elif rms_resid > RMS_TOL or max_off > OFF_TOL:
        state = "B"
        reason = "residual RMS %.1f (modeled %s vs %s)" % (rms_resid, "/".join("%.0f" % x for x in edv), "/".join("%.0f" % x for x in MOBS))
    else:
        state = "C"
        reason = ("GPS ~= BAT — essentially no GPS separation needed (median %.0fmo)" % mG) if mg_floor \
                 else ("interior fit (median %.0fmo, tail shape %.2f)" % (mG, sG))
    degenerate = state in ("A", "B")
    Sb = lambda t: Sbat(t) * Snat(t, h)
    Sg = lambda t: Sgps(t, mG, sG) * Snat(t, h)
    Sp = lambda t: Spool(t, mG, sG) * Snat(t, h)
    return dict(kind="nogpscure", cfg=cfg, w=w, cm=cm, coh=coh, MT=MT, MOBS=MOBS, WT=WT,
                h=h, pibat=pibat, obs=obs, fnr=fnr, mG=mG, sG=sG, fitShape=fit_shape,
                batMed=bat_med, ratio=(mG / bat_med if np.isfinite(bat_med) else np.nan),
                edv=edv, rmsResid=rms_resid, state=state, reason=reason,
                cureReq=cure_req, degenerate=degenerate, ed_raw=ed,
                Sbat=Sb, Sgps=Sg, Spool=Sp,
                gpsMed=median(Sg), ed=lambda t: ed(t, mG, sG))

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
    ``ps`` is the legacy fixed-scenario final rejection rate conditional on reaching
    FINAL. The return value also exposes an unconditional interim efficacy-crossing
    diagnostic using the committed two-look O'Brien-Fleming boundary."""
    cfg = M["cfg"]; N, FINAL, HRC, fnr = cfg["N"], cfg["FINAL"], cfg["HRC"], cfg["fnr"]
    FINAL = max(2, int(FINAL))                                    # engine-level guard for notebook/API callers
    h = natH(cfg.get("ndr", 0.0))                                  # background mortality competing risk (an event)
    hdrop = natH(cfg.get("drop", 0.0))                             # loss-to-follow-up (censoring, not an event)
    ZC = abs(np.log(HRC)) * np.sqrt(FINAL) / 2.0
    rng = np.random.default_rng(seed)
    coh, w, cm = M["coh"], M["w"], M["cm"]
    cohp = coh[:, 1] / coh[:, 1].sum()                              # cohort enrollment probs
    F = min(max(cfg.get("esel", 0.0), 0.0), 0.5)                    # eligibility screen-out fraction q
    TH = max(cfg.get("fvar", 0.0) or 0.0, 0.0)                      # population frailty variance theta
    ncw = np.array([w[i] * (1 - cm[i]["cure"]) for i in range(len(cm))])
    ncw = ncw / ncw.sum()                                          # BAT non-cured component mix
    n1 = N // 2
    IA = max(1, min(int(cfg.get("IA", 60)), FINAL - 1))            # interim-analysis event count
    futHR = cfg.get("futHR", 1.0)                                  # interim futility HR threshold
    ia_design = obrien_fleming_two_look(0.025, IA / FINAL)
    z_ia_efficacy = ia_design["interim_z"]
    sig = reached = ia_reached = ia_efficacy = 0
    hrs = []; hrsIA = []; aliveG = aliveB = 0.0

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
                cured = rng.random(idx.size) < c["cure"]
                z = sampf(idx.size, TH, F, rng)
                s = sampNCf(c["med"], c["cure"], c["k"], TH, rng.random(idx.size), z)
                out[idx] = np.where(cured, 1e9, s)
        return out

    def draw_cure_gps(n):
        out = np.empty(n)
        isnr = rng.random(n) < fnr
        nr = np.where(isnr)[0]; rs = np.where(~isnr)[0]
        obs = M["obs"]
        if nr.size:
            cured = rng.random(nr.size) < obs["cure"]
            z = sampf(nr.size, TH, F, rng)
            s = sampNCf(obs["med"], obs["cure"], obs["k"], TH, rng.random(nr.size), z)
            out[nr] = np.where(cured, 1e9, s)
        if rs.size:
            cured = rng.random(rs.size) < M["presp"]
            pick = rng.choice(len(cm), size=rs.size, p=ncw)
            s = np.empty(rs.size)
            for j, c in enumerate(cm):
                jj = np.where(pick == j)[0]
                if jj.size:
                    z = sampf(jj.size, TH, F, rng)
                    s[jj] = sampNCf(c["med"], c["cure"], c["k"], TH, rng.random(jj.size), z)
            out[rs] = np.where(cured, 1e9, s)
        return out

    def draw_nogpscure_gps(n):   # BAT + GPS non-responder identical to draw_cure_gps; responder = NO-CURE Weibull
        out = np.empty(n)
        isnr = rng.random(n) < fnr
        nr = np.where(isnr)[0]; rs = np.where(~isnr)[0]
        obs = M["obs"]
        if nr.size:
            cured = rng.random(nr.size) < obs["cure"]
            z = sampf(nr.size, TH, F, rng)
            s = sampNCf(obs["med"], obs["cure"], obs["k"], TH, rng.random(nr.size), z)
            out[nr] = np.where(cured, 1e9, s)
        if rs.size:
            z = sampf(rs.size, TH, F, rng)
            out[rs] = sampNCf(M["mG"], 0.0, M["sG"], TH, rng.random(rs.size), z)
        return out

    for _ in range(nsim):
        arm = rng.permutation(np.r_[np.ones(n1), np.zeros(N - n1)]).astype(int)
        en = coh[rng.choice(len(coh), size=N, p=cohp), 0]
        surv = np.empty(N)
        a1 = arm == 1; a0 = ~a1
        if M["kind"] == "plateau":
            surv[a1] = draw_cure_gps(a1.sum())
            surv[a0] = draw_cure_bat(a0.sum())
        else:   # nogpscure: BAT + GPS non-responder identical to plateau; GPS responder = no-cure Weibull
            surv[a0] = draw_cure_bat(a0.sum())
            surv[a1] = draw_nogpscure_gps(a1.sum())
        if h > 0:                                                   # natural death may preempt disease death
            surv = np.minimum(surv, -np.log(rng.random(N)) / h)
        # loss-to-follow-up: an independent censoring time; if it precedes death the patient is censored
        td = (-np.log(rng.random(N)) / hdrop) if hdrop > 0 else np.full(N, np.inf)
        isdeath = surv <= td                                        # a death is observed only if it precedes dropout
        obsT = np.minimum(surv, td)                                 # follow-up time (death or censoring)
        rawcal = en + surv                                          # death calendar ignoring dropout (alive-count basis)
        dcal = np.where(isdeath, en + surv, 1e9)                    # event calendar feeding the trigger
        fin = np.sort(dcal[dcal < 1e8])
        if fin.size < IA: continue
        ia_reached += 1
        tIA = fin[IA - 1]
        evIA = (isdeath & (dcal <= tIA)).astype(int)
        timeIA = np.minimum(obsT, np.clip(tIA - en, 0, None))
        numIA, varrIA = score(timeIA, evIA)
        hr_ia_trial = None
        if varrIA > 0:
            zIA = -numIA / np.sqrt(varrIA)
            ia_efficacy += int(zIA >= z_ia_efficacy)
            hr_ia_trial = np.exp(numIA / varrIA)
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
        # Preserve v1's median-IA-HR conditioning on trials that also reach FINAL.
        if hr_ia_trial is not None: hrsIA.append(hr_ia_trial)
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
                reach_IA=ia_reached / nsim,
                p_IA_efficacy=ia_efficacy / nsim,
                p_IA_efficacy_given_reach=(ia_efficacy / ia_reached if ia_reached else 0.0),
                z_IA_efficacy=z_ia_efficacy,
                aliveG=(aliveG / reached if reached else np.nan),
                aliveB=(aliveB / reached if reached else np.nan))

# ---------------------------------------------------------------- parallel batch execution
def _mc_task(kind, cfg, nsim, seed=987654321, override=None):
    """Worker entry point: rebuilds M from the picklable cfg exactly once (never crosses a
    process boundary — build_plateau()/build_no_gps_cure() return closures that can't be
    pickled), then decides internally whether mc() is worth running, so no separate "check
    the build first" pass is needed in the caller — build_no_gps_cure() alone (~785-point fit
    grid) can cost ~2-3s, so doing it twice (once to inspect, once to actually simulate) or
    doing it serially before a batch is submitted would erase the whole parallel win.

    override, if given, is {key: value} applied to M before mc() — used by the panel-h power
    sweep to vary presp/mG without mutating a shared M; in that case also returns the
    milestone-misfit 'E' that power_sweep needs, computed here where M is live.

    Returns {"build": <picklable summary of M, closures stripped>, "mc": <mc() result dict,
    or None if mc() wasn't run>}. mc() always runs for "plateau" or when override is given;
    for a bare "nogpscure" build it only runs when state == "C" (a boundary or residual-misfit
    alternative has no reported scenario rejection rate)."""
    M = build_plateau(cfg) if kind == "plateau" else build_no_gps_cure(cfg)
    extra = {}
    if override:
        for k, v in override.items():
            M[k] = v
        args = (M["presp"],) if kind == "plateau" else (M["mG"], M["sG"])
        extra["E"] = sum(M["WT"][j] * (M["ed_raw"](M["MT"][j], *args) - M["MOBS"][j]) ** 2
                          for j in range(3))
    run_mc = override is not None or kind == "plateau" or M.get("state") == "C"
    r = {**mc(M, nsim, seed), **extra} if run_mc else None
    # every curve/closure in M is callable and nothing else is, so this strips exactly the
    # unpicklable entries (Sbat/Sgps/Spool/ed/ed_raw/...) without a hand-maintained key list
    # that would silently go stale if a future build_*() added a new closure field
    build_summary = {k: v for k, v in M.items() if not callable(v)}
    return {"build": build_summary, "mc": r}

def run_batch(executor, specs):
    """specs: list of (kind, cfg, nsim[, seed[, override]]) tuples for _mc_task.
    Runs them across the pool (or serially if executor is None) and returns results in order.
    Uses submit() rather than map() so each spec's args are pickled individually — map() would
    require pickling a wrapper closure, which fails under Windows' spawn start method."""
    if executor is None:
        return [_mc_task(*s) for s in specs]
    futures = [executor.submit(_mc_task, *s) for s in specs]
    return [f.result() for f in futures]

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

def figure(path, nsim=1500, executor=None, base=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140})
    fig, ax = plt.subplots(3, 3, figsize=(16.5, 15.4)); tg = np.linspace(0, 48, 300)

    # base preset is reused by (a),(d),(e),(f); fit both panels once. Callers that already
    # built the base models (e.g. the CLI header) should pass base=(Mc,Ml,rc,rl) — rebuilding
    # build_no_gps_cure() here is not "cheap" (a ~785-point fit grid, ~2-3s), so redoing it is
    # worth avoiding when the caller already paid that cost. rc/rl inside base were computed at
    # the caller's own nsim, not this function's nsim argument — passing a base whose rc/rl used
    # a different nsim than the one given here would make panels a/e report displayed numbers at
    # a different MC budget than panels b/c/h; the CLI's sole caller keeps both at NSIM=800.
    cfg = apply_preset(default_cfg(), "base")
    if base is not None:
        Mc, Ml, rc, rl = base
    else:
        Mc = build_plateau(cfg); Ml = build_no_gps_cure(cfg)
        rc, rl = mc(Mc, nsim), mc(Ml, nsim)

    # (a) survival curves for the base preset, plateau (GPS-cure) panel
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

    # (b) scenario rejection rates across the non-responder sweep (base preset)
    fr = [0, 10, 20, 30, 40]
    fr_cfgs = [apply_preset(default_cfg(fnr=f / 100.0), "base") for f in fr]
    specs = []
    for c in fr_cfgs:
        specs += [("plateau", c, nsim), ("nogpscure", c, nsim)]
    results = run_batch(executor, specs)   # one flat parallel batch: build+mc, no serial pre-pass
    pc = [100 * results[2 * i]["mc"]["ps"] for i in range(len(fr_cfgs))]
    pll = [100 * results[2 * i + 1]["mc"]["ps"] if results[2 * i + 1]["mc"] else np.nan
           for i in range(len(fr_cfgs))]
    b = ax[0, 1]
    b.plot(fr, pc, color=NAVY, lw=2.4, marker="o", label="Plateau (GPS cure)")
    b.plot(fr, pll, color=ORANGE, lw=2.2, ls="-.", marker="s", label="No-GPS-cure (State C only)")
    b.axhline(50, color=GREY, ls=":", lw=1)
    b.set_ylim(0, 103); b.set_xlabel("% GPS non-responders"); b.set_ylabel("scenario rejection %")
    b.set_title("(b) Non-responders barely move the plateau scenario rate",
                fontweight="bold", fontsize=9); b.legend(fontsize=7.6)

    # (c) scenario rejection rates across the five BAT-composition presets
    names = ["base", "low", "dom", "bear", "bull"]; labels = ["Base", "Low-ven", "Ven-dom", "Bear", "Bull"]
    name_cfgs = [apply_preset(default_cfg(), nm) for nm in names]
    specs = []
    for c in name_cfgs:
        specs += [("plateau", c, nsim), ("nogpscure", c, nsim)]
    results = run_batch(executor, specs)   # one flat parallel batch: build+mc, no serial pre-pass
    gc = [100 * results[2 * i]["mc"]["ps"] for i in range(len(name_cfgs))]
    gl = [100 * results[2 * i + 1]["mc"]["ps"] if results[2 * i + 1]["mc"] else 0.0
          for i in range(len(name_cfgs))]
    c = ax[0, 2]; x = np.arange(len(names))
    c.bar(x - 0.19, gc, 0.36, color=NAVY, label="Plateau (GPS cure)")
    c.bar(x + 0.19, gl, 0.36, color=ORANGE, label="No-GPS-cure (State C; 0 = withheld)")
    c.set_xticks(x); c.set_xticklabels(labels); c.set_ylim(0, 103)
    c.set_ylabel("scenario rejection %")
    c.set_title("(c) Fixed-scenario rates and bounded-fit status",
                fontweight="bold", fontsize=9); c.legend(fontsize=7.0)

    # (d) event-accrual timeline — modeled cumulative deaths vs calendar, milestone anchors, 80-event trigger
    d = ax[1, 0]; N, FINAL = cfg["N"], cfg["FINAL"]
    t0 = float(Mc["coh"][0, 0]); t1 = 90.0
    ts = np.linspace(t0, t1, 220); dts = [_to_date(t) for t in ts]
    edc = np.array([Mc["ed"](t) for t in ts]); edl = np.array([Ml["ed"](t) for t in ts])
    d.plot(dts, edc, color=NAVY, lw=2.2, label="Plateau (GPS-cure) accrual")
    d.plot(dts, edl, color=ORANGE, lw=2.0, ls="-.", label=f"No-GPS-cure accrual (State {Ml['state']})")
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

    # (e) distribution of simulated final HRs — scenario rejection is the mass left of the threshold
    e = ax[1, 1]; HRC = cfg["HRC"]
    bins = np.linspace(0.0, 1.6, 41)
    hc = rc["hrsAll"]; hl = rl["hrsAll"]
    hc = np.clip(hc[np.isfinite(hc)], 0, 1.59); hl = np.clip(hl[np.isfinite(hl)], 0, 1.59)
    e.hist(hc, bins=bins, density=True, color=NAVY, alpha=.55, label=f"Plateau (GPS cure)  (rate={100*rc['ps']:.0f}%)")
    _npl = f"rate={100*rl['ps']:.0f}%" if Ml["state"] == "C" else f"State {Ml['state']}"
    e.hist(hl, bins=bins, density=True, color=ORANGE, alpha=.45, label=f"No-GPS-cure  ({_npl})")
    e.axvspan(0, HRC, color=TEAL, alpha=.07)
    e.axvline(HRC, color=RED, lw=1.4, ls="--")
    e.axvline(1.0, color=GREY, lw=1, ls=":")
    ytop = e.get_ylim()[1]
    e.text(HRC - 0.02, ytop * 0.92, f"significant\nHR ≤ {HRC:.3f}", color=RED, fontsize=7.2, ha="right", va="top")
    e.set_xlim(0, 1.6); e.set_xlabel("simulated final hazard ratio (GPS / BAT)"); e.set_ylabel("density")
    e.set_title("(e) Each trial's HR is a draw; rejection = mass below threshold",
                fontweight="bold", fontsize=9); e.legend(fontsize=7.2, loc="upper right")

    # (f) GPS-cure vs no-GPS-cure pooled divergence — both pinned at the milestones, fan apart in the tail
    f = ax[1, 2]
    sc = 100 * Mc["Spool"](tg); sl = 100 * Ml["Spool"](tg)
    f.fill_between(tg, sc, sl, color=GREY, alpha=.18, label="pooled disagreement")
    f.plot(tg, sc, color=NAVY, lw=2.2, label="Plateau (GPS-cure) pooled")
    f.plot(tg, sl, color=ORANGE, lw=2.0, ls="-.", label=f"No-GPS-cure pooled (State {Ml['state']})")
    for ev in cfg["ev"]:                                       # event-fraction levels both are pinned to
        f.axhline(100 * (1 - ev["n"] / N), color=RED, ls=":", lw=.9, alpha=.5)
    f.set_xlim(0, 48); f.set_ylim(0, 101)
    f.set_xlabel("months from randomization"); f.set_ylabel("% alive (pooled)")
    f.set_title("(f) Same milestones, different tail: is the plateau GPS-specific?",
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
    gx.set_title("(g) Legacy enrollment reconstruction vs sourced anchors",
                 fontweight="bold", fontsize=9); gx.legend(fontsize=7.2, loc="lower right")
    for lab in gx.get_xticklabels(): lab.set_rotation(25); lab.set_ha("right"); lab.set_fontsize(7.5)

    # (h) scenario rejection rate vs the implied treatment effect
    hx = ax[2, 1]; nsim_h = max(250, nsim // 3)

    def power_sweep(M, key, vals):
        """Sweep one effect knob (presp for plateau, GPS median mG for the bounded alternative); return implied HR,
        scenario rejection rate, and milestone misfit at each point.
        Each swept point rebuilds M inside its own worker (via M['cfg']/M['kind']) rather than mutating
        this M in place, so the sweep runs as one parallel batch instead of a serial loop."""
        specs = [(M["kind"], M["cfg"], nsim_h, 987654321, {key: v}) for v in vals]
        results = run_batch(executor, specs)   # override is set, so mc always runs (never None)
        hr = np.array([r["mc"]["medHR"] for r in results])
        ps = np.array([100 * r["mc"]["ps"] for r in results])
        E = np.array([r["mc"]["E"] for r in results])
        m = np.isfinite(hr) & np.isfinite(ps); hr, ps, E = hr[m], ps[m], E[m]
        o = np.argsort(hr); return hr[o], ps[o], E[o]

    p_hr, p_ps, p_E = power_sweep(Mc, "presp", np.linspace(0.0, 0.97, 13))
    mg_lo = Ml["batMed"] if np.isfinite(Ml["batMed"]) else 12.0
    l_hr, l_ps, l_E = power_sweep(Ml, "mG", np.linspace(mg_lo, 120.0, 13))

    def band(hr, E):                                            # HR span of the data-consistent (low-misfit) points
        if not len(E): return None
        ok = hr[E <= E.min() + 0.08 * (E.max() - E.min())]
        return (float(ok.min()), float(ok.max())) if len(ok) else None
    for hr, E, col in [(p_hr, p_E, NAVY), (l_hr, l_E, ORANGE)]:
        bd = band(hr, E)
        if bd: hx.axvspan(bd[0], bd[1], color=col, alpha=.10)
    hx.plot(p_hr, p_ps, color=NAVY, lw=2.4, marker="o", ms=3, label="Plateau (GPS cure)")
    hx.plot(l_hr, l_ps, color=ORANGE, lw=2.2, ls="-.", marker="s", ms=3, label="No-GPS-cure (vary mG)")
    hx.scatter([rc["medHR"]], [100 * rc["ps"]], color=NAVY, s=130, marker="*", zorder=6, edgecolor="#fff", linewidth=.8)
    hx.scatter([rl["medHR"]], [100 * rl["ps"]], color=ORANGE, s=130, marker="*", zorder=6, edgecolor="#fff", linewidth=.8)
    hx.axvline(HRC, color=RED, ls="--", lw=1.2)
    hx.text(HRC + 0.012, 6, f"test threshold HR={HRC:.3f}", color=RED, fontsize=7, rotation=90, va="bottom")
    hx.set_xlim(0, 1.15); hx.set_ylim(0, 103)
    hx.set_xlabel("implied trial hazard ratio (GPS / BAT)"); hx.set_ylabel("scenario rejection %")
    hx.set_title("(h) Scenario rejection rate vs effect size; ★ = current fit",
                 fontweight="bold", fontsize=9); hx.legend(fontsize=7.2, loc="lower left")

    # (i) how enrollment selection q lifts the BAT arm — median OS and cure fraction, the two
    #     quantities the comparator assumption turns on, shown explicitly.
    ix = ax[2, 2]; w0, cm0, h0 = Mc["w"], Mc["cm"], Mc["h"]
    qs = np.linspace(0.0, 0.5, 26)
    bat_med, bat_cure = [], []
    TH0 = max(Mc["cfg"].get("fvar", 0.0) or 0.0, 0.0)
    for q in qs:
        pib = sum(w0[i] * cm0[i]["cure"] for i in range(len(cm0)))   # selection does not move the cure fraction
        Sbq = lambda t, q=q: (sum(w0[i] * Scf(t, cm0[i]["med"], cm0[i]["cure"], cm0[i]["k"], TH0, q)
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

    fig.suptitle("REGAL legacy scenario explorer — fixed-scenario rejection rates and bounded-fit diagnostics; "
                 "not a posterior forecast for the ongoing trial.",
                 fontweight="bold", fontsize=10.5, y=1.01)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    return path

# ---------------------------------------------------------------- CLI
def fmt_med(m): return "NR" if not np.isfinite(m) else f"{m:.0f}mo"

if __name__ == "__main__":
    NSIM = 800   # matches the html's interactive budget (~600); raise for tighter MC error
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        base = apply_preset(default_cfg(), "base")
        Mc, Ml = build_plateau(base), build_no_gps_cure(base)
        rc, rl = mc(Mc, NSIM), mc(Ml, NSIM)   # Mc/Ml already built live; routing through the pool would rebuild for nothing
        wmode = "unweighted" if base["unweighted"] else "weighted 1/2/4"
        print(f"REGAL Scenario Explorer (base preset, f_nr=20%, natural death {100*base['ndr']:.1f}%/yr, "
              f"loss-to-FU {100*base['drop']:.0f}%/yr, eligibility screen-out {100*base['esel']:.0f}% "
              f"(frailty var {base['fvar']:.2f}), fit {wmode})")
        print(f"  BAT  : cure {100*Mc['pibat']:.0f}%  median {fmt_med(Mc['batMed'])}  @36mo {100*Mc['Sbat'](36):.0f}%")
        ci_more, ci_few = fit_ci(base, build_plateau)
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
        print(f"\n  PLATEAU SCENARIO : rejection rate {100*rc['ps']:.0f}%   medHR {rc['medHR']:.2f}   reached {100*rc['reach']:.0f}%")
        print(f"         interim: implied HR@{base['IA']} {ia}   "
              f"@80th: {rc['aliveG']:.0f} GPS alive / {rc['aliveB']:.0f} BAT alive")
        sh_tag = "fitted" if Ml['fitShape'] else "override"
        if Ml['state'] == "C":
            print(f"  BOUNDED NO-GPS-CURE : State C — adequate interior fit. A no-cure GPS responder "
                  f"(median {Ml['mG']:.0f}mo, tail sG={Ml['sG']:.2f} {sh_tag}) also fits.")
            print(f"         scenario rejection rate {100*rl['ps']:.0f}%   medHR {rl['medHR']:.2f}   ratio {Ml['ratio']:.1f}x   "
                  f"resid RMS {Ml['rmsResid']:.1f}  (conditional on this BAT assumption)")
        elif Ml['state'] == "A" and not Ml['cureReq']:
            print(f"  BOUNDED NO-GPS-CURE : State A — boundary / non-identified. {Ml['reason']}.")
            print("         scenario rejection rate withheld; boundary status is not a biological conclusion.")
        else:
            fit_status = "A (boundary / non-identified)" if Ml['state'] == "A" else "B (residual misfit)"
            print(f"  BOUNDED NO-GPS-CURE : State {fit_status}. {Ml['reason']}.")
            print(f"         scenario rejection rate withheld; this is a fit diagnostic "
                  f"(modeled {'/'.join(f'{x:.0f}' for x in Ml['edv'])} vs {'/'.join(f'{x:.0f}' for x in Ml['MOBS'])}).")
        print()

        print(f"{'preset':>8} | {'f_nr':>5} | {'plateau rate':>12} | {'alternative fit':>26} | {'BATmed':>7} {'GPSmed':>7}")
        preset_names = ["base", "low", "dom", "bear", "bull"]
        preset_cfgs = [apply_preset(default_cfg(), nm) for nm in preset_names]
        specs = []
        for c in preset_cfgs:
            specs += [("plateau", c, NSIM), ("nogpscure", c, NSIM)]
        results = run_batch(ex, specs)   # one flat parallel batch: build+mc, no serial pre-pass
        for i, (nm, c) in enumerate(zip(preset_names, preset_cfgs)):
            p_res, n_res = results[2 * i], results[2 * i + 1]
            mcc, mll, rcc = p_res["build"], n_res["build"], p_res["mc"]
            if mll['state'] == "C":
                rll = n_res["mc"]; nv = f"C · adequate (rate={100*rll['ps']:.0f}%)"
            else:
                nv = f"{mll['state']} · " + ("boundary" if mll['state'] == "A" else "residual misfit")
            print(f"{nm:>8} | {100*c['fnr']:4.0f}% | {100*rcc['ps']:9.0f}% | {nv:>26} | "
                  f"{fmt_med(mcc['batMed']):>7} {fmt_med(mll['mG']):>7}")

        out = figure("regal_explorer_panel.png", NSIM, executor=ex, base=(Mc, Ml, rc, rl))
        print(f"\nsaved {out}")
