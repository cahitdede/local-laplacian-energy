"""Run every check of the paper with one parameter dictionary."""
import time

from .checks import sec2, sec3, sec4, sec5

DEFAULTS = dict(
    SECTIONS=["sec2", "sec3", "sec4", "sec5"],
    # exhaustive data: orders 2..MAX_ORDER (9: ~1 min ; 10: ~10-15 min the first time, cached afterwards)
    MAX_ORDER=9,
    PARTS=1,                  # split order 10 into this many geng res/mod pieces (each cached)
    # Section 2
    IND_N_MAX=6, DUALITY_N_MAX=7,
    # Section 3
    LEMMA_N_MAX=8, KN_N_MAX=30,
    # Section 4
    MULTIPARTITE_N_MAX=12, STAR_N_MAX=30,
    KRR_EXACT_R_MAX=12, KRR_R_MAX=300, KRR_NUM_STEP=7,
    PALEY_QS=[5, 13, 17, 29],
    PATH_L_MAX=60, THRESHOLD_L_MAX=100000, CYCLE_GRAPH_N_MAX=60,
    CYCLE_CF_N_MAX=100000, PHI_P_MAX=50000, GRID_POINTS=10000,
    CORONA_RANDOM=20, CORONA_CHARPOLY_N_MAX=12, CORONA_GRAPH_N_MAX=30,
    CORONA_ASYMP_NS=[10, 100, 1000, 10000, 100000],
    WINDMILL_B_MAX=6, WINDMILL_R_MAX=8,
    # Section 5
    VARIANCE_N_MAX=7, REGULAR_N_MAX=8, NEAR_MISS=1e-6,
)


def run_all(**overrides):
    unknown = sorted(set(overrides) - set(DEFAULTS))
    if unknown:
        raise ValueError("unknown parameter(s): %s\nvalid: %s" % (", ".join(unknown), ", ".join(sorted(DEFAULTS))))
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    t0 = time.time()
    mods = {"sec2": sec2, "sec3": sec3, "sec4": sec4, "sec5": sec5}
    rows = []
    for s in cfg["SECTIONS"]:
        rows += mods[s].run(cfg)
    good = sum(ok for _, ok in rows)
    print("\n" + "=" * 78)
    print("TOTAL: %d / %d checks passed   (%.0f s)" % (good, len(rows), time.time() - t0))
    for name, ok in rows:
        if not ok:
            print("   FAILED:", name)
    print("=" * 78)
    return rows
