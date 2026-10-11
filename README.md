# local-laplacian-energy


Reproducibility package for

> C. Dede, **Local Laplacian Energy and Vertex-Deletion Profiles of Graphs**.

**Paper (preprint):** [paper/local_laplacian_energy_preprint.pdf](paper/local_laplacian_energy_preprint.pdf)

Every computational statement of the paper is checked by one function in `lle/checks/`, and
`run_all` prints a PASS/FAIL line for each. With `MAX_ORDER=10` all **58 checks** pass
(the logs in `results/logs/` were produced that way).

## Requirements

* Python 3.9 or later with `numpy`, `sympy`, `mpmath`, `networkx`. A virtual environment is
  recommended (on macOS with Homebrew Python it is required):
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
  ```
* nauty's `geng` for the exhaustive scans:
  `apt install nauty` (Debian/Ubuntu, installs `nauty-geng`), `brew install nauty` (macOS),
  `conda install -c conda-forge nauty`, or simply run inside SageMath, which ships `geng`.

SageMath is **not** required.

## Running

```bash
python3 run_all.py                    # orders up to 9, about 4 minutes
python3 run_all.py MAX_ORDER=10       # adds the order-ten row, about 10-15 more minutes the first time
python3 run_all.py MAX_ORDER=10 PARTS=4          # same, order ten split into four cached pieces
python3 run_all.py SECTIONS='["sec4"]' KRR_R_MAX=1000
```

In Jupyter, open `reproduce.ipynb` from the repository root, edit `PARAMS`, and run the single cell.
All parameters and their defaults are listed in `lle/run.py` (`DEFAULTS`); an unknown parameter
name raises an error listing the valid ones.

The exhaustive scans are cached in `results/cache/` (order ten is about 300 MB, hence excluded
from version control), so only the first run at a given order is slow.

## What is checked

| Paper | Check (as printed by `run_all`) | Module |
|---|---|---|
| Remark, logical independence of LE and e_L | values for P3, K3, paw, diamond; pair counts at orders 3-6 | `sec2` |
| Example, total local energy / profile / spectrum | (a) `EEvw`, `EE~w`; (b) `FCQVw`, `FCQvw`, exact | `sec2` |
| Theorem, central moments of the profile | duality for k = 1..4, all graphs of order <= 7 | `sec2` |
| Lemma, variational form; Lemma, deletion majorisation | all graphs of order <= 8 | `sec3` |
| Theorem, necessary condition for negative local energy | all (G, v) of order <= MAX_ORDER (119 611 146 pairs at 10) | `sec3` |
| Proposition / Corollary, complete multipartite graphs | Phi, stars, K_{p,q}, balanced multipartite, exact | `sec4` |
| Corollary, range of the local energy; Problem on the minimum | star centres; K_{r,r} formula; minimisers for orders 4..MAX_ORDER | `sec4` |
| Theorem, deletion spectrum of a strongly regular graph | 7 strongly regular graphs; the two identities used in the proof | `sec4` |
| Example, Petersen and Clebsch; Corollary, conference graphs | 40-digit values; Paley 5, 13, 17, 29 | `sec4` |
| Proposition, cycles and paths; Corollary, cycles | closed forms, positivity, 4/pi limit, proof chain | `sec4` |
| Proposition / Theorem / Example / Corollary, corona | spectrum of G o K_1, K_n o K_1 values, spread asymptotics | `sec4` |
| Proposition, clique joins; Corollaries, windmills | Omega, LE, e_L, negativity condition, hub and blade values | `sec4` |
| Proposition, centering shift; Theorem, variance bound; Remark, sharpness | all graphs of order <= 7; star ratios | `sec5` |
| Corollary, regular graphs; Example `FFzn_`; Corollary, spread | exact profiles and both bounds | `sec5` |
| Examples, cospectral graphs | smallest pair; rook vs Shrikhande; `HC~vfr~` vs `H?~vvj~` | `sec5` |
| Table, census of ~_L classes; Remark, exactness | orders 6..MAX_ORDER, largest class, exact certification, stability window | `sec5` |

## Exactness

Energies are first computed in double precision (batched eigensolves), and every equality the
paper relies on is then settled exactly:

* **Laplacian cospectrality** through the integer power sums tr L^k.
* **Equality of energies and profiles** through the *symbol* of a Laplacian energy
  (`lle/exact.py`): factor the characteristic polynomial over Q, count roots above the baseline
  with Sturm sequences, and normalise each straddling factor modulo rational translation and the
  reflection x -> -x. Equal symbols imply equal energies; only integer and rational arithmetic is
  involved. (Interval arithmetic cannot prove that two algebraic numbers are equal.)
* Equalities the symbols do not settle, and all near misses between distinct census classes,
  are confirmed in 60-digit arithmetic and reported as such.

## Results

| File | Content |
|---|---|
| `results/classes_order_<n>.txt` | every ~_L class of connected graphs of order n, as `class_id graph6` |
| `results/profile_pairs_order_<n>.csv` | all profile-preserving pairs, with cospectrality, exact certification, rationality, whether the reflection is needed, and the exact profile when rational |
| `results/census_summary.json` | the census table and the statistics of the largest class |
| `results/logs/sec*.txt` | full output of `run_all` with `MAX_ORDER=10` |

## Layout

```
lle/scan.py        streaming decoder for geng output, batched energies
lle/exhaustive.py  one cached pass per order (sign theorem, range, minimum)
lle/census.py      ~_L classes, invariants of pairs, exact certification
lle/exact.py       characteristic polynomials, symbols, certificates
lle/families.py    closed forms of Section 4
lle/checks/        sec2.py ... sec5.py, one check per statement
run_all.py         command line;  reproduce.ipynb  single-cell notebook
```

## Citation

See `CITATION.cff`. An archived release with a DOI is available on Zenodo: `10.5281/zenodo.23298644`.
