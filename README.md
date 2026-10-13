# Supplementary Material

## When Optimizing a Zero-Cost Proxy Is Worse Than Not Using One

This repository contains the code, processed benchmark tables, and selected numerical outputs accompanying the paper. The manuscript and publication figures are not included here. All experiments use precomputed benchmark records; a GPU is not required.

## Study design

The principal search space is NAS-Bench-201, with 15,625 architectures evaluated on CIFAR-10, CIFAR-100, and ImageNet16-120. The study uses 13 zero-cost proxy scores in their published directions, without correction based on validation labels. For a training budget `B`, an endpoint selects the `B` highest-scoring architectures and reports the best validation accuracy among them. Its reference value is the exact expected maximum validation accuracy of `B` architectures sampled uniformly without replacement.

Endpoint curves cover budgets 1–1,000. Their aggregation calibration uses 2,000 distinct architectures drawn with seed 0. The search experiment draws a separate 2,000-architecture calibration sample with replacement for each seed. Standard scores use the population standard deviation. Search fixes `B = 20` and varies the number of candidate evaluations `N` over 200, 500, 1,000, 2,000, and 5,000. Aging evolution uses an initial population of 50, tournament sizes of 2, 5, 10, 25, and 50, and 20 random seeds. The 50 initial evaluations count toward `N`. A uniform random search supplies the sixth setting at each `N`. Repeated architectures are removed before the `B` architectures are selected for training.

The test-accuracy analysis selects an architecture by validation accuracy within the proxy-ranked top `B`, then reports its official NATS-Bench test accuracy. For CIFAR-10, the official validation and test values come from different training settings; that result is an architecture-level comparison. The additional TransNAS-Bench-101 analysis covers the micro and macro spaces on scene classification, object classification, and jigsaw tasks.

## Repository map

| Paper analysis | Code | Retained output |
|---|---|---|
| NAS-Bench-201 validation endpoints, Sections 5.1–5.5 | `experiments/exp05_endpoint_curve.py` | `results/exp05_174c3d34171f.{json,npz}` |
| Aggregation variants, Section 5.4 | `experiments/exp08_aggregation_variants.py` | `results/exp08_aggregation_variants.json` |
| Search intensity and paired analysis, Section 5.6 | `experiments/exp03_kl_sweep.py`, `experiments/analyze_c1_paired.py` | `results/exp03_*.jsonl`, `results/c1_paired_randomfix.json` |
| Validation-selected test endpoints, Section 5.7 | `experiments/exp06_test_endpoint.py` | `results/exp06_test_endpoint_favbounds.{json,npz}` |
| TransNAS-Bench-101 endpoints, Section 5.8 | `experiments/exp07_transnas_endpoint.py` | `results/exp07_transnas.{json,npz}` |

The reusable benchmark, search, aggregation, and metric implementations are in `src/exnas/`. The `tests/` directory contains component checks.

## Data and provenance

| Included file | Content and source |
|---|---|
| `data/zc_nasbench201.json` | NAS-Bench-201 proxy scores and validation accuracies from [NAS-Bench-Suite-Zero / NASLib](https://github.com/automl/NASLib). |
| `data/zc_transbench101_micro.json`, `data/zc_transbench101_macro.json` | [NAS-Bench-Suite-Zero](https://github.com/automl/NASLib) proxy tables for the [TransNAS-Bench-101](https://github.com/yawen-d/TransNASBench) search spaces. |
| `data/nb201_acc.npz` | Compact 200-epoch, seed-averaged validation and test accuracies queried from the [NATS-Bench TSS API](https://github.com/D-X-Y/NATS-Bench), stored in this repository's architecture order. |

The original multi-gigabyte NATS-Bench archive is not included. `experiments/fetch_nb201_test.py` rebuilds the compact accuracy table from a locally installed TSS archive. Consult the upstream projects for their data distributions, citations, and applicable terms.

## Environment

Python 3.10 or newer is recommended. Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The optional accuracy-table extraction script also requires `nats-bench`. Component checks can be run with `python -m pytest -q`.

## Reproduction commands

Run these commands from the repository root. Generated files may be directed to a separate output directory; the retained outputs above are provided for comparison.

### Validation endpoints and aggregation

```bash
python experiments/exp05_endpoint_curve.py --n-calib 2000 --b-max 1000 --out results/reproduced
python experiments/exp08_aggregation_variants.py --out results/reproduced/exp08_aggregation_variants.json
```

The endpoint archive contains the original curves and separate lower and upper curves for proxy-score ties. The aggregation script records unrounded minima and their budgets for the eight representation and combination variants.

### Search and paired comparisons

```bash
python experiments/exp03_kl_sweep.py --datasets cifar10 cifar100 ImageNet16-120 --seeds 20 --budget 20 --n-calib 2000 --out results/reproduced
python experiments/analyze_c1_paired.py results/exp03_e1c77996195e.jsonl results/exp03_f1112324f45d.jsonl --out results/reproduced/c1_paired.json
```

The full search command performs 36,000 runs. The first retained log contains the original full experiment; the second supplies corrected architecture-keyed random-score control runs. The paired-analysis command reads them in the displayed order so the corrected control rows replace the original rows. The 20 recorded scorers consist of 13 individual proxies, four rank aggregators, a standardized-score mean, and two controls. The paired analysis uses 20 seeds, two-fold cross-fitting, and Holm-adjusted p-values.

### Test accuracy and additional search spaces

```bash
python experiments/exp06_test_endpoint.py --data data/zc_nasbench201.json --acc data/nb201_acc.npz --n-calib 2000 --out results/reproduced/exp06_test_endpoint
python experiments/exp07_transnas_endpoint.py --out results/reproduced/exp07_transnas
```

For the official accuracy table, obtain the NATS-Bench TSS archive from its project and run `python experiments/fetch_nb201_test.py --root <path-to-unpacked-tss-data>`. The retained TransNAS output uses the script's default ordinal ranks over each full search space.
