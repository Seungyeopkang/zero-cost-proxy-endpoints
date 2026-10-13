# Zero-Cost Proxy Endpoint Evaluation

Supplementary code and benchmark inputs for the study of zero-cost proxy selection under a fixed training budget. The repository contains experiment scripts and data only; manuscript files and publication figures are kept out of Git.

## Repository contents

- `src/exnas/`: NAS-Bench-201 encoding, proxy aggregation, search, benchmark access, and evaluation metrics.
- `experiments/exp03_kl_sweep.py`: 20-seed search experiment over candidate-pool sizes and evolutionary tournament sizes.
- `experiments/exp05_endpoint_curve.py`: exact endpoint and without-replacement random-baseline curves for budgets 1–1000.
- `experiments/exp06_test_endpoint.py`: validation-selected test-accuracy endpoint analysis using the official NATS-Bench results.
- `experiments/exp07_transnas_endpoint.py`: endpoint analysis on TransNAS-Bench-101 micro and macro spaces.
- `experiments/analyze_c1_paired.py`: cross-fitted paired analysis of the search experiment.
- `experiments/fetch_nb201_test.py`: optional script to rebuild the compact official accuracy table from a local NATS-Bench TSS installation.
- `tests/`: unit tests for the search-space and evaluation components.
- `data/`: proxy tables and the compact validation/test accuracy table needed by the experiments.
- `results/`: selected experiment logs and summaries used to reproduce the reported paired analysis.

## Environment

Python 3.10 or newer is recommended. Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

Run the unit tests with:

```bash
python -m pytest -q
```

All experiments use table lookups and do not require a GPU.

## Reproducing the experiments

Run the full NAS-Bench-201 search experiment (3 datasets, 20 seeds, 30 search settings):

```bash
python experiments/exp03_kl_sweep.py \
  --datasets cifar10 cifar100 ImageNet16-120 \
  --seeds 20 --budget 20 --n-calib 2000 --out results/reproduced
```

Compute deterministic validation-accuracy endpoint curves:

```bash
python experiments/exp05_endpoint_curve.py \
  --n-calib 2000 --b-max 1000 --out results/reproduced
```

Repeat the test-accuracy endpoint analysis:

```bash
python experiments/exp06_test_endpoint.py \
  --data data/zc_nasbench201.json --acc data/nb201_acc.npz \
  --n-calib 2000 --out results/reproduced/exp06_test_endpoint
```

Run the TransNAS-Bench-101 analysis:

```bash
python experiments/exp07_transnas_endpoint.py \
  --out results/reproduced/exp07_transnas
```

Recreate the cross-fitted paired analysis from the retained search logs:

```bash
python experiments/analyze_c1_paired.py \
  results/exp03_e1c77996195e.jsonl \
  results/exp03_f1112324f45d.jsonl \
  --out results/reproduced/c1_paired.json
```

The first search log contains the original full run. The second contains the corrected architecture-keyed random-score control; when both files are read in this order, the corrected rows replace the corresponding original control rows.

In aging evolution, the initial population of 50 is included in the `N` candidate evaluations. Repeated architectures are removed before selecting the top `B`. Across the 37,800 stored search records, the smallest deduplicated pool contained 75 architectures, so every recorded `B=20` run had enough distinct candidates.

## Data

- `zc_nasbench201.json` contains NAS-Bench-Suite-Zero proxy scores and validation accuracy for the 15,625 NAS-Bench-201 architectures.
- `zc_transbench101_micro.json` and `zc_transbench101_macro.json` contain the TransNAS-Bench-101 proxy tables used by the transfer experiment.
- `nb201_acc.npz` contains the official NATS-Bench 200-epoch, seed-averaged validation and test accuracies in the repository's architecture order.

The original multi-gigabyte NATS-Bench archive is not included. To regenerate `nb201_acc.npz`, obtain and unpack the official NATS-Bench TSS data locally, then run `experiments/fetch_nb201_test.py --root <path-to-unpacked-tss-data>`. The proxy tables originate from NAS-Bench-Suite-Zero; consult the upstream dataset distribution for its attribution and terms.

## Output files

Experiment scripts write generated files under `results/`. Generated results are ignored by default; only the selected logs and summaries included with this supplementary release are tracked.
