# Reproducibility audit of the final PDF

The final PDF was checked against the repository's input tables and retained results. These notes record the two discrepancies that remain in the manuscript. The PDF and HWP are intentionally excluded from this supplementary repository, so this audit does not change them.

## Table 5(b), standardized scores clipped at ±3

Using the manuscript's settings (13 original-sign proxies, seed-zero sample of 2,000 distinct architectures, population standard deviation with zero replaced by one, NAS-Bench-Suite-Zero validation accuracy, budgets 1–1,000), clipping each standardized proxy score with `np.clip(z, -3, 3)` before averaging gives:

| Dataset | Recalculated minimum Δ(B) | PDF Table 5(b) |
|---|---:|---:|
| CIFAR-10 | −0.372858 pp at B=4 | −1.75 pp at B=2 |
| CIFAR-100 | −1.377066 pp at B=2 | −0.07 pp at B=303 |
| ImageNet16-120 | −13.460601 pp at B=2 | −8.54 pp at B=1 |

The clipped aggregate has 15,625 unique scores in each dataset, so proxy-score tie ordering cannot explain this difference. The seven other Table 5(b) rows reproduce their displayed values and budgets to two decimal places. In particular, raw-score mean and median/IQR-standardized mean agree with their PDF rows, as does `-logsumexp(-z)`.

Run `python experiments/exp08_aggregation_variants.py` to regenerate the unrounded values in `results/exp08_aggregation_variants.json`. The code or original log that produced the PDF's clipped row was not present in this repository, so its exact calculation remains unidentified. The PDF's statement that ±3 clipping removes the CIFAR-100 deficit is not supported by the stated calculation.

## Search experiment scorer count

The retained 36,000-run search log contains 20 scorers: 13 single proxies, four rank-based aggregators, one standardized-score mean, and two controls. The final PDF's Experimental Setup instead describes this as 13 single proxies, five rank-based aggregators, and two controls. There is no separate `mean_rank` search-run entry; `aznas_l` and `mean_rank` yield the same endpoint ranking, but the 60 statistical comparisons in the retained log include `linear` in place of a separate `mean_rank` entry. The paper's numerical 33/60 result reproduces from those retained log rows. The Experimental Setup should state the actual 13 + 4 + 1 + 2 composition.

## TransNAS-Bench-101 percentile ties

The code that produced the final PDF's Table 8 gave tied TransNAS proxy scores different ordinal ranks via `argsort(argsort(...))`. Section IV defines a percentile from the number of scores *strictly lower* than the candidate score, so tied scores should receive the same value under that definition. TransNAS `flops` and `params` have many ties. Applying the strict definition changes the worst of three rank-rule minima from −0.87 to −1.33 pp for micro/jigsaw and from −0.36 to −0.20 pp for macro/jigsaw; the other four Table 8 rank cells keep their displayed two-decimal values. The corresponding rank-versus-score gaps also change.

`exp07_transnas_endpoint.py` now makes this choice explicit. Its default `--rank-ties ordinal` reproduces the PDF and original retained results; `--rank-ties strict` follows the Section IV formula and produces the separate `results/exp07_transnas_strict.*` artifacts. The final PDF would need the TransNAS tie rule stated explicitly or the affected Table 8 values revised.

## Checks that matched

- Regenerated NAS-Bench-201 validation endpoint curves match the retained arrays to floating-point rounding. The result files now also contain both bounds for proxy-score ties; their `flops` and `params` ranges at B=20 match Table 3.
- Regenerated official-data test endpoint results match the retained JSON and arrays exactly, including the validation-selected random baseline and proxy-favorable validation-tie bounds.
- Regenerated TransNAS-Bench-101 results with the published ordinal tie policy match the retained JSON exactly and arrays to floating-point rounding.
- Reanalyzing the original search log with the corrected random-score overlay reproduces 33 significant comparisons out of 60 under Holm correction.
