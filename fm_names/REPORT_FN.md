# Naming Prithvi's directions (FN-0 to FN-3): report

**Verdict N1: fails.**
- **Named: 5 of 20** directions (D1 Wilson lower bound above 0.5): PC1, 2, 4, 8 and 9. The pass mark was 10.
- **Variance:** those 5 carry 64% of the top-20 variance.
- **Control:** mean D1 is 0.615 against 0.598 for the deranged control D0s (Wilcoxon p = 0.69).
- **Where it works:** language reads Prithvi's few dominant directions well (PC1 0.93, PC2 0.88, PC4 0.97). Beyond the first handful, Astra's descriptions place new patches no better than another direction's description. D1 falls with PC rank (Spearman −0.52, p = 0.018).

## Deviations from DESIGN_FN_v1.md

| Id | Deviation | Status |
|---|---|---|
| — | none | |

There were no FN-E bug fixes. PC1's near-black A-end exemplars are real data: B04 and B08 near the reflectance floor (dark water or no-data). Their renders are correct.

## Setup
- **PCA:** 20 components on 1,000 discovery patches (the FB bank) explain 84.8% of variance; PC1 alone explains 32.8%.
- **Astra:** 20 naming calls, all valid on the first attempt with 0 tool use and no usage limit. Calls used: 20 of 24.
- **Reader:** Qwen3.5-9B (c2022362…) on A2 ran 1,600 prompts in 604 s, with 0 unparsed.
- **Reader bias:** it answers "1" 59.6% of the time under D1 and 59.9% under D0s (the A end is at position 1 half the time).
- **Control null:** D0s is scored at its better sign. Under pure guessing with 40 pairs that averages about 0.56, so D0s's 0.60 is only slightly above chance.

## Per direction

| PC | EVR | End A (Astra) | End B (Astra) | D1 [Wilson 95%] | D0s (better sign) | Visible | Visible R² | SWIR gain |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.328 | Near-black, low-detail patches | Bright, clearly textured land | **0.93** [0.80, 0.97] | 0.70 | high | 0.84 | 0.041 |
| 2 | 0.104 | Open fields and sparse vegetation | Dense woodland and tree cover | **0.88** [0.74, 0.95] | 0.57 | high | 0.39 | 0.055 |
| 3 | 0.078 | Snow-free terrain | Snow-covered terrain | 0.60 [0.45, 0.74] | 0.65 | high | 0.69 | 0.106 |
| 4 | 0.066 | Rugged, irregular landscapes | Green agricultural field mosaics | **0.97** [0.87, 1.00] | 0.82 | high | 0.42 | 0.110 |
| 5 | 0.050 | Pervasive surface texture | Smooth water with discrete land | 0.57 [0.42, 0.71] | 0.55 | high | 0.23 | 0.062 |
| 6 | 0.032 | Mostly snow-free terrain | Snow-covered terrain | 0.35 [0.22, 0.50] | 0.53 | high | 0.47 | 0.016 |
| 7 | 0.028 | Vegetation or open water | Exposed-soil cropland | 0.53 [0.37, 0.67] | 0.55 | high | 0.27 | 0.102 |
| 8 | 0.024 | Reflective water and open cropland | Dark water and dense woodland | **0.70** [0.55, 0.82] | 0.68 | medium | 0.18 | 0.071 |
| 9 | 0.021 | Green, densely vegetated farmland | Dry, sparsely vegetated farmland | **0.68** [0.52, 0.80] | 0.75 | high | 0.22 | 0.175 |
| 10 | 0.018 | Broad agricultural fields | Narrow agricultural strips | 0.62 [0.47, 0.76] | 0.60 | high | 0.06 | 0.035 |
| 11 | 0.017 | Vegetated, gentle terrain | Rugged, sparsely vegetated mountains | 0.60 [0.45, 0.74] | 0.57 | high | 0.15 | 0.040 |
| 12 | 0.014 | Smooth, continuous cover | Contrasting, fragmented cover | 0.62 [0.47, 0.76] | 0.62 | high | 0.23 | 0.035 |
| 13 | 0.013 | Bright, open surfaces | Dark water and green cover | 0.42 [0.29, 0.58] | 0.60 | medium | 0.03 | 0.060 |
| 14 | 0.010 | Fine, fragmented texture | Broad, continuous cover | 0.60 [0.45, 0.74] | 0.50 | medium | 0.25 | 0.055 |
| 15 | 0.009 | Strong directional structure | Mixed or irregular structure | 0.62 [0.47, 0.76] | 0.55 | medium | 0.15 | 0.069 |
| 16 | 0.008 | Bright, open surfaces | Dark water and shaded terrain | 0.50 [0.35, 0.65] | 0.55 | high | 0.16 | 0.119 |
| 17 | 0.007 | Snow-free surfaces | Snow-covered terrain | 0.47 [0.33, 0.63] | 0.57 | high | 0.14 | 0.013 |
| 18 | 0.007 | Snow-free, greener appearance | Wintry, snow-covered appearance | 0.57 [0.42, 0.71] | 0.55 | medium | 0.12 | 0.130 |
| 19 | 0.007 | Greener, summer-like surfaces | Dormant, winter-like surfaces | 0.53 [0.37, 0.67] | 0.53 | medium | 0.10 | 0.038 |
| 20 | 0.006 | Open-water dominated | Land and vegetation dominated | 0.53 [0.37, 0.67] | 0.50 | high | 0.05 | 0.048 |

Bold = named. Figure: `figs/d1_vs_rank.png` (D1 against PC rank, colored by SWIR gain).

## Pooled and structure
- **Named directions' share of top-20 variance:** 0.64.
- **Spearman, D1 against:**
  - PC rank: -0.52 (p = 0.018)
  - SWIR gain: +0.24 (p = 0.31)
  - Astra's visible rating: +0.09 (p = 0.72)

## Findings
1. **The dominant structure is nameable.**
   - PC1 is dark, low-detail patches against bright textured land. PC2 is open fields against woodland. PC4 is rugged terrain against field mosaics.
   - A different, free model uses these names to place held-out patches at 88–97% accuracy.
   - These directions carry most of the variance, so in variance terms language reads most of Prithvi's top-20 structure: 64%.
2. **The tail is not nameable this way.** From PC5 on, only PC8 and PC9 clear the bar. D1 and D0s are indistinguishable (0.615 against 0.598).
3. **Snow names fail.** Astra described five directions as snow against no snow (PC3, 6, 17, 18 and 19), and none is named.
   - Prithvi spreads season and snow over several directions. The same visible contrast cannot tell them apart, and the held-out extremes of a minor direction need not show snow.
   - PC6 scores 0.35, below chance: its description points the wrong way on held-out pairs.
4. **SWIR content does not predict failure.** The correlation of D1 with SWIR gain is positive and not significant. The bottleneck is not shortwave-only information; it is that low-variance directions mix several visible factors that six extremes do not pin down.
5. **Astra's own visibility rating does not predict success.** It rated 14 of 20 directions as high visibility.

## Limits
- There are 40 pairs per direction, so a direction near 0.6 cannot be told from chance (CI ±0.15).
- One reader model; a stronger reader may name more directions.
- The exemplars are only the 6 most extreme patches per end, and the extremes of minor directions may be outliers.
- PCA directions need not align with nameable factors. A rotation, such as ICA or varimax, may separate the mixed tail. That would need a CHANGE_REQUEST.
