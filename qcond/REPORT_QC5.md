# QC-5: the by-product projection of the QC formula (no Astra calls)

## Deviations
| Id | What | Effect |
|---|---|---|
| None | No change requests. | – |
| Note | The caption ridge picked alpha = 1000, the top of the frozen grid (5-fold CV cosine rises monotonically). | A larger alpha might do slightly better; the grid is frozen. |
| Note | v1 stores the formula as params(e) = e M + c with M of shape 2,560 × 4,097, so the addendum's M_wᵀ f is computed as M[:, :4096] f. | Same quantity, checked below. |

## Results
**A, caption retrieval:** the QC projection finds a patch's own caption far above chance but not better than a ridge trained on captions. QC Recall@10 is 0.041 [0.029, 0.053], against 0.054 [0.041, 0.068] for the caption ridge, 0.010 for the shuffled control and 0.010 for chance. QC's median rank is 301 against 234 for the ridge (QC − ridge [+19, +112]). **B, ranking questions:** the projection ranks the 23 held-out semantic questions by what is true for a patch better than the language route. The mean per-patch AUC is 0.761 [0.751, 0.770] against 0.684 [0.673, 0.695], a difference of +0.077 [0.062, 0.091]. No patch was skipped: none had all 23 answers the same.

**Check.** On 5 Test patches and 3 held-out questions, eᵀ g(f) + c_wᵀ f + c_b reproduces the v1 logits to 4.9e-15.

## A. Caption retrieval (Test, 997 candidates)
| Route | Recall@1 | Recall@10 | Median rank |
|---|---|---|---|
| QC projection | 0.006 [0.002, 0.011] | 0.041 [0.029, 0.053] | 301 [267, 338] |
| Caption ridge (300 Practice captions) | 0.008 [0.003, 0.014] | 0.054 [0.041, 0.068] | 234 [208, 263] |
| Control: QC, F rows shuffled | 0.002 [0.000, 0.005] | 0.010 [0.004, 0.016] | 504 [476, 533] |
| Chance | 0.001 | 0.010 | 499 |
| QC − caption ridge | [−0.009, +0.005] | [−0.032, +0.007] | [+19, +112] |
| QC − control | [0.000, +0.009] | [+0.019, +0.043] | [−236, −171] |

## B. Ranking the 23 held-out semantic questions per patch (Test, 997 patches)
| Route | Mean per-patch AUC |
|---|---|
| QC projection | 0.761 [0.751, 0.770] |
| Language route (v1) | 0.684 [0.673, 0.695] |
| QC − language | **+0.077 [0.062, 0.091]** |

## Rank of M_w
- **Effective rank:** 138, counting singular values above 1% of the largest. That is almost the full rank of 139 practice answerers, so the projection spans about as many text directions as there were practice questions.
- **Concentration:** the top 10 singular values hold 44% of the squared spectrum.

## Reading
- The projection carries patch content into the text space. It beats the shuffled control by a wide margin and the language route on question ranking (B).
- The projection is still a weak caption retriever. With 139 question directions it does not beat a direct F-to-caption ridge, and it trails it on median rank.
- The polarity problem from v1 also applies here. B uses the same 23 questions, including those where QC is inverted (C19, C8).

Files: qc5.json, DESIGN_QC5_addendum.md (sha256 4b097850addb84a2c765dbc472fca3e66c78809037a5fb5e89430ae3b18fb26c) and scripts/qc5.py.
