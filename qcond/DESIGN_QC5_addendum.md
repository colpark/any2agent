## Frozen addendum

### The projection
- **The formula.** Write the fitted formula as params(e) = M e + c, where e is a question embedding and params = (w, b).
- **The logit.** For a standardized Prithvi feature vector f, the logit is

  eᵀ(M_wᵀ f + m_b) + (c_wᵀ f + c_b).

- **The projection.** Define g(f) = M_wᵀ f + m_b, a vector in the 2,560-d embedding space. Here m_b is the row of M for the bias and c_w is the weight part of c.
- **Text scores.** A text t scores eᵀ(t) g(f). The temperature does not change rankings, so leave it out.

### A. Caption retrieval (Test, 997 patches)
- **Candidates.** The 997 Test captions, each represented by its VS summary field embedding (field 6), L2-normalized.
- **QC projection.** Rank all 997 candidates for each Test patch by e(summary) · g(f).
- **Caption ridge.** The conventional alignment route, given the same 300 patches but captions instead of questions.
  - Fit ridge from F to the summary embedding on the 300 Practice patches, using the AH bank captions.
  - Choose alpha from `np.logspace(-3, 3, 13)` by 5-fold CV on Practice, scored by cosine.
  - Rank candidates by cosine to the predicted embedding.
- **Control.** The QC projection with Test rows of F shuffled (seed 0).
- **Measures.**
  - Recall@1, Recall@10 and the median rank of each patch's own caption. Chance is 0.001 and 0.010.
  - Paired bootstrap 95% CIs over Test patches, 1,000 resamples.

### B. Describing a patch by ranking questions (Test, 23 held-out semantic questions)
- **QC projection.** For each Test patch, rank the 23 questions by e(q) · g(f).
- **Language route.** The v1 language-route score: the mean cosine of the question to the patch's 7 VS field embeddings.
- **Measure.**
  - Per patch, the AUC of the ranking against that patch's true answers on the 23 questions, averaged over patches.
  - Skip patches whose true answers are all yes or all no, and log the count.
  - Paired bootstrap 95% CI over patches for QC minus the language route.

### Readouts (report only, no pass or fail)
- **A.** Recall@1, Recall@10 and the median rank for the QC projection, the caption ridge and the control.
- **B.** Mean per-patch AUC for the QC projection and the language route, and their difference with its CI.
- **Rank of M.** The effective rank of M_w: the number of singular values above 1% of the largest. It bounds how many text directions the projection spans.
