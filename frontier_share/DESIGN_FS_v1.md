## Frozen design

### Patches
- **Pool.** The frozen test100 subset (`~/langres_data/arrays/test100_ids.json`).
- **Indices.** For each patch, compute mean NDVI and mean NDMI (definitions below).
- **Grid.** Cut NDVI and NDMI into tertiles over test100 to form a 3 × 3 grid.
- **Selection.**
  - Take one patch per grid cell: the one with the lowest per-patch sha256 from the frozen split file.
  - Then add the lowest-hash remaining patch from the most populated cell.
  - That gives 10 patches.
  - If a cell is empty, fill from the most populated cell and log it.
- **Anonymity.** Give each patch a random id (Python `uuid.uuid4` under `random.seed(0)`). Keep the id map only in `frontier_share/private/` (gitignored). Astra never sees a filename, tile, date or location.

### Renderings
Two 512 × 512 PNGs per patch from the 12-band arrays, bicubic upsampling from 120 × 120, with the frozen 2 to 98 percentile stretch from `LOG.md`:
- **Image 1, true color.** B04, B03, B02.
- **Image 2, false color.** B08, B04, B03.

### Index definitions (per pixel, ε = 1e-6 in the denominator)
- NDVI = (B08 − B04) / (B08 + B04)
- NDWI = (B03 − B08) / (B03 + B08)
- NDMI = (B8A − B11) / (B8A + B11)
- NDBI = (B11 − B08) / (B11 + B08)
- **Grid cells.** 40 × 40 pixels on a 3 × 3 grid.
- **Halves.** 60 pixel halves.
- **Medians.** Compute every median and threshold below on the RL pool (20,000 patches) and freeze it at FS-1.

### The 16 questions
Code computes the keys. `{…}` marks values the code fills from the RL pool and writes into the question text.

| Id | Tag | Question text | Key |
|---|---|---|---|
| Q1 | visible | Is the mean NDVI over the whole patch above 0.5, meaning most of the patch is dense, healthy vegetation? | mean NDVI > 0.5 |
| Q2 | visible | Is the mean NDVI over the whole patch below 0.2, meaning most of the patch is sparse or not vegetated? | mean NDVI < 0.2 |
| Q3 | visible | Is open water present, meaning that in at least one cell of a 3 × 3 grid more than 5% of pixels have NDWI above 0? | as stated |
| Q4 | visible | Is vegetation denser in the left half than in the right half (higher mean NDVI on the left)? | as stated |
| Q5 to Q10 | visible | Does the patch contain any {label}, following the CORINE-based BigEarthNet land cover labels? | the six most frequent RL pool labels, one per question |
| Q11 | shortwave | Is the mean NDMI (B8A and B11) above {RL pool median}, meaning the vegetation and soil are wetter than for a typical patch in this dataset? | as stated |
| Q12 | shortwave | Is the mean NDBI (B11 and B08) above 0, a sign of built-up or bare surfaces? | as stated |
| Q13 | shortwave | Is the mean NDMI higher in the top half than in the bottom half? | as stated |
| Q14 | texture | Is the near-infrared texture rougher than for a typical patch in this dataset, meaning the B08 spatial standard deviation exceeds {RL pool median}? | as stated |
| Q15 | texture | Does the patch carry more than 3 BigEarthNet land cover labels? | label count > 3 |
| Q16 | texture | Is the edge density higher than for a typical patch, meaning the mean absolute difference between neighboring B08 pixels exceeds {RL pool median}? | as stated |

- **Majority baseline.** The majority answer per question on the RL pool.

### Astra through Codex
- **Install.** Codex CLI on host A, headless `codex exec`.
- **Login.** I log in once with device authentication. When you need it, stop and ask me. Never handle my credentials yourself.
- **Confirm before any call, from `codex --version`, `codex exec --help` and the Codex config, and log each:**
  - the exact GPT-6 Astra model string
  - the flag that attaches images
  - the output schema option
  - the option that writes the last message to a file
  - the JSON event stream option
- **Settings for every call.**
  - Read-only sandbox, approvals never.
  - Web search off (never pass the search flag).
  - Skip the git repo check.
  - Reasoning effort high, set through the config override.
- **Working directory.** Run each call in a fresh empty temporary directory, so the model can read nothing else.
- **Tool check.** Save the JSON event stream. If a call shows any tool or command execution, log it, discard that output and count it as a failed call.
- **Concurrency.** At most 3 calls at once.
- **Usage limits.** On any usage-limit message, stop and report. Don't wait out the window.

### Caption call (one per patch, plus one retest per patch)

**Inputs.** Image 1 and image 2.

**Prompt (verbatim).**

> You are an expert in Earth observation and remote sensing. You see two renderings of the same Sentinel-2 L2A patch: 120 × 120 pixels at 10 m, so 1.2 km on each side, upsampled for display. North is up. Image 1 is true color (B04, B03, B02). Image 2 is false color (B08, B04, B03): near infrared appears as red, so healthy vegetation appears bright red.
>
> Answer from the two attached images only. Do not run commands or use tools. Do not name a location unless the image itself shows it, and use no outside data. Give every inference a confidence of low, medium or high.
>
> Work through these levels in order and fill the JSON schema.
>
> 1. Global scene. The dominant land covers with estimated area percentages summing to 100, the landscape type, cues of season or phenology, and overall heterogeneity.
> 2. Regional layout. Split the patch into a 3 × 3 grid: rows top, middle, bottom, and columns left, center, right. For each cell give the dominant cover, the texture and notable features. Then describe the boundaries and gradients across cells.
> 3. Local features. List each discrete feature, such as fields, roads, rivers, lakes, buildings, clearings, clouds and shadows, with its type, grid cell, approximate size in meters, shape, orientation and count.
> 4. Radiometric and spectral cues. Describe brightness, color and contrast in true color, and the strength and pattern of red in false color. State what they imply about vegetation vigor, moisture, water turbidity, bare soil and built surfaces. State what these two renderings cannot reveal.
> 5. Scientific interpretation. Land use and management (crop types, rotation, forestry, settlement), ecological state, active processes (irrigation, harvest, erosion, flooding, clearing), and the likely biome and climate setting. Give each claim its confidence and its evidence as a level and grid cell.
> 6. Uncertainties. Features you cannot resolve at 10 m, ambiguous classes and possible artifacts.
>
> Then write nested summaries for a scientist of at most 25, 50, 100, 200 and 400 words. Each summary stands alone in plain prose and carries the most informative content its budget allows. Longer summaries add finer detail and never contradict shorter ones. Return only JSON that matches the schema.

**Output schema** (`frontier_share/schemas/caption.json`, strict: every object sets `additionalProperties` to false and lists all properties as required):

```json
{
  "type": "object",
  "properties": {
    "global": {"type": "object", "properties": {
      "cover": {"type": "array", "items": {"type": "object", "properties": {
        "class": {"type": "string"}, "percent": {"type": "number"}}}},
      "landscape": {"type": "string"}, "season_cues": {"type": "string"}, "heterogeneity": {"type": "string"}}},
    "grid": {"type": "array", "items": {"type": "object", "properties": {
      "cell": {"type": "string", "enum": ["top_left", "top_center", "top_right", "middle_left", "middle_center",
                                         "middle_right", "bottom_left", "bottom_center", "bottom_right"]},
      "cover": {"type": "string"}, "texture": {"type": "string"}, "features": {"type": "string"}}}},
    "boundaries": {"type": "string"},
    "features": {"type": "array", "items": {"type": "object", "properties": {
      "type": {"type": "string"}, "cell": {"type": "string"}, "size_m": {"type": "string"},
      "shape": {"type": "string"}, "orientation": {"type": "string"}, "count": {"type": "integer"}}}},
    "spectral": {"type": "object", "properties": {
      "true_color": {"type": "string"}, "false_color": {"type": "string"},
      "implications": {"type": "array", "items": {"type": "object", "properties": {
        "claim": {"type": "string"}, "confidence": {"type": "string", "enum": ["low", "medium", "high"]}}}},
      "not_revealed": {"type": "string"}}},
    "interpretation": {"type": "array", "items": {"type": "object", "properties": {
      "claim": {"type": "string"}, "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
      "evidence": {"type": "string"}}}},
    "uncertainties": {"type": "array", "items": {"type": "string"}},
    "summaries": {"type": "object", "properties": {
      "w25": {"type": "string"}, "w50": {"type": "string"}, "w100": {"type": "string"},
      "w200": {"type": "string"}, "w400": {"type": "string"}}}
  }
}
```

**Validation.** Check every output against the schema and the word budgets. An invalid output counts as a failed call. Retry it once, within the cap.

### Image QA call (one per patch)

**Inputs.** The same two images.

**Prompt (verbatim).**

> You see two renderings of the same Sentinel-2 L2A patch: 120 × 120 pixels at 10 m, so 1.2 km on each side, upsampled for display. North is up. Image 1 is true color (B04, B03, B02). Image 2 is false color (B08, B04, B03): near infrared appears as red. Answer from the two attached images only. Do not run commands or use tools. Answer every question below with yes or no and a confidence of low, medium or high. When the images do not settle a question, give your best guess.
>
> {the 16 questions with their ids, in order}

**Schema** (`frontier_share/schemas/qa.json`, strict):
- `answers`: a list of 16 entries, each with `id`, `answer` (yes or no) and `confidence` (low, medium or high).

### Reader (open model, text only)
- **Model.** `microsoft/phi-4` at the revision already pinned in `model_revisions.json`, greedy decoding, on host A.
- **Prompt.** "Here is a description of a Sentinel-2 satellite image patch: {text}. Answer every question below from the description only, with yes or no. When the description does not settle a question, give your best guess. Return JSON with a list `answers` of 16 entries with `id` and `answer`." Follow it with the 16 questions.
- **Conditions per caption (six).**
  - each of the five summaries: w25, w50, w100, w200, w400,
  - the full structured JSON rendered as text.
- **Captions.** Run it on both captions of each patch (the main caption and the retest caption).

### Prithvi probe
- **Features.** Cached Prithvi features, standardized with RL pool statistics as in `LOG.md`. If the test features are missing for any of the 10 patches, compute them with `scripts/prithvi.py` and log it.
- **Fit.** For each question, fit a logistic regression on the RL pool (C = 1.0, max_iter 3000, standardized features), then predict the 10 patches.
- **Probe quality.** Report each probe's 5-fold cross-validated accuracy on the RL pool.

### Classification
Classify each (patch, question, condition) in this order:
1. **Said** if the reader is right.
2. **Seen, not said** if Astra with the images is right.
3. **Not seen** if the Prithvi probe is right.
4. **Neither** otherwise.

### Readouts
1. **Setup facts.** Codex version, model string, flags, effort, and calls used against the cap.
2. **Call ledger.** Latency per call, validity and failures.
3. **Main table.** Counts and shares of the four classes, by tag (visible, shortwave, texture) and by condition (w25 to w400, full JSON). Show the majority baseline beside it.
4. **Accuracy per answerer and tag.** Astra with the images, the reader per condition, and the Prithvi probe.
5. **Control check.** Shortwave questions (Q11 to Q13) should fall mostly into "not seen" or "neither", and NDVI questions (Q1, Q2, Q4) mostly into "said" or "seen, not said". Report the counts and a verdict.
6. **Test-retest.** Per condition, agreement of the reader's answers between the two captions, the difference in cover percentages, and the word Jaccard of the summaries.
7. **Caption checks by code.**
   - schema completeness,
   - number of local features,
   - place names found by a simple gazetteer scan,
   - Astra's top cover class against the BigEarthNet labels.
8. **Appendix.** All 10 main captions in full, and the summaries of the retest captions.
