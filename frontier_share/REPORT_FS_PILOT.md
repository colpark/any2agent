# Frontier share pilot (FS-0 to FS-4): report

## Deviations from DESIGN_FS_v1.md

| Id | Deviation | Reason |
|---|---|---|
| — | none | |

Bug fixes that restore v1 behaviour (see LOG.md):

| Id | Fix |
|---|---|
| FS-E1 | My reader parser expected ids written "Q5", but Phi-4 writes `5`. After the fix, the saved outputs were re-parsed without any new model call. |
| FS-E2 | 3 of 120 reader outputs hit my own 700-token limit before writing any JSON. Those 3 were rerun with a 2,000-token limit. Their decoding was unbatched, so it is not numerically identical to the batched run. |

Other notes:
- Selection cell (high NDVI, low NDMI) was empty. As the design specifies, it was filled from the most populated cell.
- `uuid.uuid4` ignores `random.seed(0)`, so the anonymous ids are not reproducible. The id map is saved.

## 1. Setup facts
- Codex: codex-cli 0.164.0, logged in through ChatGPT; no credential was handled.
- Model and effort: `-m gpt-6-astra`, reasoning effort high (`-c model_reasoning_effort="high"`).
- Call settings:
  - `codex exec --ignore-user-config --ignore-rules --ephemeral --skip-git-repo-check`
  - `-s read-only -c approval_policy="never" -c web_search="disabled"`
  - `--output-schema --json -o`, images via `-i`
  - a fresh empty working directory for each call
- Calls: **30 of the 35 allowed.** The breakdown was 10 main captions, 10 retest captions and 10 image QA calls. All 30 were valid on the first attempt, so none were retried. No call used a tool and no call hit a usage limit.
- Reader: microsoft/phi-4 at revision 2db69c1c, greedy decoding. 106 of 120 outputs are complete. The other 14 answer "unknown" to some questions; those answers count as wrong.
- Probes: 16 logistic regressions on RL-pool Prithvi features. 5-fold CV accuracy is 0.70 to 0.99.

## 2. Call ledger
Full per-call records are in `calls.jsonl`; the raw outputs are in `raw/`.

| Kind | Calls | Valid | Latency (s) | Output tokens (reasoning) |
|---|---|---|---|---|
| Caption, main | 10 | 10 | 107–134 | 4.1k–5.0k (260–1,034) |
| Caption, retest | 10 | 10 | 111–140 | 4.2k–5.1k (268–516) |
| Image QA | 10 | 10 | 13–23 | 0.4k–0.65k (169–427) |

## 3. Main table: four classes by condition
Each condition covers 160 items: 10 patches × 16 questions.

| Condition | Said | Seen, not said | Not seen | Neither | Reader acc | Majority acc |
|---|---|---|---|---|---|---|
| w25 | 92 (57%) | 48 (30%) | 14 (9%) | 6 (4%) | 0.57 | 0.64 |
| w50 | 91 (57%) | 45 (28%) | 18 (11%) | 6 (4%) | 0.57 | 0.64 |
| w100 | 88 (55%) | 45 (28%) | 21 (13%) | 6 (4%) | 0.55 | 0.64 |
| w200 | 85 (53%) | 48 (30%) | 19 (12%) | 8 (5%) | 0.53 | 0.64 |
| w400 | 87 (54%) | 49 (31%) | 17 (11%) | 7 (4%) | 0.54 | 0.64 |
| full JSON | 82 (51%) | 49 (31%) | 23 (14%) | 6 (4%) | 0.51 | 0.64 |

The same table split by tag is in `readouts_tables.md`. Shares of "said" at w100 are 63% for visible, 53% for shortwave and 30% for texture.

## 4. Accuracy per answerer and tag

| Tag (items) | Astra with images | Prithvi probe | Majority | Reader w25 | w50 | w100 | w200 | w400 | full JSON |
|---|---|---|---|---|---|---|---|---|---|
| all (160) | 0.72 | **0.85** | 0.64 | 0.57 | 0.57 | 0.55 | 0.53 | 0.54 | 0.51 |
| visible (100) | 0.77 | **0.86** | 0.66 | 0.64 | 0.64 | 0.63 | 0.60 | 0.61 | 0.57 |
| shortwave (30) | 0.63 | **0.77** | 0.60 | 0.57 | 0.57 | 0.53 | 0.53 | 0.57 | 0.53 |
| texture (30) | 0.63 | **0.90** | 0.60 | 0.37 | 0.33 | 0.30 | 0.30 | 0.30 | 0.30 |

## 5. Control check
Counts below are pooled over the six conditions.

| Group | Said | Seen, not said | Not seen | Neither | Expected | Verdict |
|---|---|---|---|---|---|---|
| Shortwave Q11–Q13 (180) | 99 | 56 | 19 | 6 | mostly not seen or neither | **fails** (14%) |
| NDVI Q1, Q2, Q4 (180) | 112 | 35 | 33 | 0 | mostly said or seen, not said | passes (82%) |

The shortwave control fails because "said" counts any correct reader answer, including lucky guesses.
- **The reader is at chance on shortwave.** Its shortwave accuracy is 0.53–0.57, below the 0.60 majority baseline. That is the accuracy of guessing, not of reading.
- **Two of the three shortwave questions have easy majority answers.** Q12 is "no" for 8 of 10 patches. The reader, which leans towards "no", gets those right without information.
- **Astra itself sits at the majority baseline on shortwave**: 0.63 against 0.60.

So the data agree with the premise that Astra does not see shortwave. The four-class rule cannot show it, because it credits chance as "said".

## 6. Test-retest

| Condition | Reader agreement, main vs retest | Summary word Jaccard |
|---|---|---|
| w25 | 0.82 | 0.40 |
| w50 | 0.88 | 0.34 |
| w100 | 0.87 | 0.31 |
| w200 | 0.94 | 0.34 |
| w400 | 0.92 | 0.36 |
| full JSON | 0.82 | 0.53 (whole JSON) |

Cover percentages are stable between the two captions:
- The mean total variation between the sorted cover vectors is 3.7 points; the worst patch differs by 9.
- The top class's percentage differs by 2.7 points on average.

Wording changes a lot between runs: summary word Jaccard is 0.31–0.40. The reader's answers mostly do not: agreement is 0.82–0.94.

## 7. Caption checks by code
- **Schema completeness:** all 20 captions name all 9 grid cells exactly once. Cover percentages sum to exactly 100 in every caption. Every summary is within its budget, at 78–100% of it.
- **Local features:** 6 to 12 per caption, 9.3 on average.
- **Place names:** the gazetteer scan found none in any of the 20 captions.
- **Top cover class vs BigEarthNet labels:** 8 of the 10 main captions match by keyword family. Both misses name "bare" ground as the top class:
  - ffa692a8, where the BigEarthNet labels are forest, shrub and water;
  - 354d86b7, where they are crop, forest, shrub and built.

  The retest caption of ffa692a8 adds forest and matches.

## Findings
1. **Astra's own words lose most of what it sees.** With the images, Astra answers 0.72 correctly. A text-only reader of its summaries scores 0.51–0.57, which is below the 0.64 majority baseline. Nearly a third of items (28–31%) are "seen, not said" at every budget.
2. **More words do not help, and the full JSON scores lowest.** Reader accuracy drifts from 0.57 at w25 to 0.51 for the full JSON. Two reader behaviours explain part of this:
   - It says "no" when the text is silent. Its yes-rate is 22–26%, while the keys say yes 41% of the time.
   - It answers "unknown" more often on long inputs: 35 of 320 answers for the full JSON.
   So this pilot measures the reader's conservatism as much as the content of the text.
3. **Texture is the clearest non-verbal gap.** The probe reaches 0.90 and Astra with the images 0.63; the reader from text reaches 0.30–0.37. Questions that compare to "a typical patch in this dataset" cannot be settled from prose. On Q14 at w100, the reader answered "no" for 8 patches and "unknown" for 2; the key is yes for 7.
4. **Prithvi beats Astra in every tag:** 0.85 vs 0.72 overall. Most of the 13-point gap comes from texture (0.90 vs 0.63) and shortwave (0.77 vs 0.63).
5. **Q4 (left vs right vegetation) is not a confidence failure.** I said earlier that Astra missed Q4 while mostly saying "high"; that was wrong. Of its 5 errors, 3 were "low", 1 "medium" and 1 "high".
6. **Q13 is hard for everyone.** Astra, the probe and the majority baseline each get 4 of 10. The probe's CV accuracy on Q13 is 0.70, so it gives the control little to work with.

## Limits
- n = 10 patches, so only 30 items per shortwave or texture cell. A difference of about 0.1 on one tag is within noise.
- The classification order credits chance-level reader answers as "said", which inflates "said" and breaks the shortwave control.
- The reader is one open model with a "no" bias. A different reader, or a "say unknown" option scored separately, would change the said and seen-not-said split.

These limits come from the frozen design. Changing any of them would need a CHANGE_REQUEST, and I have not drafted one.

## Appendix
The appendix lists all 10 main captions in full and the summaries of the 10 retest captions. It is generated from `raw/` and follows below.

### A1. Main captions (full JSON)

#### Patch fc0369af-2c2d-404e-af72-f649b48fd310

```json
{
 "global": {
  "cover": [
   {
    "class": "Open water (high confidence in class; medium confidence in estimated area)",
    "percent": 61
   },
   {
    "class": "Vegetated land, probably including woody vegetation (high confidence in vegetation; medium confidence in structure and estimated area)",
    "percent": 31
   },
   {
    "class": "Pale exposed substrate and sparsely vegetated shoreline, including mixed land-water pixels (medium confidence)",
    "percent": 8
   }
  ],
  "landscape": "An island-rich water landscape with irregular land fragments, narrow channels and a broad connected water area toward the center and south (high confidence). A coastal archipelago and an island-rich lake cannot be distinguished confidently (low confidence in either specific setting).",
  "season_cues": "Widespread red vegetation in false color indicates substantial near-infrared reflectance consistent with foliage being present (high confidence). A growing-season acquisition is plausible, but evergreen vegetation prevents reliable seasonal identification (low confidence). No extensive snow or ice is visible (high confidence).",
  "heterogeneity": "Strong spatial contrast between smooth dark water and mottled vegetated land, with numerous pale shoreline edges (high confidence). Land is concentrated in the northern portion and along the eastern and southwestern edges; connected water dominates much of the central and southern interior (high confidence)."
 },
 "grid": [
  {
   "cell": "top_left",
   "cover": "Vegetated island fragments interspersed with water (high confidence).",
   "texture": "Mottled green-gray land and red false-color vegetation contrast with smooth dark channels (high confidence).",
   "features": "Several irregular land fragments with pale margins; the largest cluster extends into middle_left (high confidence)."
  },
  {
   "cell": "top_center",
   "cover": "Mixed vegetated land and water, with the largest landmass occupying the eastern portion (high confidence).",
   "texture": "Coarsely mottled vegetation with scattered pale patches; comparatively uniform dark water between land fragments (high confidence).",
   "features": "A large landmass enters from the northern boundary and extends southward; a narrow channel separates it from western islands (high confidence)."
  },
  {
   "cell": "top_right",
   "cover": "Vegetated land on the west and dark water on the east, with additional land entering from the right boundary (high confidence).",
   "texture": "Patchy vegetation and pale shoreline rims beside smooth water (high confidence).",
   "features": "A broad channel runs approximately north-south between the main northern landmass and the eastern edge land (high confidence)."
  },
  {
   "cell": "middle_left",
   "cover": "Predominantly water with northern island margins and several small islets (high confidence).",
   "texture": "Mostly smooth dark water; small mottled land patches with light edges (high confidence).",
   "features": "The southern tip of the upper-left island cluster, several isolated islets and a compact dark patch within the northern land area (high confidence in visibility; low confidence in the dark patch's identity)."
  },
  {
   "cell": "middle_center",
   "cover": "Predominantly water, with the tip of the main northern landmass and a separate elongated island (high confidence).",
   "texture": "Smooth water interrupted by red, mottled vegetation and pale land margins (high confidence).",
   "features": "An elongated island lies southwest of the main landmass's southern tip; narrow water gaps separate these features (high confidence)."
  },
  {
   "cell": "middle_right",
   "cover": "Water surrounding several vegetated land fragments (high confidence).",
   "texture": "Strong contrast between dark water, red vegetation and pale irregular shorelines (high confidence).",
   "features": "A compact island near the cell center, small nearby islets and the northern portion of the southeastern landmass (high confidence)."
  },
  {
   "cell": "bottom_left",
   "cover": "Mostly water, with land entering from the left boundary and an elongated island near the bottom (high confidence).",
   "texture": "Dark smooth water and mottled land with conspicuous pale margins (high confidence).",
   "features": "An irregular southwestern edge landmass and a roughly east-west elongated island separated by water (high confidence)."
  },
  {
   "cell": "bottom_center",
   "cover": "Almost entirely open water (high confidence).",
   "texture": "Smooth dark blue-black appearance, with faint small tonal variations (high confidence).",
   "features": "A few tiny pale or greenish patches may be small rocks, shallow areas or mixed pixels (low confidence). No large landmass is visible (high confidence)."
  },
  {
   "cell": "bottom_right",
   "cover": "Vegetated land toward the east and open water toward the west and south (high confidence).",
   "texture": "Mottled red vegetation with a pale, indented edge against dark water (high confidence).",
   "features": "A large irregular landmass reaches the eastern image boundary; small detached patches occur near the southeastern edge (high confidence)."
  }
 ],
 "boundaries": "Land-water boundaries are mostly abrupt, irregular and locally fringed by pale cyan-gray pixels (high confidence). These fringes may combine exposed substrate, shallow water and spatial mixing (medium confidence). Narrow channels divide the northern land clusters, while water becomes more spatially continuous toward bottom_center (high confidence). There are no clearly resolved rectilinear field boundaries or regular settlement patterns (high confidence at the supplied resolution).",
 "features": [
  {
   "type": "Largest northern vegetated landmass; continuation beyond the image is unknown (high confidence)",
   "cell": "top_center, top_right, middle_center",
   "size_m": "Approximately 400–500 m wide and 550–600 m visible length (medium confidence)",
   "shape": "Irregular, elongated, with a tapering southern end (high confidence)",
   "orientation": "Approximately north-south (high confidence)",
   "count": 1
  },
  {
   "type": "Upper-left vegetated island complex; narrow internal separations are partly unresolved (high confidence)",
   "cell": "top_left, top_center, middle_left",
   "size_m": "Approximately 500 m across the main complex (medium confidence)",
   "shape": "Lobed and fragmented, with pale margins (high confidence)",
   "orientation": "Main southern cluster trends approximately west-east (medium confidence)",
   "count": 1
  },
  {
   "type": "Elongated central vegetated island (high confidence)",
   "cell": "middle_center",
   "size_m": "Approximately 100–160 m wide and 200–250 m long (medium confidence)",
   "shape": "Elongated oval with irregular edges (high confidence)",
   "orientation": "Northwest-southeast (medium confidence)",
   "count": 1
  },
  {
   "type": "Compact eastern vegetated island (high confidence)",
   "cell": "middle_right",
   "size_m": "Approximately 150–190 m across (medium confidence)",
   "shape": "Rounded to angular, with an indented perimeter (high confidence)",
   "orientation": "No strong long-axis orientation (medium confidence)",
   "count": 1
  },
  {
   "type": "Southeastern vegetated edge landmass (high confidence)",
   "cell": "middle_right, bottom_right",
   "size_m": "Approximately 300–350 m wide and 300–350 m visible length (medium confidence)",
   "shape": "Broad, lobed and irregular; clipped by the eastern boundary (high confidence)",
   "orientation": "Approximately north-south (medium confidence)",
   "count": 1
  },
  {
   "type": "Southwestern edge landmass (high confidence)",
   "cell": "bottom_left",
   "size_m": "Approximately 100–180 m visible width and 350–400 m visible length (medium confidence)",
   "shape": "Irregular and partly outside the patch (high confidence)",
   "orientation": "Approximately north-south (medium confidence)",
   "count": 1
  },
  {
   "type": "Elongated southern island (high confidence)",
   "cell": "bottom_left",
   "size_m": "Approximately 200–230 m long and 60–90 m wide (medium confidence)",
   "shape": "Low-aspect-width, irregular oval (high confidence)",
   "orientation": "Approximately east-west (high confidence)",
   "count": 1
  },
  {
   "type": "Connected open-water area with branching channels (high confidence)",
   "cell": "all cells",
   "size_m": "Extends across the 1,200 m patch; narrow visible gaps are roughly 20–100 m wide (medium confidence)",
   "shape": "Broad southern water area branching between northern land fragments (high confidence)",
   "orientation": "Multiple orientations; the eastern passage is broadly north-south (high confidence)",
   "count": 1
  },
  {
   "type": "Small detached islets or exposed rocks; approximate grouped count excludes major landmasses (medium confidence)",
   "cell": "top_left, top_center, middle_left, middle_center, middle_right, bottom_left, bottom_right",
   "size_m": "Approximately 20–100 m across (medium confidence)",
   "shape": "Rounded, oval or irregular patches (medium confidence)",
   "orientation": "Variable, without a consistent alignment (medium confidence)",
   "count": 15
  },
  {
   "type": "Compact dark patch within the western island complex; possible water or shadow (low confidence in identity)",
   "cell": "middle_left",
   "size_m": "Approximately 30–50 m across (low confidence)",
   "shape": "Compact and roughly rounded (medium confidence)",
   "orientation": "No clear orientation (medium confidence)",
   "count": 1
  }
 ],
 "spectral": {
  "true_color": "Water is very dark blue-green to nearly black; land is muted green, olive and gray, with brighter gray-beige margins and scattered interior patches (high confidence). The strongest contrast follows shorelines, while vegetation interiors show subtler mottling (high confidence).",
  "false_color": "Most substantial land patches appear moderately to strongly red, with spatial variation in saturation and brightness (high confidence). Water remains dark navy-black, and many land margins become pale cyan-gray (high confidence). Small islands contain both red centers and pale edges (high confidence).",
  "implications": [
   {
    "claim": "Vegetation occupies most island interiors; its red false-color response is consistent with substantial near-infrared scattering by foliage.",
    "confidence": "high"
   },
   {
    "claim": "Variation in red intensity may reflect vegetation cover, canopy structure, species, substrate exposure or illumination; it does not independently establish differences in vegetation health.",
    "confidence": "high"
   },
   {
    "claim": "Dark water is consistent with low near-infrared reflectance. No conspicuous bright sediment plume is visible.",
    "confidence": "high"
   },
   {
    "claim": "The water may have relatively little visible suspended sediment, but depth, dissolved organic matter and display scaling prevent a reliable turbidity estimate.",
    "confidence": "low"
   },
   {
    "claim": "Pale margins probably include exposed mineral substrate or sparse vegetation, although shallow water and mixed pixels may contribute.",
    "confidence": "medium"
   },
   {
    "claim": "The images do not establish vegetation moisture status, soil wetness or drought stress.",
    "confidence": "high"
   },
   {
    "claim": "There are no clearly resolved built surfaces; isolated pale patches cannot reliably be classified as buildings, roads or bare soil.",
    "confidence": "high"
   }
  ],
  "not_revealed": "These display renderings do not provide calibrated reflectance, reliable vegetation indices, water depth, salinity, quantitative turbidity, species identity, canopy height, soil moisture or change through time (high confidence). Their stretches may differ, so brightness cannot be compared quantitatively between renderings (high confidence)."
 },
 "interpretation": [
  {
   "claim": "The patch is dominated by a connected water system surrounding vegetated islands and edge landmasses.",
   "confidence": "high",
   "evidence": "Levels 1–3, all grid cells: dark connected water, repeated enclosed land outlines and irregular channels."
  },
  {
   "claim": "The visible landscape appears predominantly natural or weakly modified.",
   "confidence": "medium",
   "evidence": "Levels 2–3, top_left, top_center, middle_right and bottom_right: irregular shorelines and vegetation patterns, with no clear road network, field geometry or settlement."
  },
  {
   "claim": "Woody vegetation, potentially forest or shrub cover, is plausible on the larger landmasses.",
   "confidence": "medium",
   "evidence": "Levels 2 and 4, top_center, top_right and bottom_right: heterogeneous green-gray texture and widespread red false-color response. Individual crowns and heights are unresolved."
  },
  {
   "claim": "Crop type, rotation, forestry management and settlement activity cannot be established.",
   "confidence": "high",
   "evidence": "Levels 2–4, all land-bearing cells: no diagnostic agricultural parcels, harvest blocks or clearly resolved structures."
  },
  {
   "claim": "Most island interiors support substantial vegetation cover, but ecological condition and vegetation health remain undetermined.",
   "confidence": "high",
   "evidence": "Levels 1 and 4, top_left, top_center, middle_center and bottom_right: extensive red false-color response without temporal or calibrated spectral information."
  },
  {
   "claim": "Exposed shoreline substrate may reflect rocky margins, but substrate composition is uncertain.",
   "confidence": "medium",
   "evidence": "Levels 2–4, middle_center, middle_right and bottom_left: persistent pale rims and small pale detached patches."
  },
  {
   "claim": "No diagnostic evidence establishes active irrigation, harvest, clearing, erosion or flooding.",
   "confidence": "high",
   "evidence": "Levels 2–4, all cells: no clear irrigation geometry, cut blocks, sediment plumes or inundation reference. A single image cannot establish whether shorelines have changed."
  },
  {
   "claim": "A temperate or boreal wooded lake or archipelago is a plausible environmental setting, but biome and climate cannot be identified reliably.",
   "confidence": "low",
   "evidence": "Levels 1, 2 and 4, northern and eastern land cells: mottled vegetated islands with pale margins. These traits are not geographically or climatically unique."
  }
 ],
 "uncertainties": [
  "Cover percentages are visual estimates rather than measured segmentation; uncertainty is several percentage points, especially for pale margins and sparse vegetation (medium confidence).",
  "At 10 m, small buildings, narrow paths, individual trees and tiny channels may be unresolved or mixed within pixels (high confidence).",
  "Upsampling smooths boundaries without adding spatial information; exact island counts and narrow land connections remain uncertain (high confidence).",
  "Feature dimensions are approximate measurements from the stated patch extent, with greater relative uncertainty for objects only a few pixels across (high confidence).",
  "Pale shoreline pixels may represent rock, sediment, sparse vegetation, shallow bottoms or land-water mixing (medium confidence).",
  "The compact dark inland patch may be a small water body, canopy shadow or another low-reflectance surface (low confidence in any specific identification).",
  "Forest, shrubland and other dense vegetation cannot be separated reliably from these renderings alone (high confidence).",
  "Water depth, clarity, salinity and whether the system is coastal or inland remain unresolved (high confidence).",
  "No obvious cloud obscuration is visible, but small dark patches cannot be assigned confidently to cloud or terrain shadows (medium confidence).",
  "Season, climate, management history and active change cannot be established from this single paired observation (high confidence)."
 ],
 "summaries": {
  "w25": "Vegetated islands and irregular shores surround connected dark water (high confidence). Water covers approximately 61% (medium confidence); season and coastal versus inland setting remain unresolved.",
  "w50": "Connected dark water surrounds irregular vegetated islands (high confidence), occupying approximately 61% of the patch (medium confidence). Red false-color interiors indicate vegetation (high confidence); pale shorelines suggest exposed substrate or mixed pixels (medium confidence). Coastal versus inland setting, season, vegetation health and management remain unresolved (high confidence).",
  "w100": "The patch shows connected dark water surrounding irregular vegetated islands and edge landmasses (high confidence). Estimated cover is 61% water, 31% vegetation and 8% pale substrate or shoreline mixtures (medium confidence). Larger landmasses occupy the north and southeast; broad water dominates the southern center (high confidence). Red false-color interiors indicate vegetation, while pale margins may include exposed rock, sparse cover or shallow-water mixtures (medium confidence). No clear fields, roads or settlement patterns are resolved (high confidence). Season, water properties, vegetation health and coastal versus inland setting remain uncertain (high confidence).",
  "w200": "An island-rich water landscape dominates this 1.2 km patch (high confidence). Visual estimates assign 61% to water, 31% to vegetation and 8% to pale substrate, sparse cover and shoreline mixtures (medium confidence). The largest northern landmass extends southward between channels; smaller western islands and southeastern edge land surround a broad central-southern water area (high confidence).\n\nLand is mottled green-gray in true color and broadly red in false color, supporting substantial vegetation cover (high confidence). Woody vegetation is plausible, but forest and shrub cover cannot be separated reliably (medium confidence). Pale irregular margins suggest exposed substrate, potentially rock, with shallow water and pixel mixing also contributing (medium confidence).\n\nWater remains dark in both renderings, with no conspicuous sediment plume (high confidence); this does not establish depth, clarity or salinity. No clear agricultural parcels, road network, settlement or harvest blocks are resolved (high confidence). The scene appears predominantly natural or weakly modified (medium confidence). Season, ecological health, management and active shoreline change remain unresolved (high confidence). A temperate or boreal setting is plausible but weakly constrained (low confidence).",
  "w400": "The two renderings show an island-rich landscape with connected dark water, irregular vegetated land and pale shoreline margins (high confidence). Approximate cover is 61% water, 31% vegetation and 8% exposed substrate, sparse vegetation or shoreline mixtures (medium confidence). These are visual estimates, with uncertainty concentrated along narrow margins and small islands.\n\nThe largest northern landmass extends approximately 550–600 m into the patch, tapering southward (medium confidence in size). A fragmented western island complex is separated from it by narrow channels. An elongated central island trends northwest-southeast, a compact island occupies the middle-right area, and a larger southeastern landmass enters from the eastern boundary (high confidence in layout). The southwest contains an edge landmass and a separate east-west elongated island. Broad connected water dominates bottom_center (high confidence). Numerous small detached patches are likely islets or rocks, although their exact count is uncertain (medium confidence).\n\nVegetated interiors are muted green-gray in true color and moderately to strongly red in false color (high confidence). Their near-infrared response supports foliage presence but does not establish physiological health, moisture status or species (high confidence). Mottled texture suggests woody vegetation on larger landmasses (medium confidence). Pale cyan-gray margins may combine exposed rock or sediment, sparse vegetation, shallow bottoms and mixed pixels (medium confidence).\n\nWater is consistently dark, without a conspicuous bright sediment plume (high confidence). Turbidity, depth, salinity and whether the water is inland or coastal cannot be determined reliably (high confidence). No clear field boundaries, roads, settlement network or forestry cuts are resolved (high confidence); a predominantly natural or weakly modified landscape is therefore plausible (medium confidence).\n\nThere is no diagnostic evidence for active irrigation, harvest, clearing, erosion or flooding (high confidence in this observational limitation). A temperate or boreal wooded lake or archipelago is plausible, but biome and climate attribution have low confidence. Season is likewise weakly constrained because evergreen foliage could produce red land signatures outside summer. Upsampling does not resolve features smaller than the original 10 m pixels, limiting identification of buildings, paths, tiny channels and individual trees (high confidence)."
 }
}
```

#### Patch bcec7e27-2d53-4111-b902-8fca173ba248

```json
{
 "global": {
  "cover": [
   {
    "class": "Woody vegetation, including probable forest and scrub; medium confidence",
    "percent": 64
   },
   {
    "class": "Open water; high confidence",
    "percent": 23
   },
   {
    "class": "Low vegetation and probable wet shoreline vegetation; medium confidence",
    "percent": 9
   },
   {
    "class": "Sparse vegetation or exposed ground, potentially including rock; low confidence",
    "percent": 4
   }
  ],
  "landscape": "Vegetation-dominated shoreline mosaic with a large water body along the western edge and a smaller water area in the southeast; high confidence. A wooded landscape with locally wet openings is likely; medium confidence.",
  "season_cues": "Widespread red vegetation in false color indicates substantial near-infrared reflectance, consistent with foliage being present and photosynthetically active vegetation; medium confidence. Exact season, deciduous phenology and evergreen contribution are unresolved; low confidence.",
  "heterogeneity": "Strong water–land contrast, moderately mottled woody cover and narrow, brighter vegetation corridors produce substantial spatial heterogeneity; high confidence. Percentages are approximate visual estimates rather than a measured classification; medium confidence."
 },
 "grid": [
  {
   "cell": "top_left",
   "cover": "Predominantly open water, with wooded land entering from the right; high confidence.",
   "texture": "Water is smooth and very dark; land is mottled; high confidence.",
   "features": "An irregular shoreline runs approximately northeast–southwest through the eastern portion of the cell; high confidence."
  },
  {
   "cell": "top_center",
   "cover": "Predominantly woody vegetation, with water in the upper-left portion; medium confidence.",
   "texture": "Coarse mottling with red vegetation interspersed with muted gray-green patches in false color; high confidence.",
   "features": "An irregular wooded shore and scattered sparse-cover patches are visible; medium confidence."
  },
  {
   "cell": "top_right",
   "cover": "Predominantly woody vegetation; medium confidence.",
   "texture": "Mottled canopy with a narrow, relatively smooth diagonal interruption; high confidence.",
   "features": "A pale, locally dark-edged linear corridor trends northwest–southeast and exits the right edge; high confidence for its presence, low confidence that it is a track."
  },
  {
   "cell": "middle_left",
   "cover": "Open water and wooded shoreline occupy comparable portions; medium confidence.",
   "texture": "Smooth dark water contrasts with irregular vegetation and narrow shore strips; high confidence.",
   "features": "A constricted water passage separates the main eastern landmass from a western land projection; high confidence. A tiny bright shoreline or water-adjacent patch is unresolved; low confidence."
  },
  {
   "cell": "middle_center",
   "cover": "Mostly woody vegetation with small openings; medium confidence.",
   "texture": "Mottled vegetation crossed by lighter, discontinuous bands; high confidence.",
   "features": "An approximately east–west to northeast–southwest irregular band crosses the upper part, while brighter vegetation occurs toward the lower edge; medium confidence."
  },
  {
   "cell": "middle_right",
   "cover": "Predominantly woody vegetation with narrow low-vegetation bands; medium confidence.",
   "texture": "Dark, coarse patches bordered by stronger pink-red vegetation in false color; high confidence.",
   "features": "A bright vegetation strip follows the eastern margin and bends inward near the lower edge; high confidence. Several tiny bright spots cannot be classified; low confidence."
  },
  {
   "cell": "bottom_left",
   "cover": "Woody vegetation and western-edge water, with a lighter land patch near the bottom; medium confidence.",
   "texture": "Mottled red vegetation grades into muted gray-green cover near the bottom; high confidence.",
   "features": "The western water passage continues beyond the image; high confidence. A diffuse sparse-cover opening is probable; medium confidence."
  },
  {
   "cell": "bottom_center",
   "cover": "Woody vegetation with a conspicuous band of low vegetation; medium confidence.",
   "texture": "Coarse dark-red patches contrast with smoother, bright pink-red strips; high confidence.",
   "features": "An irregular branching vegetation corridor approaches the southeastern water area; high confidence. Wet ground within this corridor is plausible; medium confidence."
  },
  {
   "cell": "bottom_right",
   "cover": "Open water bordered by woody and low vegetation; high confidence.",
   "texture": "Smooth dark water, a muted blue-green northern margin and rough surrounding vegetation; high confidence.",
   "features": "A partially cropped pond- or lake-like water area has an irregular shoreline and a tiny bright interior patch; high confidence for the water, low confidence for the patch's identity."
  }
 ],
 "boundaries": "Water–land boundaries are generally sharp but locally softened by mixed shoreline pixels; high confidence. The western shore curves inward in middle_left, producing a narrow passage; high confidence. Woody-cover boundaries with openings are mostly irregular and diffuse, without a clear agricultural parcel pattern; medium confidence. Bright low-vegetation bands link parts of middle_left, bottom_center and the northern shore of bottom_right, with another strip along middle_right; high confidence for the pattern, medium confidence for ecological continuity. The top_right diagonal corridor is comparatively straight; high confidence.",
 "features": [
  {
   "type": "Western open-water body and constricted passage; high confidence",
   "cell": "top_left, top_center, middle_left, bottom_left",
   "size_m": "Visible length approximately 1200 m; width roughly 100–450 m; medium confidence",
   "shape": "Irregular, broad in the northwest and constricted centrally; high confidence",
   "orientation": "Broadly north–south; high confidence",
   "count": 1
  },
  {
   "type": "Southeastern open-water area; high confidence",
   "cell": "bottom_right",
   "size_m": "Visible footprint approximately 350 × 300 m, extending beyond the bottom edge; medium confidence",
   "shape": "Irregular rounded basin with indented margins; high confidence",
   "orientation": "Broadly north–south elongation within the visible portion; medium confidence",
   "count": 1
  },
  {
   "type": "Wooded western land projection; high confidence",
   "cell": "middle_left",
   "size_m": "Visible footprint approximately 180 × 250 m; medium confidence",
   "shape": "Rounded projection entering from the western edge; high confidence",
   "orientation": "Projects eastward; high confidence",
   "count": 1
  },
  {
   "type": "Narrow linear corridor, possibly a track or exposed-ground strip; low confidence for function",
   "cell": "top_center, top_right",
   "size_m": "Approximately 450–550 m long and 10–30 m wide; low confidence for width",
   "shape": "Mostly straight with a slight bend near the eastern edge; high confidence",
   "orientation": "Northwest–southeast; high confidence",
   "count": 1
  },
  {
   "type": "Discontinuous lighter band through woody cover; medium confidence",
   "cell": "middle_left, middle_center, middle_right",
   "size_m": "Approximately 850–1000 m long and 20–60 m wide; low confidence",
   "shape": "Irregular and gently sinuous; high confidence",
   "orientation": "Broadly west–east; high confidence",
   "count": 1
  },
  {
   "type": "Branching bright vegetation corridor, possibly wet ground or connected openings; medium confidence",
   "cell": "middle_left, bottom_center, middle_right, bottom_right",
   "size_m": "Visible network spans approximately 850–1000 m; individual bands roughly 20–100 m wide; low confidence",
   "shape": "Irregular, branching and locally expanded; high confidence",
   "orientation": "Mainly west–east with northeast–southwest branches; medium confidence",
   "count": 1
  },
  {
   "type": "Diffuse sparse-cover opening; medium confidence",
   "cell": "bottom_left",
   "size_m": "Approximately 200–300 m across, partly cropped; low confidence",
   "shape": "Irregular and diffuse; high confidence",
   "orientation": "No clear preferred orientation; medium confidence",
   "count": 1
  },
  {
   "type": "Tiny bright patch within southeastern water, possibly rock, emergent vegetation or a small islet; low confidence",
   "cell": "bottom_right",
   "size_m": "Approximately 20–40 m across; low confidence",
   "shape": "Compact and slightly elongated; low confidence",
   "orientation": "Unresolved; low confidence",
   "count": 1
  }
 ],
 "spectral": {
  "true_color": "The scene is relatively dark: water is nearly black to blue-green, woody cover is dark green and olive, and narrow openings are brighter green; high confidence. Scattered gray-brown patches interrupt the vegetation; high confidence. No large bright built surface or opaque cloud is apparent; medium confidence.",
  "false_color": "Most land becomes mottled red to dark crimson, while narrow shoreline and interior vegetation bands become vivid pink-red; high confidence. Water remains near-black to dark blue, with a more cyan northern margin in the southeastern water area; high confidence. Muted gray-green patches show weaker vegetation-like contrast; high confidence.",
  "implications": [
   {
    "claim": "Widespread red supports extensive foliage-bearing vegetation, but relative redness alone does not establish plant health or productivity.",
    "confidence": "high"
   },
   {
    "claim": "Brighter pink-red bands likely differ from surrounding woody cover in vegetation density, structure, species composition or illumination.",
    "confidence": "medium"
   },
   {
    "claim": "Some bright bands may represent moist low vegetation because they adjoin water and occupy connecting openings.",
    "confidence": "medium"
   },
   {
    "claim": "Dark, smooth areas with low near-infrared response are open water.",
    "confidence": "high"
   },
   {
    "claim": "The southeastern cyan margin could reflect shallow water, submerged material, suspended matter or shoreline mixing; turbidity cannot be isolated.",
    "confidence": "low"
   },
   {
    "claim": "Muted gray-brown land patches could contain exposed soil, rock or sparse vegetation; built surfaces are not securely identified.",
    "confidence": "low"
   }
  ],
  "not_revealed": "These display composites do not establish calibrated reflectance, quantitative vegetation indices, soil moisture, water depth, turbidity, species, crop identity or vegetation health; high confidence. Without shortwave-infrared bands, temporal observations and ancillary information, substrate composition, wetness and disturbance timing remain poorly constrained; high confidence."
 },
 "interpretation": [
  {
   "claim": "The patch is predominantly a wooded aquatic landscape.",
   "confidence": "high",
   "evidence": "Levels 1–2: widespread mottled vegetation across center and right cells, beside smooth dark water in the left column and bottom_right."
  },
  {
   "claim": "Some shoreline and interior openings plausibly support wetland or moist herbaceous vegetation.",
   "confidence": "medium",
   "evidence": "Levels 2–4: vivid pink-red strips in bottom_center, middle_right and along the northern margin of bottom_right, with irregular shapes and proximity to water."
  },
  {
   "claim": "Agricultural cultivation is not clearly expressed in the visible patch.",
   "confidence": "medium",
   "evidence": "Levels 2–3: center and right cells lack convincing rectangular fields, repeated crop rows or a regular parcel network."
  },
  {
   "claim": "A possible access corridor indicates limited human modification, but forestry management is unconfirmed.",
   "confidence": "low",
   "evidence": "Level 3: the narrow diagonal line in top_right could be a track; no associated clear-cut geometry or timber infrastructure is resolved."
  },
  {
   "claim": "No settlement can be identified reliably.",
   "confidence": "medium",
   "evidence": "Levels 2–4: tiny bright spots in middle_right lack resolved building shapes, associated streets or a coherent built footprint."
  },
  {
   "claim": "Vegetation is extensive and spectrally heterogeneous, but ecological condition and degradation cannot be established.",
   "confidence": "high",
   "evidence": "Levels 1 and 4: widespread red vegetation with variable brightness across all land cells; no temporal or calibrated health evidence is available."
  },
  {
   "claim": "The openings provide insufficient evidence to diagnose active clearing, harvest, irrigation, erosion or flooding.",
   "confidence": "high",
   "evidence": "Levels 2–4: irregular lighter patches and bright vegetation bands have multiple plausible explanations, and no diagnostic machinery, sediment plume, irrigation layout or dated change is visible."
  },
  {
   "claim": "A forest–freshwater or forest–wetland biome setting is plausible, but salinity and climate zone are unresolved.",
   "confidence": "low",
   "evidence": "Levels 1–4: wooded shorelines and small water areas are compatible with multiple climatic and hydrological settings; no unique geographic or climatic indicator appears."
  }
 ],
 "uncertainties": [
  "High confidence: cover percentages are visual approximations, with potentially several percentage points of error per class.",
  "High confidence: upsampling adds display pixels without improving the native 10 m spatial resolution.",
  "High confidence: individual trees, narrow channels, small buildings, road surfaces and small shoreline objects are not reliably resolved.",
  "Medium confidence: dark land patches may combine canopy composition, canopy shadow and lower vegetation cover.",
  "High confidence: bright red is a relative composite response, not a direct measurement of health, moisture or productivity.",
  "High confidence: low vegetation, scrub, wetland vegetation and open woodland can overlap spectrally and spatially.",
  "High confidence: pale patches cannot be securely separated into rock, bare soil, senescent vegetation or built material.",
  "High confidence: water bodies continue beyond the patch, so their full extent and external connectivity are unknown.",
  "High confidence: no visible connecting channel establishes whether the western and southeastern waters are hydrologically linked.",
  "Medium confidence: shoreline color fringes and tiny bright spots may be affected by mixed pixels, resampling and display contrast.",
  "Medium confidence: no obvious opaque cloud obscures the patch, but subtle haze or cloud shadow cannot be excluded from these renderings alone.",
  "High confidence: exact season, crop rotation, forestry history and active environmental change require information beyond a single paired rendering."
 ],
 "summaries": {
  "w25": "Wooded shores surround western and southeastern water areas (high confidence). Bright vegetation bands may mark wet openings (medium confidence); land management remains unresolved.",
  "w50": "This wooded aquatic mosaic contains approximately 64% woody cover and 23% water (medium confidence). Western water narrows centrally; another water area occupies the southeast (high confidence). Bright red vegetation bands may indicate moist openings (medium confidence). A possible northeastern track is uncertain (low confidence); management and climate remain unresolved.",
  "w100": "The patch is a wooded shoreline mosaic (high confidence), with estimated cover of 64% woody vegetation, 23% water, 9% low vegetation and 4% sparse cover or exposed ground (medium confidence overall). Western water narrows into a passage, while a smaller water area occupies the southeast (high confidence). Vivid false-color vegetation bands cross the lower center and border water; some may be moist openings (medium confidence). A northeastern diagonal corridor could be a track (low confidence). Agriculture, settlement and active disturbance are not securely identified (medium confidence). Exact season, climate and vegetation health remain unresolved (high confidence).",
  "w200": "A wooded aquatic landscape dominates the patch (high confidence). Visual estimates allocate 64% to woody vegetation, 23% to water, 9% to low vegetation and 4% to sparse cover or exposed ground (medium confidence overall). Open water occupies much of the western edge, narrowing around a western land projection; another irregular water area is cropped by the southeastern boundary (high confidence).\n\nLand has a mottled dark-green appearance in true color and extensive crimson-red response in false color (high confidence). Brighter pink-red strips cross the lower center and follow parts of the eastern and southeastern shores. These likely represent vegetation differing in structure or density, with moist herbaceous cover plausible in some strips (medium confidence).\n\nA diagonal northeastern corridor could be an access track (low confidence). There is no convincing field pattern or settlement footprint (medium confidence). Gray-brown openings cannot be assigned confidently to soil, rock or sparse vegetation (low confidence). The southeastern water's cyan margin has ambiguous causes, including shallow water and mixed shoreline pixels (low confidence). These images cannot establish active flooding, clearing, harvest, ecological health, exact season or climate zone (high confidence).",
  "w400": "The scene is a wooded shoreline mosaic containing substantial open water (high confidence). Approximate visual cover estimates are 64% woody vegetation, 23% water, 9% low vegetation and 4% sparse vegetation or exposed ground (medium confidence overall). These are interpretive estimates rather than pixel-derived measurements.\n\nWater dominates the northwest and continues down the western edge, constricting around an eastward-projecting land fragment in the middle-left cell (high confidence). Its visible length spans approximately 1.2 km, with widths roughly 100–450 m (medium confidence). A separate visible water area occupies the southeast and extends beyond the lower boundary; its visible footprint is approximately 350 by 300 m (medium confidence). Connectivity between these waters is unresolved (high confidence).\n\nMost central and eastern land is mottled, consistent with woody vegetation (medium confidence). Narrow bright vegetation strips cross the lower center and border the southeastern water, while another strip follows the eastern margin (high confidence). Moist herbaceous vegetation or wetland margins are plausible interpretations of some strips (medium confidence). A diffuse lighter opening occurs near the southwestern bottom edge (medium confidence). A roughly 450–550 m diagonal corridor in the northeast could be an access track, but its function is uncertain (low confidence).\n\nTrue color shows dark-green and olive vegetation, nearly black water, brighter green strips and scattered gray-brown patches (high confidence). False color shows extensive crimson vegetation with vivid pink-red bands and dark water (high confidence). Redness supports foliage-bearing cover but cannot independently establish health or productivity (high confidence). The southeastern water's cyan northern margin could reflect shallow water, suspended material, submerged cover or mixed pixels (low confidence).\n\nNo convincing agricultural parcel system or settlement footprint is apparent (medium confidence). Forestry, crop rotation and active irrigation, harvest, clearing, erosion or flooding cannot be diagnosed from these observations (high confidence). A forest–wetland setting is plausible, but climate, salinity and biome subtype remain uncertain (low confidence). Native 10 m resolution limits interpretation of tiny bright spots, narrow channels, individual trees and small structures (high confidence)."
 }
}
```

#### Patch ffa692a8-0cb0-4c93-9fd4-d57a22d5294f

```json
{
 "global": {
  "cover": [
   {
    "class": "Bright white/cyan material: probable cloud, with snow/ice an unresolved alternative; medium confidence in interpretation, low confidence in percentage",
    "percent": 46
   },
   {
    "class": "Gray-blue textured areas: visible surface or shaded material of unresolved land-cover type; high confidence in appearance, low confidence in identity and percentage",
    "percent": 45
   },
   {
    "class": "Dark blue-gray bands: probable shadows or shaded boundaries; medium confidence in interpretation, low confidence in percentage",
    "percent": 9
   }
  ],
  "landscape": "Medium confidence: a strongly heterogeneous scene with extensive bright material and darker textured areas. Cloud obscuration is plausible and provisionally favored, but snow/ice and terrain shading cannot be excluded. Low confidence: the underlying landscape type is indeterminate.",
  "season_cues": "High confidence: neither rendering shows a strong vegetation signal. Low confidence: this could reflect obscuration, sparse vegetation, or a snow-covered season; no specific season or phenological stage is defensible.",
  "heterogeneity": "High confidence: strong brightness contrasts occur over tens to hundreds of meters, with irregular bright patches, granular gray areas, and narrow dark margins. Medium confidence: obscuration and shading contribute substantially to the apparent heterogeneity."
 },
 "grid": [
  {
   "cell": "top_left",
   "cover": "High confidence: bright material along the upper edge and gray-blue material below; medium confidence: cloud and a shadowed or exposed surface.",
   "texture": "High confidence: smooth, nearly saturated white above a granular gray area.",
   "features": "High confidence: a curved dark band borders the upper bright patch; a narrow bright indentation occurs toward the right edge."
  },
  {
   "cell": "top_center",
   "cover": "High confidence: gray textured material on the left and broad white/cyan material on the right; low confidence in physical cover identity.",
   "texture": "High confidence: mottled gray texture contrasts with smoother bright material.",
   "features": "High confidence: an irregular, approximately north–south transition crosses the cell, with dark segments along its western and lower margins."
  },
  {
   "cell": "top_right",
   "cover": "High confidence: extensive bright white/cyan material with a gray patch near the upper-right edge.",
   "texture": "High confidence: mostly smooth brightness with localized coarse gray mottling.",
   "features": "High confidence: a small bright cap touches the top edge, and darker material extends beneath it; medium confidence: these may be separate cloud and shadow elements."
  },
  {
   "cell": "middle_left",
   "cover": "High confidence: interlocking bright white/cyan and gray-blue patches.",
   "texture": "High confidence: irregular lobes and granular gray interiors.",
   "features": "High confidence: bright material crosses diagonally through the cell, separating several darker patches. Low confidence: these gaps do not establish a river or road."
  },
  {
   "cell": "middle_center",
   "cover": "High confidence: predominantly gray textured material, bordered by bright areas toward the left and lower edge.",
   "texture": "High confidence: fine mottling with darker, locally angular margins.",
   "features": "High confidence: a broad gray area extends toward the lower right. Low confidence: individual objects within it cannot be identified."
  },
  {
   "cell": "middle_right",
   "cover": "High confidence: white/cyan material interspersed with gray areas and dark blue-gray bands.",
   "texture": "High confidence: smooth bright areas surround irregular mottled patches.",
   "features": "High confidence: a pronounced, roughly east–west dark band crosses the upper part; medium confidence: it is a shadow or shaded boundary."
  },
  {
   "cell": "bottom_left",
   "cover": "High confidence: gray-blue material and dark bands, with bright material near the upper-right and left margins.",
   "texture": "High confidence: granular gray texture and a broad, relatively smooth dark margin.",
   "features": "High confidence: a dark diagonal band extends toward the bottom center; medium confidence: it belongs to a shadow system."
  },
  {
   "cell": "bottom_center",
   "cover": "High confidence: a large bright patch above darker gray-blue material.",
   "texture": "High confidence: a smooth white interior, cyan fringe, and mottled lower portion.",
   "features": "High confidence: a conspicuous curved dark margin lies below the bright patch; medium confidence: the pairing supports a cloud-shadow interpretation."
  },
  {
   "cell": "bottom_right",
   "cover": "High confidence: mixed gray-blue material and bright patches along the right edge.",
   "texture": "High confidence: mottled background with diffuse cyan-white areas.",
   "features": "High confidence: an irregular bright patch enters from the right, with a dark margin above or beside it. Low confidence: no distinct building, water body, or field is resolved."
  }
 ],
 "boundaries": "High confidence: boundaries are predominantly irregular, lobate, and locally angular, with white interiors grading through cyan into gray or dark blue. Several bright patches have dark margins on their lower sides, especially in top_left and bottom_center. Medium confidence: this repeated association favors illumination or cloud-shadow effects over a mapped land-use boundary. Low confidence: a consistent shadow displacement, terrain gradient, or drainage direction cannot be established.",
 "features": [
  {
   "type": "Bright patch, probable cloud; medium confidence",
   "cell": "top_left",
   "size_m": "Approximately 350–450 long and 100–180 wide; low confidence",
   "shape": "Rounded, edge-truncated patch; high confidence",
   "orientation": "Approximately east–west; medium confidence",
   "count": 1
  },
  {
   "type": "Large bright complex, probable cloud with snow/ice alternative; medium confidence",
   "cell": "top_center, top_right, middle_right",
   "size_m": "Approximately 400–650 across; low confidence",
   "shape": "Irregular connected or overlapping lobes, truncated at the top; high confidence",
   "orientation": "Broadly north–south; medium confidence",
   "count": 1
  },
  {
   "type": "Small bright edge patch, possible cloud; low confidence in identity",
   "cell": "top_right",
   "size_m": "Approximately 100–180 across; low confidence",
   "shape": "Rounded and truncated by the top edge; high confidence",
   "orientation": "No clear long axis; medium confidence",
   "count": 1
  },
  {
   "type": "Bright branching complex, possible cloud or snow-covered material; low confidence in identity",
   "cell": "middle_left",
   "size_m": "Approximately 300–450 across; low confidence",
   "shape": "Lobate and branching around gray inclusions; high confidence",
   "orientation": "Broadly southwest–northeast; medium confidence",
   "count": 1
  },
  {
   "type": "Bright patch with cyan fringe and dark lower margin, probable cloud; medium confidence",
   "cell": "bottom_left, bottom_center",
   "size_m": "Approximately 400–500 long and 200–300 wide; low confidence",
   "shape": "Irregular broad lobe with small darker inclusions; high confidence",
   "orientation": "Approximately east–west; medium confidence",
   "count": 1
  },
  {
   "type": "Bright edge patch, possible cloud; low confidence in identity",
   "cell": "middle_right, bottom_right",
   "size_m": "Approximately 150–250 across; low confidence",
   "shape": "Irregular and truncated by the right edge; high confidence",
   "orientation": "No reliable long axis; low confidence",
   "count": 1
  },
  {
   "type": "Prominent dark boundary bands, probable shadows; medium confidence; count is approximate",
   "cell": "top_left, top_center, middle_right, bottom_center",
   "size_m": "Approximately 150–500 long and 20–80 wide; low confidence",
   "shape": "Curved or segmented bands adjoining bright patches; high confidence",
   "orientation": "Variable, with several approximately east–west segments; medium confidence",
   "count": 4
  }
 ],
 "spectral": {
  "true_color": "High confidence: the image is dominated by nearly saturated white, pale cyan, mottled gray-blue, and dark blue-gray. Bright regions have little visible internal detail, while darker regions show coarse granular texture. Green and brown tones are weak or absent.",
  "false_color": "High confidence: the same bright and dark structures persist. Cyan fringes are conspicuous, and gray areas have muted bluish or slightly purplish tones. There are no extensive bright-red patches indicating an unambiguous strong near-infrared vegetation response.",
  "implications": [
   {
    "claim": "The renderings do not provide a clear signal of vigorous exposed vegetation; obscuration prevents interpreting this as vegetation absence.",
    "confidence": "high"
   },
   {
    "claim": "Bright lobate patches with adjacent dark margins are consistent with clouds and shadows, although snow/ice remains a credible alternative.",
    "confidence": "medium"
   },
   {
    "claim": "Cyan coloration alone cannot establish surface wetness or distinguish cloud, snow, ice, and display effects.",
    "confidence": "high"
   },
   {
    "claim": "No water body is securely identified, so water turbidity and flooding cannot be assessed.",
    "confidence": "high"
   },
   {
    "claim": "Gray textured areas could include rock, sparse vegetation, built surfaces, or shaded snow/ice; their color does not securely distinguish these classes.",
    "confidence": "low"
   },
   {
    "claim": "Bare-soil exposure, soil moisture, and built-surface abundance cannot be estimated reliably from these renderings.",
    "confidence": "high"
   }
  ],
  "not_revealed": "High confidence: these display images do not provide calibrated reflectance, quantitative vegetation indices, cloud height, snow depth, surface temperature, soil moisture, water depth, or reliable material composition. They also cannot establish temporal change, crop identity, or management history."
 },
 "interpretation": [
  {
   "claim": "Cloud obscuration and associated shadows are the provisional explanation for much of the scene's brightness pattern.",
   "confidence": "medium",
   "evidence": "Levels 2–4; top_left, top_center, middle_right, and bottom_center: bright lobes, cyan fringes, and repeated adjacent dark bands."
  },
  {
   "claim": "Snow/ice with shaded or exposed rough terrain remains an alternative scene interpretation.",
   "confidence": "low",
   "evidence": "Levels 1–4; all cells: extensive white/cyan material, gray rough areas, and weak red response are compatible with this alternative but are not diagnostic."
  },
  {
   "claim": "Agricultural land use, crop types, crop rotation, and field management cannot be established.",
   "confidence": "high",
   "evidence": "Levels 2–3; middle_left, middle_center, and bottom_center: no securely identifiable field boundaries, crop rows, or irrigation structures."
  },
  {
   "claim": "Forestry and settlement cannot be distinguished confidently within the gray textured areas.",
   "confidence": "high",
   "evidence": "Levels 2–4; top_left, middle_center, and bottom_right: granular texture without resolved tree crowns, building footprints, or a convincing road network."
  },
  {
   "claim": "Vegetation vigor and ecological condition remain indeterminate.",
   "confidence": "high",
   "evidence": "Level 4; all cells: little conspicuous red in false color, combined with extensive bright material and shading."
  },
  {
   "claim": "Irrigation, harvest, erosion, flooding, and clearing are not demonstrable.",
   "confidence": "high",
   "evidence": "Levels 2–4; all cells: no securely identified process-specific morphology and no temporal comparison."
  },
  {
   "claim": "A biome or climate setting cannot be assigned reliably.",
   "confidence": "high",
   "evidence": "Levels 1 and 4; all cells: cloud versus snow/ice ambiguity and unresolved underlying land cover. Even confirmed snow would not uniquely identify climate or biome."
  }
 ],
 "uncertainties": [
  "High confidence: the main ambiguity is whether bright material is cloud, surface snow/ice, or a mixture.",
  "High confidence: the percentages describe apparent image classes, not a validated inventory of underlying land cover.",
  "Low confidence: class percentages may differ by roughly 10–15 percentage points because cyan fringes and shaded regions have ambiguous membership.",
  "High confidence: display upsampling does not add information beyond the native 10 m sampling.",
  "High confidence: small buildings, individual trees, narrow roads, drainage channels, and crop rows cannot be reliably resolved here.",
  "Medium confidence: interpolation, strong display stretching, or saturation may contribute to cyan rims and loss of bright-area detail.",
  "Medium confidence: dark bands may combine cast shadows, terrain shading, and darker surface materials.",
  "High confidence: feature dimensions and counts describe visible patch groupings; they do not establish separate physical clouds or surface objects.",
  "High confidence: scene-edge truncation makes the full size and connectivity of several bright features unknown.",
  "High confidence: a single acquisition cannot establish change, seasonality, management history, or active surface processes."
 ],
 "summaries": {
  "w25": "Bright patches and gray textured areas dominate. Cloud/shadow interpretation has medium confidence; snow/ice remains possible. Underlying land cover and vegetation condition are unresolved.",
  "w50": "The patch contains bright white/cyan material, gray textured areas, and dark boundary bands. Clouds and shadows are provisionally favored with medium confidence; snow/ice remains possible. No strong red vegetation response is visible with high confidence. Land use, ecological condition, season, and climate cannot be reliably assigned.",
  "w100": "Bright white/cyan material occupies approximately 46%, gray textured areas 45%, and dark bands 9%, with low confidence in these visual estimates. Clouds and shadows are provisionally favored with medium confidence because bright lobes repeatedly adjoin dark margins, especially at top left and bottom center. Snow/ice remains an unresolved alternative. High confidence: false color shows no extensive bright-red vegetation response, but obscuration prevents conclusions about vegetation absence or health. High confidence: roads, fields, buildings, and water bodies are not securely identified, leaving land use, ecological state, active processes, biome, and climate unresolved.",
  "w200": "The 1.2 km patch is strongly heterogeneous, with high confidence in its visible pattern but limited confidence in physical interpretation. Estimated apparent cover is 46% bright white/cyan material, 45% gray textured areas, and 9% dark bands; these percentages have low confidence and do not describe validated ground cover.\n\nClouds and associated shadows are provisionally favored with medium confidence. Bright lobes and neighboring dark margins recur at top left, upper center/right, middle right, and bottom center. Snow/ice and terrain shading remain an alternative with low confidence. The central gray area is granular, while bright interiors are comparatively smooth and nearly saturated, with high confidence in those observations.\n\nHigh confidence: false color contains no extensive strong red response. This limits evidence for vigorous exposed vegetation but cannot establish vegetation absence, dormancy, or stress. High confidence: no convincing field system, road network, settlement, or water body is resolved. Consequently, crop identity, forestry, irrigation, harvest, flooding, erosion, ecological condition, and climate remain indeterminate. High confidence: native 10 m sampling, display enlargement, bright-area saturation, and uncertain obscuration restrict object identification and feature measurements.",
  "w400": "The scene contains extensive bright white/cyan patches interspersed with gray-blue textured areas and dark boundary bands. High confidence applies to this visual description. Low-confidence area estimates assign 46% to bright material, 45% to gray textured areas, and 9% to dark bands. These are apparent image classes, not a reliable inventory of underlying land cover.\n\nCloud obscuration and associated shadows are provisionally favored with medium confidence. At top left, a bright patch borders a curved dark band. A larger bright complex extends through the upper center and right. Middle left contains interlocking bright and gray lobes, while middle center contains a broad granular gray region. Bottom center has a conspicuous white patch with a cyan fringe and dark lower margin. These spatial observations have high confidence. Snow/ice adjoining shaded or exposed terrain remains an alternative physical interpretation with low confidence.\n\nThe major bright groupings span roughly 100–650 m, with low confidence in their dimensions and separation. High confidence: boundaries are irregular and often lobate, and no convincing field system, road network, individual building, or water body is identifiable. Dark bands could include cast shadows or terrain shading, with medium confidence in that ambiguity.\n\nHigh confidence: both renderings retain similar bright and dark structures, and false color shows no extensive bright-red vegetation response. This supports only the absence of a clear exposed vigorous-vegetation signal. It does not establish vegetation absence, poor health, or dormancy. High confidence: moisture, water turbidity, bare-soil abundance, and built-surface abundance cannot be assessed securely.\n\nHigh confidence: crop types, rotation, forestry, settlement, irrigation, harvest, clearing, erosion, flooding, ecological condition, biome, and climate are unresolved. The principal limitations are uncertain cloud versus snow/ice identity, native 10 m sampling, edge-truncated features, and bright-area loss of detail. Medium confidence: display stretching and interpolation may accentuate cyan fringes. High confidence: a single pair of rendered band combinations cannot establish temporal change or quantitative surface properties."
 }
}
```

#### Patch 354d86b7-c2c2-4231-854d-767edc81e5bb

```json
{
 "global": {
  "cover": [
   {
    "class": "Exposed rock or soil with sparse, dry-looking vegetation; medium confidence",
    "percent": 56
   },
   {
    "class": "Distinct green vegetation patches; high confidence",
    "percent": 12
   },
   {
    "class": "Deep shadow or otherwise dark, obscured terrain; medium confidence",
    "percent": 32
   }
  ],
  "landscape": "Apparently rugged terrain with elongated ridges, intervening depressions and scattered vegetation; medium confidence. Percentages are approximate visible-appearance fractions, with shadow treated separately because underlying cover is obscured.",
  "season_cues": "Brown background surfaces contrast with localized green vegetation, consistent with limited vegetation cover or seasonal dryness; medium confidence. A particular season or phenological stage cannot be assigned; low confidence.",
  "heterogeneity": "Strong heterogeneity in illumination, surface color and vegetation density, organized into diagonal terrain bands and irregular vegetation patches; high confidence."
 },
 "grid": [
  {
   "cell": "top_left",
   "cover": "Predominantly brown exposed or sparsely vegetated surfaces; medium confidence.",
   "texture": "Coarse, mottled surfaces interrupted by a dark diagonal band; high confidence.",
   "features": "An apparent ridge flank and adjacent shadow trend northwest–southeast; medium confidence. Small vegetation patches occur near the left edge; high confidence."
  },
  {
   "cell": "top_center",
   "cover": "Brown terrain with a conspicuous green vegetation strip; high confidence.",
   "texture": "Irregular, banded and locally rough; high confidence.",
   "features": "A vegetation strip bends southeastward across the cell and becomes bright red in false color; high confidence. Dark depressions occur toward the upper edge; medium confidence."
  },
  {
   "cell": "top_right",
   "cover": "Mixed brown terrain, scattered vegetation and dark areas; high confidence.",
   "texture": "Rough and dissected, with narrow vegetation streaks; medium confidence.",
   "features": "Several elongated red patches in false color follow apparent terrain recesses; medium confidence. A large dark zone enters near the lower-right boundary; high confidence."
  },
  {
   "cell": "middle_left",
   "cover": "Mostly brown terrain with a compact green patch and substantial shadow; high confidence.",
   "texture": "Mottled illuminated surfaces separated by smooth-looking dark bands; high confidence.",
   "features": "A compact vegetation patch straddles the upper boundary; high confidence. A dark wedge crosses the western side; high confidence."
  },
  {
   "cell": "middle_center",
   "cover": "Brown terrain in the upper part and a broad dark zone below; high confidence.",
   "texture": "Irregular rocky-looking texture grading into very low brightness; medium confidence.",
   "features": "Small bright-red vegetation patches lie near the upper-left side of the dark boundary; high confidence. An apparent projecting slope or spur borders the shadow; medium confidence."
  },
  {
   "cell": "middle_right",
   "cover": "Predominantly dark, obscured terrain; high confidence.",
   "texture": "Mostly low contrast, with faint brown surface detail and a narrow brighter band; high confidence.",
   "features": "A large dark region connects with the central diagonal band; high confidence. A brown transverse feature may be an illuminated terrain rib; medium confidence."
  },
  {
   "cell": "bottom_left",
   "cover": "Dark terrain above and brown, sparsely vegetated surfaces below; high confidence.",
   "texture": "Broad shadow adjoining rough, locally banded surfaces; high confidence.",
   "features": "The main dark corridor reaches the western edge; high confidence. A small green patch touches the left edge near the cell's upper boundary; high confidence."
  },
  {
   "cell": "bottom_center",
   "cover": "Brown terrain with a distinct vegetation patch and adjacent shadow; high confidence.",
   "texture": "Rough, with several approximately parallel terrain bands; medium confidence.",
   "features": "A compact green patch, red in false color, occupies the upper-right portion; high confidence. Curved bands below it suggest slope structure; medium confidence."
  },
  {
   "cell": "bottom_right",
   "cover": "Extensive dark terrain bordered by brown surfaces; high confidence.",
   "texture": "Low-contrast interior with irregular, brighter margins; high confidence.",
   "features": "An elongated dark lobe extends southward, bordered by an apparent terrain rib; medium confidence. No discrete constructed feature is confidently resolved; high confidence."
  }
 ],
 "boundaries": "High-confidence brightness boundaries separate illuminated brown surfaces from connected dark zones. The largest dark band runs southwest–northeast through the middle and broadens toward the right edge. Additional northwest–southeast bands occur in the upper-left portion. Their irregular margins and association with textured slopes favor terrain-controlled illumination; medium confidence. Vegetation boundaries are mostly irregular, with no clear parcel grid; high confidence.",
 "features": [
  {
   "type": "Connected dark corridor, probably terrain shadow; medium confidence",
   "cell": "bottom_left, middle_center, middle_right",
   "size_m": "Approximately 1,200 m visible length and 120–300 m width; medium confidence",
   "shape": "Irregular, branching band; high confidence",
   "orientation": "Southwest–northeast; high confidence",
   "count": 1
  },
  {
   "type": "Additional elongated dark bands, probably terrain shadows; medium confidence",
   "cell": "top_left, middle_left, bottom_right",
   "size_m": "Approximately 350–800 m long and 60–220 m wide; low confidence",
   "shape": "Tapered bands or lobes, partly connected to other dark areas; medium confidence",
   "orientation": "Mostly northwest–southeast, with a more north–south lobe at lower right; medium confidence",
   "count": 3
  },
  {
   "type": "Elongated vegetation patch; high confidence",
   "cell": "top_center",
   "size_m": "Approximately 280–330 m long and 40–100 m wide; medium confidence",
   "shape": "Bent, irregular strip; high confidence",
   "orientation": "Northwest–southeast; high confidence",
   "count": 1
  },
  {
   "type": "Compact vegetation patch; high confidence",
   "cell": "top_left, middle_left",
   "size_m": "Approximately 170 × 130 m; medium confidence",
   "shape": "Irregular and lobed; high confidence",
   "orientation": "No strong axis; medium confidence",
   "count": 1
  },
  {
   "type": "Small vegetation patches near a dark boundary; high confidence",
   "cell": "middle_left, middle_center",
   "size_m": "Approximately 50–120 m across each; medium confidence",
   "shape": "Irregular compact patches; high confidence",
   "orientation": "Patch grouping trends southwest–northeast; medium confidence",
   "count": 2
  },
  {
   "type": "Compact vegetation patch; high confidence",
   "cell": "middle_center, bottom_center",
   "size_m": "Approximately 160 × 150 m; medium confidence",
   "shape": "Rounded to irregularly polygonal; high confidence",
   "orientation": "No strong axis; medium confidence",
   "count": 1
  },
  {
   "type": "Prominent elongated vegetation patches; high confidence",
   "cell": "top_right",
   "size_m": "Approximately 100–180 m long and 20–70 m wide; low confidence",
   "shape": "Narrow, irregular streaks; high confidence",
   "orientation": "Roughly north–south to northeast–southwest; medium confidence",
   "count": 3
  }
 ],
 "spectral": {
  "true_color": "High confidence: orange-brown and muted tan surfaces dominate illuminated terrain, with discrete green patches and extensive near-black zones. Surface brightness changes sharply across several diagonal boundaries.",
  "false_color": "High confidence: the green patches become bright pink-red, particularly in top_center, middle_left and middle_center. The bottom_center patch is a more muted red. Much of the brown background becomes olive, tan or gray-green, while the darkest zones remain dark.",
  "implications": [
   {
    "claim": "The green-to-red correspondence identifies localized photosynthetically active vegetation.",
    "confidence": "high"
   },
   {
    "claim": "Red intensity varies with vegetation amount or condition, but illumination, canopy structure and rendering also contribute; relative physiological vigor is not securely measurable.",
    "confidence": "medium"
   },
   {
    "claim": "The widespread brown and weakly red background is consistent with exposed mineral surfaces, dry vegetation or a mixture of both.",
    "confidence": "medium"
   },
   {
    "claim": "Vegetation concentrated in strips and pockets may reflect locally greater moisture availability or deeper soil.",
    "confidence": "low"
   },
   {
    "claim": "Terrain shadow is the leading explanation for the broad dark zones, given their banded configuration and adjoining rough slopes.",
    "confidence": "medium"
   },
   {
    "claim": "Open water is not confidently identified, so water turbidity cannot be assessed.",
    "confidence": "high"
   },
   {
    "claim": "No coherent built-surface pattern is resolved; isolated structures could remain undetected.",
    "confidence": "medium"
   }
  ],
  "not_revealed": "High confidence: these displayed composites cannot provide calibrated reflectance, quantitative vegetation indices, soil moisture, vegetation species, crop identity, canopy height, water depth or temperature. They also cannot establish temporal change, management history, exact season or climate."
 },
 "interpretation": [
  {
   "claim": "The patch most likely represents rugged, predominantly exposed or sparsely vegetated terrain.",
   "confidence": "medium",
   "evidence": "Levels 1–3; all cells: repeated rough brown surfaces, apparent slope bands and extensive adjoining dark zones."
  },
  {
   "claim": "Vegetation is spatially localized rather than forming a continuous dense canopy.",
   "confidence": "high",
   "evidence": "Levels 2 and 4; top_center, middle_left, middle_center and bottom_center: discrete green patches correspond to red false-color responses within a weakly red background."
  },
  {
   "claim": "Some vegetation may occupy sheltered depressions or soil-retaining terrain pockets.",
   "confidence": "low",
   "evidence": "Levels 2–4; top_center, top_right and bottom_center: irregular strips and patches appear associated with terrain recesses, but elevation and soil data are absent."
  },
  {
   "claim": "There is no clear visual evidence of organized agriculture, forestry operations or settlement.",
   "confidence": "medium",
   "evidence": "Levels 2–3; all cells: no convincing rectilinear parcels, planting rows, road network, building clusters or regular clearings."
  },
  {
   "claim": "Crop types, rotation, irrigation and harvest cannot be inferred.",
   "confidence": "high",
   "evidence": "Levels 2–4; all cells: agricultural features are not securely identified, and only one paired observation is supplied."
  },
  {
   "claim": "The terrain pattern is compatible with dissection by erosion, but active erosion is not demonstrated.",
   "confidence": "low",
   "evidence": "Levels 2–3; top_right and bottom_center: apparent ribs and recesses suggest relief, without identifiable fresh sediment movement."
  },
  {
   "claim": "Flooding, recent clearing and vegetation degradation are not established.",
   "confidence": "high",
   "evidence": "Levels 1–4; all cells: dark areas are ambiguous and no temporal comparison or diagnostic disturbance pattern is available."
  },
  {
   "claim": "A dry, rocky upland setting with sparse herbaceous or shrubby cover is plausible, but a specific biome or climate is unresolved.",
   "confidence": "low",
   "evidence": "Levels 1 and 4; illuminated portions throughout: brown mineral-looking surfaces and patchy vegetation could also reflect seasonality, substrate or elevation."
  }
 ],
 "uncertainties": [
  "High confidence: cover percentages are visual estimates rather than a measured classification; shadow obscures underlying land cover.",
  "High confidence: display upsampling does not add spatial information beyond the native 10 m pixels.",
  "Medium confidence: the darkest areas are mainly terrain shadow, but water or very dark surfaces cannot be excluded everywhere.",
  "High confidence: trees, shrubs, grasses and crops cannot be reliably separated within these small patches.",
  "High confidence: narrow tracks, channels, small buildings and individual crowns may be unresolved or mixed within pixels.",
  "High confidence: apparent feature dimensions are approximate, and counts depend on whether adjacent patches are grouped.",
  "High confidence: contrast stretching, resampling and mixed pixels can alter apparent color intensity, sharpness and patch boundaries.",
  "High confidence: brightness differences alone do not establish moisture, vegetation health, slope angle or elevation.",
  "Medium confidence: no obvious clouds are visible, but a contribution from cloud shadow cannot be completely excluded."
 ],
 "summaries": {
  "w25": "Rugged, mostly brown terrain contains scattered active vegetation and broad dark bands, probably terrain shadows. Landscape interpretation has medium confidence; vegetation identification has high confidence.",
  "w50": "The patch likely shows rugged, sparsely vegetated terrain with extensive terrain shadow; medium confidence. Approximate visible fractions are 56% brown surfaces, 12% green vegetation and 32% obscured dark terrain. Green patches become red in false color; high confidence. Water, agriculture and settlement are not securely identified.",
  "w100": "The patch likely depicts rugged terrain dominated by exposed or sparsely vegetated brown surfaces; medium confidence. Estimated visible fractions are 56% brown terrain, 12% green vegetation and 32% dark, obscured terrain. A broad southwest–northeast dark corridor and subsidiary bands probably reflect terrain shadow; medium confidence. Green patches in the upper center, middle left and lower center become red in false color, confirming localized active vegetation; high confidence. No convincing parcel grid, road network or settlement is resolved. Season, biome, moisture status and active land management remain uncertain.",
  "w200": "The 1.2 km patch likely contains rugged, sparsely vegetated terrain with extensive terrain shadow; medium confidence. Approximate visible fractions are 56% brown exposed or sparsely vegetated surfaces, 12% distinct green vegetation and 32% dark, obscured terrain. These are appearance estimates, and shadow conceals underlying cover.\n\nA broad dark corridor crosses southwest–northeast through the middle, with additional diagonal bands and a southward lobe on the right. Their association with rough brown surfaces favors terrain-controlled illumination; medium confidence. Vegetation occurs as a bent strip in the upper center, a compact patch at middle left, smaller central patches and a larger patch near the lower center. Their green true-color appearance and red false-color response identify active vegetation with high confidence.\n\nIrregular vegetation pockets may indicate locally favorable moisture or soil conditions; low confidence. No convincing agricultural parcel pattern, road network or settlement is resolved; medium confidence. Water is not confidently identified, preventing turbidity assessment. Exact season, vegetation type, climate and active processes such as irrigation, harvesting, flooding or clearing cannot be established from these renderings.",
  "w400": "The patch most likely represents rugged terrain dominated by exposed rock or soil and sparse vegetation; medium confidence. Estimated appearance fractions are 56% brown exposed or sparsely vegetated surfaces, 12% distinct green vegetation and 32% dark, obscured terrain. These approximate fractions describe what is visible; shadow is not an underlying land-cover class.\n\nThe strongest spatial feature is a broad southwest–northeast dark corridor crossing the central area and reaching the western and eastern margins. Additional northwest–southeast dark bands occur on the left, while an elongated dark lobe occupies much of the lower right. Rough adjoining surfaces, irregular boundaries and repeated banding favor terrain shadow as the main explanation; medium confidence. Water cannot be excluded from every dark pixel, but no water body is securely identified.\n\nVegetation is concentrated in a bent upper-center strip approximately 280–330 m long, a roughly 170 × 130 m patch at middle left, smaller patches along the central dark boundary, and an approximately 160 × 150 m patch near the lower center. Narrow vegetation streaks also occur in the upper right. Their green true-color appearance and pink-red false-color response provide high-confidence evidence of localized active vegetation. The surrounding weakly red, brown terrain is consistent with mineral surfaces mixed with dry or sparse vegetation; medium confidence.\n\nVegetation pockets could reflect sheltered positions, deeper soils or greater moisture availability, but that explanation has low confidence. Red intensity alone cannot isolate plant vigor from canopy cover, illumination and display effects. A dry, rocky upland setting is plausible at low confidence; a particular biome, climate or season is unresolved.\n\nNo convincing parcel grid, planting rows, road network, building cluster or forestry clearing is apparent; medium confidence. Crop identity, rotation, irrigation, harvesting, flooding and recent clearing cannot be established. Apparent terrain dissection may reflect erosion, but active sediment movement is not demonstrated. At 10 m resolution, narrow channels, tracks, small structures and vegetation types remain uncertain. Display upsampling adds no spatial detail, and the two composites cannot supply quantitative moisture, elevation, canopy height or temporal change."
 }
}
```

#### Patch 3f747420-913a-44d0-8cc8-49088e9b77c8

```json
{
 "global": {
  "cover": [
   {
    "class": "Green agricultural fields and open herbaceous vegetation; medium confidence",
    "percent": 43
   },
   {
    "class": "Bare, sparsely vegetated or senescent agricultural fields; medium confidence",
    "percent": 21
   },
   {
    "class": "Settlement mosaic of buildings, gardens and small plots; medium confidence",
    "percent": 28
   },
   {
    "class": "Woody vegetation, hedges and tree belts outside the settlement mosaic; medium confidence",
    "percent": 5
   },
   {
    "class": "Visible road corridors outside the settlement mosaic; medium confidence",
    "percent": 3
   }
  ],
  "landscape": "High confidence: an agricultural landscape adjoining a dispersed to moderately compact settlement, with large fields concentrated toward the left and buildings concentrated toward the right and lower center.",
  "season_cues": "Medium confidence: strong vegetation signals coexist with pale, weakly vegetated fields, suggesting contrasting field management or phenological stages. Low confidence: this could represent a growing season with some fallow, harvested or recently prepared parcels; the calendar season is indeterminate.",
  "heterogeneity": "High confidence: substantial heterogeneity between broad, comparatively uniform fields and fine-grained settlement, gardens, roads and vegetated boundaries. Medium confidence: the field mosaic also contains pronounced differences in vegetation cover."
 },
 "grid": [
  {
   "cell": "top_left",
   "cover": "High confidence: predominantly agricultural parcels, including a pale field and darker vegetated margins.",
   "texture": "High confidence: broad smooth interiors divided by narrow, irregular dark lines.",
   "features": "Medium confidence: a large pale parcel extends into top_center; a dark wedge along its western side likely contains woody or dense boundary vegetation."
  },
  {
   "cell": "top_center",
   "cover": "High confidence: green fields in the upper portion, pale agricultural land toward the lower left and a settlement fringe toward the lower right.",
   "texture": "High confidence: smooth field surfaces transition into small, contrasting settlement elements.",
   "features": "High confidence: strong red vegetation occupies the northern edge. Medium confidence: an oblique road separates fields from small developed plots; a conspicuous bright roof lies near the lower edge."
  },
  {
   "cell": "top_right",
   "cover": "High confidence: settlement and gardens beneath a strip of green agricultural land at the top.",
   "texture": "High confidence: fine, irregular mottling with pale or cyan spots and dark linear gaps.",
   "features": "High confidence: a roughly north–south road corridor traverses the cell. Medium confidence: branching lanes connect clusters of buildings."
  },
  {
   "cell": "middle_left",
   "cover": "High confidence: a large green field dominates, with settlement or small plots along its southwestern margin.",
   "texture": "High confidence: mostly smooth vegetation with a darker elongated band and local tonal variation.",
   "features": "Medium confidence: field boundaries trend approximately northwest–southeast; the darker band may reflect vegetation or management differences."
  },
  {
   "cell": "middle_center",
   "cover": "High confidence: the eastern part of the large green field meets a settlement fringe and small vegetated plots.",
   "texture": "High confidence: smooth field texture on the left becomes fragmented and mottled on the right.",
   "features": "Medium confidence: an irregular road corridor follows the field edge, with several bright roof-like objects and narrow garden strips nearby."
  },
  {
   "cell": "middle_right",
   "cover": "High confidence: settlement mixed with gardens and woody vegetation.",
   "texture": "High confidence: dense, irregular alternation of bright surfaces, dark vegetation and red vegetation patches in false color.",
   "features": "Medium confidence: branching roads and elongated building clusters occupy most of the cell; several larger bright roofs are distinguishable."
  },
  {
   "cell": "bottom_left",
   "cover": "High confidence: pale agricultural fields with settlement extending into the upper portion.",
   "texture": "High confidence: broad smooth parcels with faint parallel streaks, bordered by narrow dark lines.",
   "features": "Medium confidence: a diagonal road or lane separates small developed plots from agricultural parcels; faint field streaks may be cultivation traces."
  },
  {
   "cell": "bottom_center",
   "cover": "High confidence: settlement and small gardens, with pale fields toward the left and open vegetation along the bottom.",
   "texture": "High confidence: closely spaced linear and rectangular elements surrounded by smoother fields.",
   "features": "Medium confidence: short rows of buildings or plots follow local lanes. High confidence: a tiny multicolored bright anomaly occurs near the upper center of this cell."
  },
  {
   "cell": "bottom_right",
   "cover": "High confidence: mixed settlement, vegetated plots and an elongated open field.",
   "texture": "High confidence: mottled development surrounds a smoother, horizontally elongated parcel.",
   "features": "Medium confidence: a curving road lies near the right edge, while a dark vegetated boundary follows the lower edge of the open parcel."
  }
 ],
 "boundaries": "High confidence: agricultural parcels have sharp polygonal boundaries, while the settlement–field interface is irregular and penetrated by small plots. The main spatial gradient runs from larger fields in the west and north to finer settlement texture in the east and lower center. Medium confidence: dark linear boundaries represent a mixture of hedges, trees, tracks and possibly ditches. Their composition cannot be separated consistently.",
 "features": [
  {
   "type": "Large pale agricultural parcel; high confidence",
   "cell": "top_left, top_center",
   "size_m": "Approximately 450–550 by 250–350; medium confidence",
   "shape": "Irregular polygon; high confidence",
   "orientation": "Long axis approximately west–east; medium confidence",
   "count": 1
  },
  {
   "type": "Large green agricultural parcel or adjoining parcels with weak internal separation; high confidence",
   "cell": "middle_left, middle_center, top_left",
   "size_m": "Approximately 650–750 by 350–450; medium confidence",
   "shape": "Broad irregular quadrilateral; medium confidence",
   "orientation": "Long axis northwest–southeast; high confidence",
   "count": 1
  },
  {
   "type": "Northern green field group, partly clipped by image edges; high confidence",
   "cell": "top_center, top_right",
   "size_m": "Combined visible extent approximately 650–750 by 150–250; medium confidence",
   "shape": "Adjoining polygonal parcels; high confidence",
   "orientation": "Mixed, with approximately north–south internal divisions; medium confidence",
   "count": 1
  },
  {
   "type": "Southwestern pale field group; high confidence",
   "cell": "bottom_left, bottom_center",
   "size_m": "Combined visible extent approximately 550–650 by 300–400; medium confidence",
   "shape": "Adjoining irregular quadrilaterals, partly clipped; high confidence",
   "orientation": "Predominantly northeast–southwest; medium confidence",
   "count": 1
  },
  {
   "type": "Elongated open agricultural parcel; medium confidence",
   "cell": "bottom_center, bottom_right",
   "size_m": "Approximately 350–450 by 100–160; medium confidence",
   "shape": "Elongated irregular quadrilateral; high confidence",
   "orientation": "Approximately west–east; high confidence",
   "count": 1
  },
  {
   "type": "Connected settlement mosaic; high confidence",
   "cell": "top_right, middle_right, bottom_right, bottom_center, bottom_left",
   "size_m": "Approximately 700–1000 across its branching extent; medium confidence",
   "shape": "Irregular, branching clusters and strips; high confidence",
   "orientation": "Main eastern extent approximately north–south, with a southwestern branch; medium confidence",
   "count": 1
  },
  {
   "type": "Road and lane network; high confidence",
   "cell": "top_center, top_right, middle_center, middle_right, bottom_left, bottom_center, bottom_right",
   "size_m": "Individual visible segments approximately 150–1000 long; widths generally unresolved or mixed within 10–20 m pixels; medium confidence",
   "shape": "Connected straight and curving linear corridors; high confidence",
   "orientation": "Principal eastern corridor north–south, with oblique branches; high confidence",
   "count": 1
  },
  {
   "type": "Prominent bright roof or compact building group; medium confidence",
   "cell": "top_center, middle_center",
   "size_m": "Approximately 30–60 by 50–90, including mixed edge pixels; low confidence",
   "shape": "Compact elongated patch; medium confidence",
   "orientation": "Approximately north–south; medium confidence",
   "count": 1
  },
  {
   "type": "Dark woody boundary network; medium confidence",
   "cell": "top_left, middle_left, bottom_left, bottom_center, bottom_right",
   "size_m": "Segments approximately 100–500 long and 10–40 apparent width; low confidence",
   "shape": "Narrow connected or interrupted strips; high confidence",
   "orientation": "Follows multiple field edges; high confidence",
   "count": 1
  },
  {
   "type": "Compact multicolored bright anomaly; high confidence for its presence, low confidence for its cause",
   "cell": "bottom_center",
   "size_m": "Approximately 30–50 across its displayed footprint; low confidence",
   "shape": "Compact, pixelated patch; high confidence",
   "orientation": "No meaningful orientation; high confidence",
   "count": 1
  }
 ],
 "spectral": {
  "true_color": "High confidence: fields range from medium or dark green to muted tan and gray-brown. Settlement contains small white and gray bright surfaces mixed with dark green vegetation. Dark field margins contrast with comparatively smooth interiors. The compact anomaly near bottom_center has unusually saturated colors.",
  "false_color": "High confidence: northern fields are strongly red, the large central-western field is red to pink with a darker internal band, and gardens form fragmented red patches among cyan-gray built surfaces. The large pale northern parcel and southwestern pale fields are comparatively gray-green, indicating weaker relative near-infrared expression in this rendering.",
  "implications": [
   {
    "claim": "Bright red fields contain substantial photosynthetically active vegetation; relative differences may reflect canopy amount, structure or vigor.",
    "confidence": "high"
   },
   {
    "claim": "Differences in red intensity cannot be attributed uniquely to plant health because crop identity, cover fraction, soil background and display scaling also affect appearance.",
    "confidence": "high"
   },
   {
    "claim": "Pale true-color fields with muted false-color responses likely have less green vegetation and more exposed soil, dry vegetation or residue.",
    "confidence": "medium"
   },
   {
    "claim": "White or cyan compact objects within the settlement are consistent with reflective roofs or other artificial surfaces.",
    "confidence": "medium"
   },
   {
    "claim": "Dark strips may contain dense woody vegetation or shadows; darkness alone does not establish high moisture or standing water.",
    "confidence": "high"
   },
   {
    "claim": "No substantial open-water body is confidently identifiable, so water turbidity cannot be assessed.",
    "confidence": "high"
   },
   {
    "claim": "Soil and vegetation moisture differences cannot be reliably inferred from these rendered visible and near-infrared composites alone.",
    "confidence": "high"
   }
  ],
  "not_revealed": "High confidence: these renderings do not provide calibrated reflectance, quantitative vegetation indices, soil moisture, crop species, yield, irrigation status, water chemistry or temporal change. They also do not establish building use, vegetation species, exact phenological dates or the cause of individual spectral anomalies."
 },
 "interpretation": [
  {
   "claim": "Agriculture and settlement are the principal land uses.",
   "confidence": "high",
   "evidence": "Level 1, whole patch; Level 2, top_left, middle_left and bottom_left show bounded fields, while top_right, middle_right and bottom_center show building and road mosaics."
  },
  {
   "claim": "Agricultural management varies between parcels, producing contrasting vegetation cover or growth stages.",
   "confidence": "medium",
   "evidence": "Level 4, top_center and middle_left have strong red vegetation, whereas top_left and bottom_left contain pale parcels with muted false-color responses."
  },
  {
   "claim": "The large green fields could support grassland, forage or other crops; specific crop types cannot be distinguished.",
   "confidence": "low",
   "evidence": "Levels 2 and 4, middle_left and top_center show relatively uniform vegetated interiors without diagnostic crop structure."
  },
  {
   "claim": "Some pale fields may be harvested, fallow, recently cultivated or senescent, but no individual process is established.",
   "confidence": "low",
   "evidence": "Levels 2 and 4, top_left and bottom_left show tan or gray-green parcels with limited red response."
  },
  {
   "claim": "The settlement likely includes residential plots and gardens, with possible agricultural buildings along its edge.",
   "confidence": "medium",
   "evidence": "Levels 2 and 3, middle_right and bottom_center show small repeated bright objects and narrow vegetated plots; top_center contains a larger bright roof-like feature."
  },
  {
   "claim": "Woody field boundaries and gardens provide a fragmented vegetation network, although habitat quality cannot be assessed.",
   "confidence": "medium",
   "evidence": "Levels 2–4, top_left, bottom_center and bottom_right show dark linear boundaries and discontinuous red vegetation among developed surfaces."
  },
  {
   "claim": "There is no clear large managed forest stand within the patch.",
   "confidence": "medium",
   "evidence": "Levels 1 and 2, whole patch: broad smooth fields and settlement dominate, with woody cover mainly appearing as narrow strips or small patches."
  },
  {
   "claim": "No convincing visible evidence establishes active irrigation, flooding, major erosion or recent clearing.",
   "confidence": "medium",
   "evidence": "Levels 2 and 3, whole patch: no diagnostic inundation pattern, irrigation geometry, major erosion scar or distinct clearing front is apparent."
  },
  {
   "claim": "Crop rotation and changes in management cannot be demonstrated from this single observation.",
   "confidence": "high",
   "evidence": "Level 4, whole patch: spectral differences are spatial contrasts without a temporal sequence."
  },
  {
   "claim": "A temperate agricultural setting is plausible, but biome and climate classification remain weakly constrained.",
   "confidence": "low",
   "evidence": "Levels 1, 2 and 4, whole patch: green fields, woody boundaries and settlement gardens are compatible with temperate conditions but are not geographically diagnostic."
  }
 ],
 "uncertainties": [
  "High confidence: cover percentages are visual estimates, not measured classifications; settlement includes unresolved gardens, lanes and small vegetated areas.",
  "High confidence: upsampling adds no spatial information. Small roofs, individual trees, narrow roads, ditches and field margins commonly occupy mixed 10 m pixels.",
  "High confidence: feature counts refer to major visible groups where stated; individual building, field and tree counts are not reliably recoverable.",
  "Medium confidence: dimensions of large features are approximate to tens of meters, while small bright objects may have substantially inflated apparent footprints.",
  "High confidence: bare soil, crop residue, dry grass and sparse crops can have overlapping appearances in these composites.",
  "High confidence: dark vegetation, narrow shadows and possible wet ground cannot always be separated.",
  "High confidence: no obvious cloud obscuration is visible, but small shadows cannot be assigned confidently to clouds, trees or buildings.",
  "Medium confidence: the tiny saturated anomaly in bottom_center could involve a reflective object, band misregistration, resampling or display effects; its physical identity is unresolved.",
  "High confidence: north-up orientation supports approximate directions, but irregular shapes make single feature axes approximate.",
  "High confidence: season, recent weather, terrain slope, crop rotation, moisture stress and climate cannot be established from these images alone."
 ],
 "summaries": {
  "w25": "High confidence: agricultural fields border a settlement. Medium confidence: contrasting vegetation signals reflect differing field conditions. Crop identity, season and active management remain unresolved.",
  "w50": "High confidence: large agricultural fields dominate the west and north, while settlement concentrates eastward and across the lower center. Bright false-color reds indicate substantial green vegetation. Medium confidence: pale fields have sparse cover, dry vegetation or exposed soil. Crop identity, season, moisture and recent management remain unresolved.",
  "w100": "High confidence: this 1.2 km patch contains an agricultural and settlement mosaic, with broad fields westward and northward and finer building, garden and road texture eastward. Strong false-color reds identify substantial green vegetation, contrasting with pale parcels in the northwest and southwest. Medium confidence: these differences reflect vegetation cover or management stages, including possible bare soil or residue. Woody boundaries divide some fields. No substantial open water is confidently identifiable. Individual buildings and narrow boundaries are poorly resolved at 10 m. Crop types, season, moisture status and active agricultural processes remain uncertain.",
  "w200": "High confidence: the patch shows agriculture adjoining a settlement, with large fields concentrated in the west and north and buildings, gardens and roads concentrated in the east and lower center. Medium-confidence visual estimates assign 43% to green fields and open vegetation, 21% to pale or sparsely vegetated agricultural land, 28% to the settlement mosaic, 5% to woody boundaries and 3% to separately visible roads.\n\nHigh confidence: northern fields and a large central-western field show strong red responses in the near-infrared composite. Pale northwestern and southwestern parcels have weaker responses. Medium confidence: this contrast reflects differing green cover, canopy condition or management stage; exposed soil, residue and senescent vegetation remain competing explanations.\n\nMedium confidence: dark field margins form a fragmented woody network. Roads connect irregular settlement clusters, and several bright roof-like objects are visible. No substantial open water or convincing large-scale flooding is identified. A small saturated anomaly near bottom_center has an uncertain cause.\n\nHigh confidence: 10 m mixed pixels limit building counts and boundary measurements. Crop identity, rotation, irrigation, moisture stress and season cannot be established. A temperate agricultural setting is plausible with low confidence.",
  "w400": "High confidence: this north-up, 1.2 km square patch depicts an agricultural landscape adjoining a settlement. Large, smooth fields occupy much of the west and north; buildings, gardens, small plots and branching roads produce finer texture in the east and lower center. Medium-confidence visual cover estimates are 43% green agricultural and open herbaceous cover, 21% bare or sparsely vegetated agricultural land, 28% settlement mosaic including gardens, 5% woody vegetation outside that mosaic and 3% separately visible road corridors.\n\nHigh confidence: a large pale polygon spans the northwestern cells, while a broad green parcel extends across middle_left into middle_center. Northern fields also retain strong vegetation signals. Pale fields occupy much of bottom_left, and an elongated open parcel lies toward bottom_right. Medium confidence: dark boundaries include hedges or tree belts, although tracks, ditches and shadows may contribute. A principal road runs approximately north–south through the eastern settlement, with oblique branches connecting other clusters.\n\nHigh confidence: the false-color rendering distinguishes strongly red vegetated fields from muted gray-green pale parcels. Medium confidence: these contrasts indicate differences in canopy cover or growth stage and may reflect cultivation, fallow conditions, harvest residue or senescence. They do not uniquely diagnose plant health. Reflective white or cyan patches are consistent with roofs and other built surfaces with medium confidence. A compact saturated anomaly in bottom_center is visible with high confidence, but its physical or processing-related origin has low confidence.\n\nMedium confidence: land use includes cultivated fields, possible grassland or forage, and likely residential gardens. Specific crop types remain low-confidence possibilities. Woody margins provide a fragmented vegetation network, but ecological quality is unresolved. No substantial open water is confidently identified, and no convincing visual signature establishes irrigation, major flooding, erosion or recent clearing. A temperate agricultural setting is plausible with low confidence; biome and climate are not diagnostic from this scene.\n\nHigh confidence: display upsampling does not improve the native 10 m resolution. Individual buildings, narrow lanes, trees and boundary strips often form mixed pixels. The cover fractions and feature dimensions are approximate. These renderings alone cannot establish calibrated vegetation indices, soil moisture, crop rotation, yield, exact season or recent changes."
 }
}
```

#### Patch 2c3e8d2b-ff94-421d-b1fe-773fe8f2164d

```json
{
 "global": {
  "cover": [
   {
    "class": "Vegetated agricultural fields; medium confidence, approximate visual estimate",
    "percent": 45
   },
   {
    "class": "Bare, harvested, sparsely vegetated or residue-covered agricultural ground; medium confidence",
    "percent": 40
   },
   {
    "class": "Trees, hedgerows and woody patches; medium confidence",
    "percent": 13
   },
   {
    "class": "Tracks and possible built surfaces; low confidence",
    "percent": 2
   }
  ],
  "landscape": "High confidence: predominantly agricultural mosaic of geometric fields with woody margins and small patches.",
  "season_cues": "Medium confidence: actively vegetated fields coexist with pale fields carrying little green vegetation. Low confidence: this could represent a growing-season or harvest-transition mosaic; the month and season are not identifiable.",
  "heterogeneity": "High confidence: strong differences between parcels, relatively uniform interiors in several fields, and more irregular, textured woody cover toward the upper left."
 },
 "grid": [
  {
   "cell": "top_left",
   "cover": "Medium confidence: woody vegetation mixed with small agricultural parcels.",
   "texture": "High confidence: irregular coarse patches surrounding smoother green and pale areas.",
   "features": "Medium confidence: branching woody margins, small clearings or fields, and partial parcels cut by the image edges."
  },
  {
   "cell": "top_center",
   "cover": "High confidence: alternating vegetated and pale agricultural strips, with woody vegetation toward the left.",
   "texture": "High confidence: smooth elongated parcels contrast with irregular vegetation.",
   "features": "High confidence: approximately north–south field strips; medium confidence: broad pale agricultural ground extending into the center."
  },
  {
   "cell": "top_right",
   "cover": "High confidence: agricultural land containing a dark green northern parcel and extensive pale ground.",
   "texture": "High confidence: mostly smooth, with faint linear variation.",
   "features": "High confidence: straight parcel edges and a pale oblique boundary or access corridor; low confidence: its exact function."
  },
  {
   "cell": "middle_left",
   "cover": "High confidence: a large, strongly vegetated agricultural field.",
   "texture": "High confidence: comparatively uniform green surface with subtle mottling.",
   "features": "High confidence: the western part of the large rectangular field; medium confidence: narrow contrasting strips along its edges."
  },
  {
   "cell": "middle_center",
   "cover": "High confidence: a mixture of green crop cover, dark vegetated strips and pale agricultural ground.",
   "texture": "High confidence: smooth interiors separated by sharp linear boundaries.",
   "features": "High confidence: eastern edge of the large green field, a small darker inset strip, a broad pale north–south parcel, and part of the narrow arm of the eastern L-shaped field."
  },
  {
   "cell": "middle_right",
   "cover": "High confidence: pale agricultural ground surrounding a dark green vegetated parcel.",
   "texture": "High confidence: mostly smooth with abrupt tonal transitions.",
   "features": "High confidence: broad portion of an L-shaped field; medium confidence: narrow diagonal vegetated slivers near the right edge."
  },
  {
   "cell": "bottom_left",
   "cover": "Medium confidence: mostly vegetated agricultural land, with a small pale parcel and possible settlement at the corner.",
   "texture": "High confidence: smooth dark fields with local fine mottling and irregular corner features.",
   "features": "Medium confidence: a narrow curving access route and small pale rectangular parcel; low confidence: tiny roof-like objects at the lower-left edge."
  },
  {
   "cell": "bottom_center",
   "cover": "High confidence: dark vegetated fields beside a paler, weakly vegetated parcel.",
   "texture": "High confidence: generally smooth, with faint curved and linear markings.",
   "features": "High confidence: a pale elongated parcel reaching the southern edge; medium confidence: curving field track or management boundary to its west."
  },
  {
   "cell": "bottom_right",
   "cover": "High confidence: predominantly pale agricultural ground.",
   "texture": "High confidence: faint parallel diagonal bands and subtle tonal differences.",
   "features": "High confidence: oblique field divisions; medium confidence: a narrow access strip and small vegetated wedge near the southern edge."
  }
 ],
 "boundaries": "High confidence: most agricultural boundaries are abrupt, straight or angular, with prominent north–south divisions through the center and near east–west edges around the large western field. Medium confidence: woody margins in the upper left are irregular, while lower-right divisions trend northeast–southwest. High confidence: the dominant spectral contrast is between green or red vegetated parcels and pale tan or cyan parcels, rather than a continuous scene-wide gradient.",
 "features": [
  {
   "type": "Large green agricultural field; high confidence",
   "cell": "middle_left, middle_center",
   "size_m": "Approximately 520–560 east–west by 380–420 north–south; medium confidence",
   "shape": "Broad rectangle with slightly irregular margins; high confidence",
   "orientation": "Edges approximately north–south and east–west; high confidence",
   "count": 1
  },
  {
   "type": "Darker inset strip within or beside the large green field; medium confidence",
   "cell": "middle_center",
   "size_m": "Approximately 60–90 by 150–180; low confidence",
   "shape": "Narrow rectangle; high confidence",
   "orientation": "North–south; high confidence",
   "count": 1
  },
  {
   "type": "L-shaped vegetated agricultural parcel; high confidence",
   "cell": "middle_center, middle_right, bottom_right",
   "size_m": "Overall envelope approximately 250 east–west by 520 north–south; medium confidence",
   "shape": "L-shaped, with a narrow northern arm and wider southern block; high confidence",
   "orientation": "Long axis north–south; high confidence",
   "count": 1
  },
  {
   "type": "Broad pale agricultural strip; high confidence",
   "cell": "middle_center",
   "size_m": "Approximately 110–150 wide and 500 long; medium confidence",
   "shape": "Elongated, slightly tapering parcel; high confidence",
   "orientation": "Approximately north–south; high confidence",
   "count": 1
  },
  {
   "type": "Prominent northern vegetated strips; medium confidence in grouping and count",
   "cell": "top_center, top_right",
   "size_m": "Approximately 80–250 wide and at least 180–250 long within the patch; medium confidence",
   "shape": "Elongated quadrilateral parcels clipped by the northern image edge; high confidence",
   "orientation": "Approximately north–south; high confidence",
   "count": 2
  },
  {
   "type": "Connected woody patch and margin complex; medium confidence",
   "cell": "top_left, top_center",
   "size_m": "Overall visible envelope approximately 600 east–west by 400 north–south, containing agricultural openings; low confidence",
   "shape": "Irregular and branching; high confidence",
   "orientation": "Mixed, with a substantial east–west component; medium confidence",
   "count": 1
  },
  {
   "type": "Southern dark vegetated field complex; high confidence in cover, low confidence in internal parcel count",
   "cell": "bottom_left, bottom_center",
   "size_m": "Overall visible envelope approximately 550 by 400; medium confidence",
   "shape": "Irregular block with straight outer boundaries and curved internal markings; medium confidence",
   "orientation": "Mixed; medium confidence",
   "count": 1
  },
  {
   "type": "Pale southern agricultural parcel with weak vegetation signal; medium confidence",
   "cell": "bottom_center",
   "size_m": "Approximately 200–240 wide and at least 300 long; medium confidence",
   "shape": "Elongated quadrilateral clipped by the southern edge; high confidence",
   "orientation": "Approximately north–south; high confidence",
   "count": 1
  },
  {
   "type": "Small pale agricultural parcel; medium confidence",
   "cell": "bottom_left",
   "size_m": "Approximately 110–140 by 180–210; medium confidence",
   "shape": "Approximately rectangular; high confidence",
   "orientation": "North–south; high confidence",
   "count": 1
  },
  {
   "type": "Curving field track or management boundary; medium confidence",
   "cell": "bottom_left, bottom_center",
   "size_m": "Approximately 350–500 visible length; width unresolved or about one native pixel; low confidence",
   "shape": "Curving linear feature; high confidence",
   "orientation": "Turns from approximately north–south toward northeast–southwest; medium confidence",
   "count": 1
  },
  {
   "type": "Diagonal agricultural divisions or access strips; medium confidence in type and count",
   "cell": "bottom_right",
   "size_m": "Approximately 250–400 visible length each; widths uncertain at 10 m; low confidence",
   "shape": "Approximately parallel linear features; high confidence",
   "orientation": "Northeast–southwest; high confidence",
   "count": 2
  },
  {
   "type": "Possible small built cluster; low confidence",
   "cell": "bottom_left",
   "size_m": "Visible cluster approximately 100–150 across; low confidence",
   "shape": "Irregular group of small bright and gray elements; low confidence",
   "orientation": "No reliable dominant orientation; low confidence",
   "count": 1
  }
 ],
 "spectral": {
  "true_color": "High confidence: the large western field is medium to bright green; several northern, eastern and southern fields are much darker green. Extensive intervening ground is pale yellow or tan. Upper-left vegetation has irregular dark green texture. The southern central parcel is pale greenish tan.",
  "false_color": "High confidence: the large western field is bright pink-red, while the L-shaped eastern field and southern dark fields are strongly red. Upper-left woody cover varies from vivid red to dark maroon. Pale agricultural ground is mostly cyan or muted gray-green. The southern central parcel has a weak, muted pink-gray response.",
  "implications": [
   {
    "claim": "Strong red responses identify substantial live vegetation in the western, eastern L-shaped and southern fields.",
    "confidence": "high"
   },
   {
    "claim": "The bright western field likely has relatively continuous green cover, but display color alone does not establish superior plant health or productivity.",
    "confidence": "medium"
   },
   {
    "claim": "Pale tan areas that become cyan probably contain little green vegetation and may be bare soil, stubble, dry vegetation or mixtures of these.",
    "confidence": "medium"
   },
   {
    "claim": "Several dark true-color fields remain strongly red in false color, supporting vegetation rather than open water as their dominant cover.",
    "confidence": "high"
   },
   {
    "claim": "The southern central parcel likely has less green vegetation cover than the strongly red neighboring parcels.",
    "confidence": "medium"
   },
   {
    "claim": "Soil or canopy moisture differences could contribute to tonal variation, but neither moisture nor irrigation can be isolated from these renderings.",
    "confidence": "low"
   },
   {
    "claim": "No distinct open-water body is apparent; water turbidity cannot be assessed.",
    "confidence": "medium"
   },
   {
    "claim": "Small pale or gray features in the lower-left corner may include built surfaces, although their spectra overlap soil and mixed pixels.",
    "confidence": "low"
   }
  ],
  "not_revealed": "High confidence: these displayed composites do not provide calibrated reflectance, quantitative vegetation indices, crop identity, yield, soil moisture, canopy temperature, water chemistry, management history or temporal change. Red intensity also depends on canopy structure, soil background and rendering stretch, so it is not a direct measure of vegetation health."
 },
 "interpretation": [
  {
   "claim": "Agriculture is the dominant land use.",
   "confidence": "high",
   "evidence": "Levels 1–3; all grid cells: geometric parcels, smooth interiors and sharp differences in vegetation cover."
  },
  {
   "claim": "Different crop covers or management stages occur in neighboring parcels.",
   "confidence": "medium",
   "evidence": "Levels 2 and 4; middle_left, middle_center, middle_right and bottom_center: strongly red vegetation directly adjoins cyan and weakly pink fields."
  },
  {
   "claim": "Some pale parcels may have been harvested, left fallow or prepared for planting.",
   "confidence": "medium",
   "evidence": "Levels 2–4; top_right, middle_center, middle_right and bottom_right: pale agricultural surfaces with weak vegetation response and field-shaped boundaries."
  },
  {
   "claim": "Specific crop types, pasture versus arable use, and crop rotation cannot be established.",
   "confidence": "high",
   "evidence": "Levels 2 and 4; all agricultural cells: parcel geometry and single-date spectral contrasts are insufficiently diagnostic."
  },
  {
   "claim": "Woody patches and margins add structural habitat diversity within the agricultural mosaic.",
   "confidence": "medium",
   "evidence": "Levels 2–4; top_left and top_center: irregular textured vegetation contrasts with smooth managed fields."
  },
  {
   "claim": "Commercial forestry is not demonstrated.",
   "confidence": "medium",
   "evidence": "Levels 2–3; top_left and top_center: woody cover appears irregular, without clearly resolved plantation rows or forestry compartments."
  },
  {
   "claim": "A small settlement or farmstead may lie partly outside the lower-left image edge.",
   "confidence": "low",
   "evidence": "Levels 2–3; bottom_left: tiny bright elements occur beside a narrow curving linear feature."
  },
  {
   "claim": "Within-field linear markings may record agricultural operations, but individual machinery passes are unresolved.",
   "confidence": "medium",
   "evidence": "Levels 2–3; bottom_center and bottom_right: faint curved and parallel markings occur inside or alongside parcels."
  },
  {
   "claim": "Active irrigation, ongoing harvest, erosion, flooding and recent clearing cannot be confirmed.",
   "confidence": "high",
   "evidence": "Levels 2–4; all cells: no diagnostic equipment, inundation pattern, sediment plume, erosion network or temporal evidence is resolved."
  },
  {
   "claim": "A temperate agricultural setting is plausible, but the biome and climate remain weakly constrained.",
   "confidence": "low",
   "evidence": "Levels 1–2; scene-wide: green fields, woody margins and pale agricultural parcels are compatible with temperate farmland but occur in other climates."
  }
 ],
 "uncertainties": [
  "High confidence: area percentages are visual estimates, not a classified pixel inventory; mixed boundaries and ambiguous pale fields affect the totals.",
  "High confidence: upsampling does not add spatial information beyond the native 10 m pixels.",
  "High confidence: individual buildings, narrow roads, ditches, hedgerow widths, crop rows and machinery cannot be reliably resolved.",
  "Medium confidence: some apparent woody vegetation could be dense crops or scrub; some dark areas could include canopy shadows.",
  "High confidence: bare soil, crop residue, dry grass and sparse crops can share similar displayed colors.",
  "High confidence: a single observation cannot establish rotation, harvest timing, vegetation change or management activity.",
  "Medium confidence: no obvious cloud or broad cloud-shadow feature is visible, although small or subtle atmospheric effects cannot be excluded.",
  "High confidence: display stretching, resampling and mixed pixels can alter apparent contrast and boundary sharpness.",
  "High confidence: feature dimensions are approximate, edge-clipped features extend beyond the patch, and grouped feature counts do not constitute an exhaustive parcel inventory."
 ],
 "summaries": {
  "w25": "High confidence: geometric agricultural fields dominate. Medium confidence: contrasting vegetation stages, pale soil or residue, and woody margins form a heterogeneous managed landscape.",
  "w50": "High confidence: this agricultural mosaic contains a large green western field, a strongly vegetated eastern L-shaped parcel and extensive pale fields. Medium confidence: woody cover concentrates toward the upper left. Spectral contrasts suggest differing vegetation cover or management stages; crop identities, season and irrigation remain unresolved.",
  "w100": "High confidence: geometric agricultural parcels dominate the 1.2 km patch. A large western field is bright green and pink-red, while an eastern L-shaped field and southern fields are dark green but strongly red in false color. Medium confidence: pale tan-to-cyan fields carry little green vegetation, potentially reflecting soil, residue or harvested cover. Woody patches concentrate toward the upper left. Low confidence: a small built cluster occupies the lower-left edge. Crop species, rotation, irrigation, moisture and exact season cannot be determined reliably from these two renderings.",
  "w200": "High confidence: the patch is a heterogeneous agricultural mosaic with sharp parcel boundaries and mostly smooth field interiors. A large rectangular western field, approximately 540 by 400 m, contrasts with pale central ground and an eastern L-shaped vegetated field. Northern strips run approximately north–south; lower-right divisions are oblique. Medium confidence: irregular woody patches and margins concentrate toward the upper left, while a curving track crosses the southern fields. Low confidence: small lower-left bright elements belong to a farmstead or settlement.\n\nMedium-confidence visual estimates assign 45% to vegetated agriculture, 40% to bare, sparse or residue-covered agricultural ground and 13% to woody cover; the remaining 2% represents tracks and possible built surfaces with low confidence. High confidence: strong false-color red supports live vegetation even where true color is dark. Medium confidence: tan-to-cyan parcels contain little green vegetation and represent differing cover or management stages. Crop identity, rotation, irrigation and exact season remain unresolved. A temperate setting is plausible with low confidence. No distinct water body is apparent with medium confidence.",
  "w400": "High confidence: this 1.2 km square is predominantly agricultural, with geometric parcels, abrupt boundaries and contrasting vegetation cover. Medium-confidence visual estimates allocate 45% to vegetated fields, 40% to bare, sparse or residue-covered agricultural ground and 13% to woody vegetation. Tracks and possible built surfaces account for an estimated 2% with low confidence.\n\nHigh confidence: a large rectangular green field occupies much of the middle-left cell and extends into the center. Its dimensions are approximately 540 by 400 m with medium confidence. A narrow darker inset lies near its eastern edge. A broad pale north–south parcel separates western vegetation from an eastern L-shaped field. Northern fields form elongated strips, and lower-right field divisions trend northeast–southwest. Medium confidence: woody patches and branching margins concentrate in the upper-left sector. Southern fields contain a curving track or management boundary, a small pale western parcel and a larger pale central parcel. Low confidence: tiny bright features at the lower-left edge indicate a small built cluster.\n\nHigh confidence: the western field is bright pink-red in false color, and several dark green eastern and southern fields also become strongly red, indicating substantial live vegetation. Medium confidence: pale tan areas becoming cyan contain little green vegetation and could be bare soil, stubble, dry cover or sparse crops. The southern central parcel has a weaker vegetation response. Red brightness alone cannot establish crop health, productivity or moisture with confidence.\n\nMedium confidence: neighboring parcels represent different covers or agricultural management stages, potentially including harvested or fallow fields. Specific crops, rotation and commercial forestry are not established. High confidence: these images cannot confirm active irrigation, harvest, erosion, flooding or recent clearing. A temperate farming landscape is plausible with low confidence, but biome, climate and exact season remain unresolved.\n\nMedium confidence: no distinct open-water body or obvious cloud feature is apparent. High confidence: native 10 m resolution limits identification of individual buildings, ditches, narrow tracks and crop rows. Display stretching, mixed pixels and resampling affect visual interpretation; feature sizes and cover proportions remain approximate."
 }
}
```

#### Patch 5f9d5e6f-e221-43ef-b73f-7e4f895df271

```json
{
 "global": {
  "cover": [
   {
    "class": "Dense vegetation, probably predominantly woody canopy; medium confidence",
    "percent": 68
   },
   {
    "class": "Open or sparse vegetation mixed with exposed substrate; medium confidence",
    "percent": 25
   },
   {
    "class": "Predominantly exposed soil or rock; low confidence",
    "percent": 7
   }
  ],
  "landscape": "An irregular, vegetation-dominated landscape with apparent wooded slopes and patchy openings; medium confidence. No clearly organized agricultural or urban pattern is visible; high confidence.",
  "season_cues": "Extensive green vegetation and strong near-infrared response indicate substantial leafy vegetation at acquisition; high confidence. The date, season, evergreen versus deciduous character, and any dry-season influence cannot be determined reliably; low confidence.",
  "heterogeneity": "Moderate to high spatial heterogeneity, with contiguous dark-green vegetation interrupted by pale, irregular openings concentrated in the upper center and central-to-eastern areas; high confidence. Cover percentages are approximate visual estimates; medium confidence."
 },
 "grid": [
  {
   "cell": "top_left",
   "cover": "Mostly dense vegetation with scattered lighter, more open patches; medium confidence.",
   "texture": "Mottled and irregular, becoming darker toward the lower edge; high confidence.",
   "features": "Small pale patches and the western edge of the larger upper opening; medium confidence."
  },
  {
   "cell": "top_center",
   "cover": "A mixture of dense vegetation and a prominent pale opening containing vegetated islands; high confidence.",
   "texture": "Strong contrast between smooth pale areas and darker, coarse-looking vegetation; high confidence.",
   "features": "An irregular, lobed opening with curved internal edges occupies much of the northern portion; high confidence. A dark vegetated belt lies along its southern side; medium confidence."
  },
  {
   "cell": "top_right",
   "cover": "Predominantly dense vegetation, interspersed with elongated areas of thinner cover; medium confidence.",
   "texture": "Mottled, with approximately north–south tonal streaks; high confidence.",
   "features": "A darker strip approaches the northeastern edge, while diffuse pale patches extend toward the middle-right cell; high confidence."
  },
  {
   "cell": "middle_left",
   "cover": "Predominantly dense vegetation; medium confidence.",
   "texture": "Broad dark areas with subtle curved and diagonal tonal bands; high confidence.",
   "features": "A conspicuous dark belt runs approximately southwest–northeast across the upper portion; high confidence. It may reflect shaded terrain, denser canopy, or both; medium confidence."
  },
  {
   "cell": "middle_center",
   "cover": "Mixed dense vegetation and substantial open or sparsely vegetated ground; medium confidence.",
   "texture": "Patchy, with pale lobes separated by darker curving strips; high confidence.",
   "features": "An irregular central opening complex and a narrow pale, curved trace along part of its southern margin; high confidence. The trace could be a track or exposed terrain edge; low confidence."
  },
  {
   "cell": "middle_right",
   "cover": "Mixed dense vegetation and elongated sparse-cover patches; medium confidence.",
   "texture": "Strongly mottled, with branching pale areas and dark intervening strips; high confidence.",
   "features": "A roughly north–south chain of openings and a small bright patch near the lower-left part of the cell; high confidence."
  },
  {
   "cell": "bottom_left",
   "cover": "Mostly dense vegetation; medium confidence.",
   "texture": "Relatively continuous vegetation with diagonal dark streaks and isolated light flecks; high confidence.",
   "features": "Several southwest–northeast dark lineaments are visible, but no continuous road or water channel can be identified; medium confidence."
  },
  {
   "cell": "bottom_center",
   "cover": "Predominantly vegetation with small scattered openings; medium confidence.",
   "texture": "Mottled, with pale flecks and narrow dark bands; high confidence.",
   "features": "Small discontinuous openings, especially toward the eastern side, connect visually with the central patchwork; medium confidence."
  },
  {
   "cell": "bottom_right",
   "cover": "Mostly dense vegetation with sparse-cover areas near the bottom edge; medium confidence.",
   "texture": "Dark, irregular vegetation interrupted by small pale patches; high confidence.",
   "features": "A diagonal dark belt and a partially cropped pale opening along the southeastern boundary; high confidence."
  }
 ],
 "boundaries": "Most boundaries are curved, lobed, or diffuse rather than straight and rectangular; high confidence. The upper-center opening has comparatively sharp pale-to-vegetated edges, whereas central and eastern openings grade into mottled vegetation; high confidence. The lower-left portion is more continuously vegetated than the upper-center and middle-right portions; high confidence. Several southwest–northeast dark bands may follow terrain, but their origin is uncertain; medium confidence.",
 "features": [
  {
   "type": "Pale opening complex containing vegetated islands; high confidence in appearance, low confidence in origin",
   "cell": "top_center, extending into top_left",
   "size_m": "Approximately 300–350 wide by 350–420 long; medium confidence",
   "shape": "Irregular and lobed, with curved internal margins; high confidence",
   "orientation": "Broadly north–south; medium confidence",
   "count": 1
  },
  {
   "type": "Central sparse-cover and exposed-ground complex; medium confidence",
   "cell": "middle_center, extending toward top_center",
   "size_m": "Approximately 350–450 across; medium confidence",
   "shape": "Branching and patchy, with darker vegetated inclusions; high confidence",
   "orientation": "No strong single axis; medium confidence",
   "count": 1
  },
  {
   "type": "Eastern elongated opening complex; medium confidence",
   "cell": "middle_right, extending into top_right",
   "size_m": "Approximately 150–250 wide by 400–500 long; medium confidence",
   "shape": "Discontinuous, elongated and lobed; high confidence",
   "orientation": "Approximately north–south; medium confidence",
   "count": 1
  },
  {
   "type": "Dark vegetated or shaded belt; medium confidence",
   "cell": "middle_left, extending toward top_center",
   "size_m": "Approximately 500–650 long and 70–150 wide; low confidence",
   "shape": "Broad, irregular linear belt; high confidence",
   "orientation": "Southwest–northeast; high confidence",
   "count": 1
  },
  {
   "type": "Narrow pale trace, possibly a track or exposed terrain edge; low confidence",
   "cell": "middle_center",
   "size_m": "Approximately 150–250 long; width unresolved near the 10 m sampling scale; low confidence",
   "shape": "Curved and discontinuous; medium confidence",
   "orientation": "Approximately west–east; medium confidence",
   "count": 1
  },
  {
   "type": "Compact bright patch of unresolved material; high confidence in visibility, low confidence in classification",
   "cell": "middle_right",
   "size_m": "Approximately 20–40 across; low confidence",
   "shape": "Compact and irregular; medium confidence",
   "orientation": "No resolved orientation; low confidence",
   "count": 1
  },
  {
   "type": "Partially visible pale opening at image boundary; medium confidence",
   "cell": "bottom_right",
   "size_m": "Approximately 200–300 along the boundary and 50–100 visible inward; low confidence",
   "shape": "Irregular, truncated by the image edge; high confidence",
   "orientation": "Approximately southwest–northeast along its visible margin; low confidence",
   "count": 1
  }
 ],
 "spectral": {
  "true_color": "Dark to medium green dominates. Tan, olive-gray, and pale beige patches form irregular openings, with the clearest tan contrast in the upper center. Narrow dark bands cross parts of the scene, and a few compact pale flecks occur centrally and farther south. Confidence: high for these visual observations.",
  "false_color": "Red to crimson vegetation covers most of the patch. Pale true-color openings generally become gray-green or muted pink-gray, while much of the surrounding vegetation becomes strongly red. Some dark bands remain dark red rather than becoming clearly water-like. Confidence: high for the displayed color relationships.",
  "implications": [
   {
    "claim": "Widespread strong red response supports substantial leafy vegetation with relatively high near-infrared reflectance.",
    "confidence": "high"
   },
   {
    "claim": "Muted red and gray-green openings probably contain less leafy cover, more exposed substrate, or a mixture of both.",
    "confidence": "medium"
   },
   {
    "claim": "Differences in red intensity may reflect canopy density, vegetation type, illumination, or physiological condition; they do not uniquely measure vegetation health.",
    "confidence": "high"
   },
   {
    "claim": "The upper-center pale surfaces are plausible exposed soil or rock, but their composition cannot be distinguished.",
    "confidence": "medium"
   },
   {
    "claim": "No unambiguous open-water body is visible, so water turbidity cannot be assessed.",
    "confidence": "medium"
   },
   {
    "claim": "Dark vegetation and terrain shading cannot establish soil or canopy moisture differences from these renderings.",
    "confidence": "high"
   },
   {
    "claim": "The small bright patches could include rock, soil, or built material; their colors do not establish buildings or pavement.",
    "confidence": "high"
   }
  ],
  "not_revealed": "These display renderings do not provide quantitative reflectance, reliable vegetation indices, canopy height, species, crop identity, soil composition, water content, or change through time. They also cannot uniquely separate physiological stress from sparse cover and illumination effects. Confidence: high."
 },
 "interpretation": [
  {
   "claim": "The patch is probably predominantly wooded, with irregular openings rather than extensive cultivated fields.",
   "confidence": "medium",
   "evidence": "Levels 1–2: widespread dark-green, mottled cover in middle_left, bottom_left, bottom_center, and bottom_right; Level 4: extensive red vegetation response."
  },
  {
   "claim": "Terrain likely influences the vegetation and illumination pattern, suggesting an uneven or hilly landscape.",
   "confidence": "medium",
   "evidence": "Levels 2–3: curving tonal boundaries and diagonal dark belts in middle_left, middle_center, and bottom_right; no elevation measurements are available."
  },
  {
   "claim": "The openings could be natural rocky or thin-soil areas, managed clearings, or a mixture; their origin is unresolved.",
   "confidence": "low",
   "evidence": "Levels 2–4: irregular pale surfaces and reduced red response in top_center, middle_center, and middle_right, without diagnostic management geometry."
  },
  {
   "claim": "A minor access track may cross the central area.",
   "confidence": "low",
   "evidence": "Level 3, middle_center: a narrow curved pale trace, insufficiently continuous or resolved to confirm a road."
  },
  {
   "claim": "Substantial live leafy vegetation is present, but ecological condition, biodiversity, and degradation cannot be established.",
   "confidence": "high",
   "evidence": "Level 4, all grid cells: widespread green true-color cover and red false-color response; these are not direct measurements of ecosystem condition."
  },
  {
   "claim": "Crop types, crop rotation, plantation forestry, and recent harvest cannot be identified.",
   "confidence": "high",
   "evidence": "Levels 2–3, all grid cells: no clear crop blocks, planting rows, or diagnostic harvest pattern; only one acquisition is shown."
  },
  {
   "claim": "No substantial settlement is evident, although isolated small structures remain possible.",
   "confidence": "medium",
   "evidence": "Levels 2–3, all grid cells: no coherent roof pattern or street network; the compact bright patch in middle_right is unresolved."
  },
  {
   "claim": "Active irrigation, flooding, erosion, or recent clearing is not demonstrated by these images.",
   "confidence": "high",
   "evidence": "Levels 2–4, top_center, middle_center, and middle_right: pale openings lack a diagnostic process signature, and no temporal comparison is available."
  },
  {
   "claim": "A wooded terrestrial setting is plausible, but a specific biome or climate class cannot be assigned reliably.",
   "confidence": "low",
   "evidence": "Levels 1 and 4, all grid cells: abundant leafy vegetation is compatible with multiple climates and vegetation formations."
  }
 ],
 "uncertainties": [
  "Medium confidence: cover fractions are visual estimates, and mixed pixels make the division between sparse vegetation and bare substrate uncertain.",
  "High confidence: upsampling does not add spatial information; narrow tracks, small roofs, individual trees, and small channels cannot be reliably resolved at 10 m.",
  "High confidence: the two renderings show the same acquisition and provide no evidence of change, rotation, clearing date, or vegetation recovery rate.",
  "Medium confidence: some dark bands may be terrain shadow, denser canopy, or both; they should not be classified as streams solely from darkness.",
  "High confidence: exposed soil, rock, dry vegetation, and some built materials can overlap in displayed color.",
  "Medium confidence: apparent curved strips in the upper opening could follow terrain or human modification; terracing is not established.",
  "High confidence: feature dimensions are approximate visual measurements; counts refer to broad visible complexes rather than every constituent patch.",
  "High confidence: display stretching, interpolation, and mixed pixels can alter apparent contrast and edge sharpness.",
  "Medium confidence: no obvious cloud or cloud-shadow feature is visible, but subtle atmospheric and illumination effects cannot be excluded."
 ],
 "summaries": {
  "w25": "Probably wooded terrain contains irregular sparse-cover openings, especially centrally and northward (medium confidence). Strong red false-color response indicates extensive leafy vegetation (high confidence).",
  "w50": "The patch is probably predominantly wooded, with irregular openings concentrated in the upper center and central-to-eastern areas (medium confidence). Strong false-color red indicates extensive leafy vegetation (high confidence). Pale patches likely mix sparse cover and exposed substrate; their origin and any management remain uncertain (low confidence).",
  "w100": "This 1.2 km patch probably contains predominantly woody vegetation, with estimated cover of 68% dense vegetation, 25% sparse or mixed cover, and 7% exposed substrate (medium confidence overall). Irregular openings cluster in the upper center and central-to-eastern areas; the southwest is more continuously vegetated (high confidence). Strong red false-color response supports extensive leafy cover (high confidence). Curving boundaries and dark diagonal belts suggest terrain influence (medium confidence). A possible central track is uncertain (low confidence). No clear water body or substantial settlement is evident (medium confidence); biome, season, and active land-management processes remain unresolved.",
  "w200": "The patch is probably a predominantly wooded landscape with irregular open areas (medium confidence). Visual cover estimates are 68% dense vegetation, 25% sparse vegetation mixed with exposed ground, and 7% predominantly bare substrate; the sparse-versus-bare distinction is uncertain (medium confidence overall).\n\nA lobed pale opening with vegetated islands occupies the upper center. Additional branching openings occur centrally and along the eastern side, while the lower-left area is more continuously vegetated (high confidence). Dark southwest–northeast belts and curved boundaries suggest uneven terrain, although canopy density and shading are confounded (medium confidence).\n\nStrong red false-color response indicates widespread leafy vegetation, while muted gray-green openings imply reduced leafy cover or greater substrate exposure (high and medium confidence, respectively). These colors do not independently establish vegetation health or moisture (high confidence).\n\nA narrow central pale trace could be a track, and a small eastern bright patch could be soil, rock, or built material (low confidence). No clear open-water body or substantial settlement is evident (medium confidence). Natural openings and managed clearings remain difficult to distinguish. Crop types, forestry practices, active erosion, flooding, season, and a specific biome cannot be established from this single paired rendering (high confidence).",
  "w400": "This 1.2 × 1.2 km patch is probably predominantly wooded, with irregular sparse-cover and exposed-ground openings (medium confidence). Approximate visual fractions are 68% dense vegetation, 25% sparse vegetation mixed with substrate, and 7% predominantly exposed soil or rock. The overall division is tentative, particularly between sparse vegetation and bare ground (medium confidence).\n\nThe most distinctive pale opening lies in the upper center and extends slightly into the upper-left cell. It is lobed, contains darker vegetated islands, and measures roughly 300–350 m wide by 350–420 m long (medium confidence in dimensions). Central and eastern cells contain additional branching or elongated openings, separated by dark vegetated strips. The lower-left portion is more continuously vegetated, whereas small pale patches persist across the lower center and southeastern edge (high confidence in spatial pattern).\n\nBroad dark belts trend approximately southwest–northeast through portions of the patch. Along with curved cover boundaries, they suggest terrain-related organization or shading, but denser canopy could contribute to the same appearance (medium confidence). A discontinuous pale trace in the center may be an access track or simply an exposed terrain edge (low confidence). A compact bright patch near the lower part of the middle-right cell is visible, but its material is unresolved (high confidence in visibility; low confidence in classification).\n\nTrue color is dominated by dark and medium greens, interrupted by tan and olive-gray surfaces. False color shows extensive strong red vegetation, with muted red or gray-green openings (high confidence). This supports substantial leafy cover and locally reduced vegetation density or greater substrate exposure (high and medium confidence, respectively). Red intensity alone cannot isolate physiological vigor, moisture, species, or canopy structure (high confidence).\n\nNo unambiguous open-water body, substantial settlement, or organized crop-field pattern is apparent (medium confidence). The openings could be natural rocky areas, managed clearings, or a mixture (low confidence). Recent harvest, irrigation, erosion, flooding, and clearing are not demonstrated; their timing would require additional evidence (high confidence). A specific biome, climate, or season remains unresolved. At 10 m, mixed pixels and display interpolation limit interpretation of narrow tracks, small structures, channels, and precise boundaries (high confidence)."
 }
}
```

#### Patch 0bf20967-86c2-4e27-96df-5220f4108873

```json
{
 "global": {
  "cover": [
   {
    "class": "Tree-dominated vegetation, including possible scrub or regeneration; medium confidence",
    "percent": 75
   },
   {
    "class": "Grass or other low vegetation in openings and field margins; medium confidence",
    "percent": 10
   },
   {
    "class": "Exposed soil or sparsely vegetated agricultural parcels; medium confidence",
    "percent": 13
   },
   {
    "class": "Roads, tracks and other narrow exposed surfaces; low confidence",
    "percent": 2
   }
  ],
  "landscape": "A predominantly wooded rural mosaic with agricultural openings concentrated along the eastern side; high confidence. Percentages are approximate visual estimates, with medium confidence overall.",
  "season_cues": "Extensive green cover in true color and strong red responses in false color suggest an active vegetative period; medium confidence. Pale fields may be bare, recently worked, harvested or sparsely established; low confidence for any specific stage. The season cannot be identified reliably.",
  "heterogeneity": "Large-scale cover is dominated by a connected wooded matrix, but canopy tone, small openings and sharply bounded eastern fields create moderate heterogeneity; high confidence."
 },
 "grid": [
  {
   "cell": "top_left",
   "cover": "Predominantly woody vegetation; high confidence.",
   "texture": "Coarsely mottled dark green, with strong but variable red in false color; high confidence.",
   "features": "A pale linear corridor follows the upper edge, and a lighter elongated opening lies near the right side of the cell; medium confidence."
  },
  {
   "cell": "top_center",
   "cover": "Mostly woody vegetation, with part of a pale agricultural opening at the upper right; high confidence.",
   "texture": "Irregular dark and medium green patches with uneven red intensity; high confidence.",
   "features": "A roughly north–south lighter strip continues near the western side; the diagonal edge of the northeastern field enters the upper right; medium confidence."
  },
  {
   "cell": "top_right",
   "cover": "A substantial pale field embedded in woody vegetation; high confidence.",
   "texture": "Relatively smooth tan field surfaces contrast with mottled canopy; parallel strips are visible toward the eastern edge; high confidence.",
   "features": "An irregular field complex is crossed by an approximately east–west line, with approximately north–south internal striping in its eastern section; medium confidence."
  },
  {
   "cell": "middle_left",
   "cover": "Predominantly woody vegetation; high confidence.",
   "texture": "Dark, irregular canopy texture with scattered lighter patches; high confidence.",
   "features": "A small pale, roughly rectangular opening sits near the eastern side, with another partial exposed patch at the western boundary; medium confidence."
  },
  {
   "cell": "middle_center",
   "cover": "Woody vegetation with local changes in canopy density or composition; medium confidence.",
   "texture": "Mottled green and red, including a darker rounded patch toward the upper right; high confidence.",
   "features": "No clearly resolved buildings or water features; high confidence in their lack of clear identification."
  },
  {
   "cell": "middle_right",
   "cover": "Woodland transitioning to a green open parcel and pale agricultural ground; high confidence.",
   "texture": "Coarse canopy at the top and left, smoother green vegetation to the right, and bright pale ground near the bottom; high confidence.",
   "features": "A roughly rectangular green parcel occupies the eastern side, bordered by a narrow linear corridor at the image edge; medium confidence."
  },
  {
   "cell": "bottom_left",
   "cover": "Mostly woody vegetation with scattered lighter openings; medium confidence.",
   "texture": "Irregular, patchy canopy with locally smoother and paler vegetation; high confidence.",
   "features": "A faint diagonal line runs toward the lower edge and may be a track or vegetation boundary; low confidence."
  },
  {
   "cell": "bottom_center",
   "cover": "Mixed woodland, low vegetation and the western ends of pale cultivated parcels; medium confidence.",
   "texture": "Highly mottled vegetation grades into smoother bright surfaces toward the southeast; high confidence.",
   "features": "A broad rounded lighter patch lies toward the upper left; an elongated opening lies farther south; pale fields and narrow access-like lines enter from the right; medium confidence."
  },
  {
   "cell": "bottom_right",
   "cover": "Predominantly pale agricultural parcels interspersed with woody strips and narrow exposed corridors; medium confidence.",
   "texture": "Smooth tan and pale cyan surfaces, angular boundaries and dark vegetated separators; high confidence.",
   "features": "Several adjoining parcels, an eastern edge corridor and branching internal lines are visible. Small bright marks cannot be confirmed as buildings; medium confidence."
  }
 ],
 "boundaries": "Eastern pale parcels have relatively sharp, angular boundaries against woodland, supporting managed land divisions; high confidence. Internal woodland transitions are diffuse and irregular, consistent with differences in vegetation structure, cover or illumination; medium confidence. Open agricultural cover increases toward the northeast and especially the southeast; high confidence. No continuous riverbank or lake shoreline is resolved; high confidence.",
 "features": [
  {
   "type": "Northeastern pale field complex; high confidence in the feature, medium confidence in agricultural use and dimensions",
   "cell": "top_right, extending into top_center",
   "size_m": "Approximately 450–500 m across and 250–350 m north–south",
   "shape": "Irregular polygon with a tapered northwestern portion",
   "orientation": "Broadly northwest–southeast",
   "count": 1
  },
  {
   "type": "Parallel striping within the northeastern field; medium confidence",
   "cell": "top_right",
   "size_m": "Striped zone approximately 130–170 m wide and 120–180 m long",
   "shape": "Repeated narrow parallel bands; individual band counts are uncertain",
   "orientation": "Approximately north–south",
   "count": 1
  },
  {
   "type": "Linear field divider or access track crossing the northeastern opening; medium confidence",
   "cell": "top_right",
   "size_m": "Approximately 350–400 m long; width near the effective resolution limit",
   "shape": "Narrow, nearly straight line",
   "orientation": "Approximately east–west",
   "count": 1
  },
  {
   "type": "Upper-edge road or cleared corridor; medium confidence",
   "cell": "top_left",
   "size_m": "Approximately 350–450 m visible length and 10–30 m apparent width",
   "shape": "Slightly oblique linear strip, partly clipped",
   "orientation": "West-southwest–east-northeast",
   "count": 1
  },
  {
   "type": "Elongated vegetated opening; medium confidence",
   "cell": "top_left and top_center",
   "size_m": "Approximately 70–100 m wide and 250–300 m long",
   "shape": "Tapered strip with irregular edges",
   "orientation": "Approximately north–south",
   "count": 1
  },
  {
   "type": "Small pale clearing or low-vegetation patch; medium confidence",
   "cell": "middle_left",
   "size_m": "Approximately 60–80 m across",
   "shape": "Roughly rectangular",
   "orientation": "Approximately east–west",
   "count": 1
  },
  {
   "type": "Green open parcel; medium confidence",
   "cell": "middle_right",
   "size_m": "Approximately 150–200 m wide and 180–230 m long",
   "shape": "Roughly rectangular with an uneven northern edge",
   "orientation": "Approximately north–south",
   "count": 1
  },
  {
   "type": "Southeastern pale parcel complex; high confidence in the feature, medium confidence in agricultural use",
   "cell": "bottom_right, extending into bottom_center and middle_right",
   "size_m": "Approximately 450–500 m across and 450–500 m long within the patch",
   "shape": "Several adjoining angular parcels separated by narrow vegetated or exposed strips; subdivisions uncertain",
   "orientation": "Mixed, with broadly north–south and east–west edges",
   "count": 1
  },
  {
   "type": "Eastern road or access corridor; medium confidence",
   "cell": "middle_right and bottom_right",
   "size_m": "Approximately 550–600 m visible length; apparent width about 10–20 m",
   "shape": "Narrow, slightly bending line partly clipped by the eastern boundary",
   "orientation": "North-northeast–south-southwest",
   "count": 1
  },
  {
   "type": "Rounded lighter vegetation patch; medium confidence",
   "cell": "bottom_center, extending into bottom_left",
   "size_m": "Approximately 180–250 m across",
   "shape": "Rounded and irregular",
   "orientation": "No strong orientation",
   "count": 1
  },
  {
   "type": "Possible southwestern track or linear vegetation boundary; low confidence",
   "cell": "bottom_left",
   "size_m": "Approximately 250–350 m visible length; width unresolved",
   "shape": "Faint diagonal line",
   "orientation": "Northwest–southeast",
   "count": 1
  }
 ],
 "spectral": {
  "true_color": "Dark to medium green vegetation dominates. Pale tan and gray-beige parcels in the northeast and southeast provide the strongest brightness contrast. A smoother green parcel appears along the middle eastern edge. These visible color and texture patterns have high confidence.",
  "false_color": "Vegetated areas appear predominantly red, with bright red patches across the western woodland and eastern green parcel, and darker burgundy patches within the central vegetation. Pale true-color fields become cyan, gray-green or muted pink. Confidence in these displayed patterns is high.",
  "implications": [
   {
    "claim": "The extensive red response is consistent with substantial live vegetation and relatively strong near-infrared reflectance.",
    "confidence": "high"
   },
   {
    "claim": "Red intensity differences may reflect canopy cover, leaf area, vegetation type, physiological condition, shadow or display scaling; they do not uniquely measure vigor.",
    "confidence": "high"
   },
   {
    "claim": "Pale tan surfaces that appear cyan or gray-green in false color likely have low live vegetation cover and may include exposed soil or crop residue.",
    "confidence": "medium"
   },
   {
    "claim": "Darker vegetated patches could include shadow or structural and species differences; elevated moisture cannot be established from their darkness.",
    "confidence": "medium"
   },
   {
    "claim": "Narrow bright lines may be roads or exposed access surfaces, but their material and permeability are unresolved.",
    "confidence": "medium"
   },
   {
    "claim": "No unambiguous open-water body is visible, so water turbidity cannot be assessed.",
    "confidence": "high"
   },
   {
    "claim": "Small cyan or bright marks are insufficient evidence for buildings because bare soil and mixed pixels can produce similar colors.",
    "confidence": "high"
   }
  ],
  "not_revealed": "These rendered composites do not provide calibrated reflectances, quantitative vegetation indices, soil or canopy moisture, species identity, crop identity, canopy height, water turbidity, management history or change rates. Image stretching and mixed pixels limit quantitative comparisons. Confidence in these limitations is high."
 },
 "interpretation": [
  {
   "claim": "The patch is a wooded rural landscape with agricultural openings along its eastern side.",
   "confidence": "high",
   "evidence": "Levels 1–2, all cells: a connected mottled green/red matrix surrounds pale angular parcels in top_right and bottom_right."
  },
  {
   "claim": "The eastern parcels are likely managed fields, with different amounts of vegetation cover at acquisition.",
   "confidence": "medium",
   "evidence": "Levels 2–4, top_right, middle_right and bottom_right: sharp parcel edges, internal striping and contrasts between green/red and tan/cyan surfaces."
  },
  {
   "claim": "Some pale parcels may be fallow, recently cultivated or harvested, but the images cannot distinguish these stages.",
   "confidence": "low",
   "evidence": "Levels 3–4, top_right and bottom_right: low apparent vegetation cover and relatively smooth pale surfaces."
  },
  {
   "claim": "Crop species and crop rotation cannot be inferred reliably.",
   "confidence": "high",
   "evidence": "Levels 2–4, eastern cells: field geometry and spectral appearance lack diagnostic crop detail, and no temporal sequence is supplied."
  },
  {
   "claim": "Lighter patches within the woody matrix may represent canopy gaps, scrub or regeneration.",
   "confidence": "medium",
   "evidence": "Levels 2–4, top_left, middle_left and bottom_center: irregular lighter areas embedded in darker canopy retain some vegetation response."
  },
  {
   "claim": "Commercial forestry or recent logging is possible but not demonstrated.",
   "confidence": "low",
   "evidence": "Levels 2–3, central and western cells: openings and possible tracks occur, but no clear logging pattern or confidently identified plantation rows are visible."
  },
  {
   "claim": "The vegetation appears broadly active, but ecological health, biodiversity and disturbance severity cannot be established.",
   "confidence": "medium",
   "evidence": "Levels 1 and 4, most cells: extensive green true-color cover and red false-color response, with internal tonal variation."
  },
  {
   "claim": "The linear corridors suggest access infrastructure, but a settlement is not clearly resolved.",
   "confidence": "medium",
   "evidence": "Levels 2–3, top_left, middle_right and bottom_right: narrow connected lines are visible without a distinct cluster of identifiable roofs."
  },
  {
   "claim": "There is no clear positive evidence of active irrigation, flooding, erosion or ongoing clearing.",
   "confidence": "medium",
   "evidence": "Levels 2–4, all cells: no diagnostic irrigation geometry, inundated surface, erosion channel or fresh clearing front is resolved; smaller processes could be missed."
  },
  {
   "claim": "A seasonally green temperate or subtropical wooded agricultural setting is plausible, but biome and climate remain weakly constrained.",
   "confidence": "low",
   "evidence": "Levels 1 and 4, all cells: extensive active woody vegetation and agricultural openings are compatible with several climatic regions."
  }
 ],
 "uncertainties": [
  "High confidence: visual cover percentages are approximate rather than derived from pixel classification; woodland, scrub and low vegetation boundaries are uncertain.",
  "High confidence: native 10 m pixels mix narrow roads, field margins, small crowns and small structures; display upsampling adds no spatial information.",
  "High confidence: feature dimensions are approximate, and clipped parcels may extend substantially beyond the patch.",
  "Medium confidence: some apparent woodland may be dense scrub or regenerating vegetation.",
  "High confidence: bare soil, dry residue, sparse crops and some built surfaces can overlap in these displayed colors.",
  "High confidence: dark patches may reflect vegetation properties, shadow or rendering effects; they cannot be classified as water solely from darkness.",
  "Medium confidence: the northeastern parallel bands may represent cultivation strips, vegetation rows or processing-enhanced texture; individual rows are not reliably resolved.",
  "High confidence: bright red is not a calibrated measure of plant health, productivity or moisture.",
  "High confidence: a single paired observation cannot establish rotation, harvest timing, clearing history or active land-cover change.",
  "Medium confidence: no obvious clouds or broad cloud shadows are visible, but small canopy shadows and subtle atmospheric effects cannot be separated confidently."
 ],
 "summaries": {
  "w25": "High confidence: woodland dominates a rural mosaic with eastern fields. Medium confidence: pale parcels have sparse vegetation. Crop identity, moisture and management remain unresolved.",
  "w50": "High confidence: a connected wooded matrix dominates, with angular fields concentrated in the northeast and southeast. Medium confidence: pale fields contain exposed soil or sparse vegetation, while red false-color areas indicate active vegetation. Narrow access corridors and small openings occur. Crop types, moisture, settlement and recent management remain uncertain.",
  "w100": "High confidence: this 1.2 km patch is predominantly wooded, with agricultural openings concentrated along its eastern side. Medium-confidence visual estimates assign 75% to woody cover, 10% to low vegetation, 13% to pale fields and 2% to linear exposed surfaces. Northeastern fields show parallel bands; southeastern parcels form an angular complex. Strong red false-color responses support extensive live vegetation, but do not quantify health or moisture. Small openings and possible access tracks interrupt the canopy. No unambiguous water body or settlement is resolved. Crop identity, harvest timing, forestry activity and climate remain uncertain.",
  "w200": "High confidence: the patch contains a connected wooded matrix with managed-looking openings concentrated in the northeast and southeast. Medium-confidence cover estimates are 75% woody vegetation, 10% grass or other low vegetation, 13% exposed or sparsely vegetated fields and 2% narrow exposed corridors.\n\nThe northeastern opening has an irregular outline, an east–west dividing line and approximately north–south bands. The southeast contains adjoining angular parcels separated by dark vegetation and narrow lines. A smoother green parcel occupies the middle eastern edge. Small pale openings and diffuse canopy variations occur across the western and central areas; their interpretation as gaps or regeneration has medium confidence.\n\nExtensive red in false color indicates substantial live vegetation with high confidence. Pale tan surfaces appearing cyan or gray-green probably have low vegetation cover, with medium confidence. Red brightness alone cannot establish vigor, and dark tones cannot establish moisture. Roads or tracks are plausible, but buildings and water are not clearly identified. Specific crops, rotation, harvest, irrigation and logging remain unresolved. A temperate or subtropical setting is only a low-confidence possibility.",
  "w400": "High confidence: this 1.2 km square is a predominantly wooded rural mosaic, with agricultural-looking openings concentrated along the eastern side. Medium-confidence visual estimates assign 75% to woody vegetation, 10% to grass or other low vegetation, 13% to pale fields and 2% to roads, tracks or other narrow exposed surfaces. These estimates are not a pixel classification.\n\nThe western and central cells contain a connected, irregularly textured green canopy. Small openings and broad changes in tone interrupt this matrix. A north–south lighter strip near the upper-left/upper-center boundary, a small rectangular pale opening in middle_left and a rounded lighter patch in bottom_center are distinguishable. Their interpretation as canopy gaps, scrub or regeneration has medium confidence.\n\nIn top_right, an irregular pale field complex extends roughly 450–500 m across. An approximately east–west line crosses it, and its eastern part contains approximately north–south bands. A smoother green parcel occupies middle_right. The southeastern pale parcel complex spans approximately 450–500 m in each direction within the image and contains angular subdivisions, vegetated separators and access-like lines. Dimensions and corridor interpretations have medium confidence. A narrow corridor follows the eastern edge, while another pale line clips the upper-left boundary.\n\nHigh-confidence spectral observations include dark to medium green vegetation in true color, widespread red in false color, and pale tan fields becoming cyan or gray-green. These patterns support extensive live vegetation and, with medium confidence, reduced vegetation cover on the pale parcels. Variation in red intensity can arise from canopy structure, vegetation type, shadow and display scaling; it does not independently quantify health or moisture.\n\nAgricultural management is a medium-confidence interpretation supported by parcel geometry and internal bands. Specific crops, rotation, harvest timing and irrigation cannot be established. Forestry, recent clearing and climate assignments have low confidence. No unambiguous water body, settlement, flood pattern or erosion feature is resolved. At native 10 m resolution, small structures, narrow tracks and field edges contain mixed pixels; upsampling does not recover missing detail. A temporal sequence and additional spectral information would be needed to distinguish management stages and environmental processes."
 }
}
```

#### Patch 1bb3844f-c810-4173-b6f5-2654009e3d90

```json
{
 "global": {
  "cover": [
   {
    "class": "Dense woody vegetation, including forest and possible managed stands",
    "percent": 67
   },
   {
    "class": "Open herbaceous vegetation, including possible grassland or fields",
    "percent": 21
   },
   {
    "class": "Sparse vegetation or exposed soil in clearings",
    "percent": 10
   },
   {
    "class": "Roads, tracks and other narrow open corridors",
    "percent": 2
   }
  ],
  "landscape": "A predominantly wooded rural mosaic with irregular openings and narrow access corridors; medium confidence. Cover percentages are approximate visual estimates with medium confidence.",
  "season_cues": "Extensive green foliage and strong false-color red suggest substantial photosynthetically active vegetation; high confidence. A growing-season acquisition is plausible, but evergreen cover prevents reliable seasonal dating; low confidence.",
  "heterogeneity": "Strong contrasts between textured woody cover, smoother open parcels and thin pale corridors; high confidence. Differences among wooded blocks may reflect stand composition, density or management; medium confidence."
 },
 "grid": [
  {
   "cell": "top_left",
   "cover": "Predominantly woody vegetation with a partly clipped pale opening along the left edge; medium confidence.",
   "texture": "Coarse, mottled canopy surrounding smoother brownish and muted-green patches; high confidence.",
   "features": "An irregular upper-left clearing and the northern tip of a larger opening extending into middle_left; high confidence."
  },
  {
   "cell": "top_center",
   "cover": "Woody vegetation mixed with an elongated green opening; medium confidence.",
   "texture": "Smoother central vegetation bordered by darker, more heterogeneous canopy; high confidence.",
   "features": "The upper portion of a central open parcel and faint linear stand boundaries; medium confidence."
  },
  {
   "cell": "top_right",
   "cover": "Mostly woody vegetation surrounding a narrow pale corridor; medium confidence.",
   "texture": "Dark canopy blocks with variable false-color red intensity; high confidence.",
   "features": "A roughly north–south winding corridor bends southeast near the upper portion, with a small pale lateral extension; high confidence for its presence, low confidence for its function."
  },
  {
   "cell": "middle_left",
   "cover": "Woody vegetation surrounding a large brownish clearing; high confidence.",
   "texture": "The clearing is relatively smooth but contains several dark vegetation-like inclusions; medium confidence.",
   "features": "An irregular open parcel with an angular northern edge and a rounded southern end; high confidence."
  },
  {
   "cell": "middle_center",
   "cover": "Mixed open herbaceous vegetation and woody margins; medium confidence.",
   "texture": "Relatively smooth olive-green surfaces interrupted by darker lobes and a pale line; high confidence.",
   "features": "A pale cross-patch track appears to connect the central openings with the eastern winding corridor; medium confidence."
  },
  {
   "cell": "middle_right",
   "cover": "Predominantly woody vegetation, including a relatively dark angular block; medium confidence.",
   "texture": "Broad, comparatively uniform dark patches bounded by narrow contrasting strips; high confidence.",
   "features": "The winding corridor follows the western side of several angular vegetation blocks; medium confidence."
  },
  {
   "cell": "bottom_left",
   "cover": "Mostly woody vegetation with small open patches; medium confidence.",
   "texture": "Mottled canopy crossed by a smooth bright linear feature; high confidence.",
   "features": "A curved pale road or substantial track enters from the left edge and continues toward bottom_center; high confidence."
  },
  {
   "cell": "bottom_center",
   "cover": "An elongated pale grassy or sparsely vegetated opening surrounded by woodland; medium confidence.",
   "texture": "The opening is mottled internally but smoother than adjacent canopy; high confidence.",
   "features": "A north–south open parcel occupies the eastern part of the cell; the southwestern road curves out through the lower edge; high confidence."
  },
  {
   "cell": "bottom_right",
   "cover": "Predominantly woody vegetation with a smaller angular opening; medium confidence.",
   "texture": "Dark canopy contrasts with a lighter triangular or quadrilateral patch and a thin pale corridor; high confidence.",
   "features": "The winding corridor continues toward the lower edge, west of the smaller opening; high confidence for geometry, low confidence for corridor type."
  }
 ],
 "boundaries": "Open–wooded boundaries are mostly distinct, irregular or locally angular; high confidence. A sequence of central openings extends from top_center through middle_center into bottom_center, bordered by wooded areas; high confidence. The eastern winding corridor separates this central mosaic from darker vegetation blocks in middle_right and bottom_right; medium confidence. Some straight edges suggest parcel or stand boundaries, while scalloped margins suggest uneven woody encroachment or retained vegetation; medium confidence. No consistent scene-wide moisture or elevation gradient can be established; high confidence.",
 "features": [
  {
   "type": "Partly clipped sparse-vegetation clearing; medium confidence for classification and approximate geometry",
   "cell": "top_left",
   "size_m": "Approximately 150–200 wide and 250–330 long within the image",
   "shape": "Irregular, partly outside the image",
   "orientation": "Approximately north–south",
   "count": 1
  },
  {
   "type": "Brownish open parcel with scattered dark inclusions; high confidence for presence, medium confidence for approximate geometry",
   "cell": "top_left, top_center, middle_left, middle_center",
   "size_m": "Approximately 250–320 wide and 350–430 long",
   "shape": "Irregular polygon with an angular upper boundary and tapering southern end",
   "orientation": "North-northwest–south-southeast",
   "count": 1
  },
  {
   "type": "Elongated vegetated opening; medium confidence for classification, delineation and geometry",
   "cell": "top_center, middle_center",
   "size_m": "Approximately 200–280 wide and 350–480 long",
   "shape": "Elongated, with uneven wooded margins",
   "orientation": "Approximately north–south",
   "count": 1
  },
  {
   "type": "Central herbaceous opening; medium confidence for classification and geometry",
   "cell": "middle_center",
   "size_m": "Approximately 250–330 across",
   "shape": "Broad irregular polygon with woody indentations",
   "orientation": "Weak northwest–southeast elongation",
   "count": 1
  },
  {
   "type": "Pale southern open parcel; medium confidence for classification and geometry",
   "cell": "bottom_center",
   "size_m": "Approximately 120–180 wide and 300–380 long",
   "shape": "Elongated, locally tapering",
   "orientation": "Approximately north–south",
   "count": 1
  },
  {
   "type": "Small angular vegetated opening; medium confidence for classification and geometry",
   "cell": "middle_right, bottom_right",
   "size_m": "Approximately 160–220 by 150–210",
   "shape": "Irregular quadrilateral or wedge",
   "orientation": "Northwest–southeast",
   "count": 1
  },
  {
   "type": "Curved road or substantial track; high confidence for presence, medium confidence for dimensions",
   "cell": "bottom_left, bottom_center",
   "size_m": "Approximately 550–650 visible length; width unresolved, likely around one 10 m pixel or less",
   "shape": "Gently curving linear feature",
   "orientation": "West-northwest–east-southeast, bending south at the lower edge",
   "count": 1
  },
  {
   "type": "Pale winding corridor, possibly a track or drainage feature; high confidence for presence, low confidence for function, medium confidence for geometry",
   "cell": "top_right, middle_right, bottom_right",
   "size_m": "Approximately 1,200–1,350 visible path length; width unresolved at 10 m",
   "shape": "Narrow, sinuous and locally discontinuous",
   "orientation": "Approximately north–south",
   "count": 1
  },
  {
   "type": "Probable connecting track; medium confidence for classification and geometry",
   "cell": "middle_center, middle_right",
   "size_m": "Approximately 300–380 visible length; width unresolved at 10 m",
   "shape": "Bent linear feature",
   "orientation": "Approximately west–east",
   "count": 1
  }
 ],
 "spectral": {
  "true_color": "Woody areas are predominantly dark green, while openings range from olive green to pale tan and brown; high confidence. Narrow corridors are pale gray or whitish and contrast strongly with their surroundings; high confidence. Bright roofs or a coherent built-up pattern are not apparent; medium confidence.",
  "false_color": "Much of the textured woody cover becomes strong red, while several eastern blocks remain darker maroon; high confidence. Central vegetated openings show muted pink or red, and the large brownish western opening becomes gray-green; high confidence. Narrow corridors remain pale cyan or whitish; high confidence.",
  "implications": [
   {
    "claim": "Strong red over much of the woody cover indicates substantial near-infrared response consistent with abundant live foliage.",
    "confidence": "high"
   },
   {
    "claim": "Muted red and gray-green openings likely have less live green vegetation, more exposed soil, dry vegetation or some combination of these.",
    "confidence": "medium"
   },
   {
    "claim": "Differences between bright-red and dark-maroon vegetation may reflect canopy density, vegetation type, shadow or phenological differences; they do not uniquely measure vigor.",
    "confidence": "high"
   },
   {
    "claim": "Brown true-color and gray-green false-color surfaces are compatible with exposed soil or senescent vegetation, but cannot reliably distinguish them.",
    "confidence": "medium"
   },
   {
    "claim": "The pale linear surfaces are compatible with roads or tracks, but their spectra do not establish paving material or exclude drainage-related features.",
    "confidence": "medium"
   },
   {
    "claim": "No clearly resolved open-water body supports an assessment of water turbidity.",
    "confidence": "high"
   },
   {
    "claim": "Moisture differences cannot be isolated from these display colors because vegetation structure, soil background and shadow can produce similar contrasts.",
    "confidence": "high"
   }
  ],
  "not_revealed": "These renderings do not provide calibrated reflectance, quantitative vegetation indices, soil moisture, canopy height, species identification or reliable crop identification; high confidence. They cannot establish irrigation, water turbidity, management history or temporal change; high confidence. Display stretching and mixed pixels limit interpretation of relative brightness; high confidence."
 },
 "interpretation": [
  {
   "claim": "The patch is a wooded rural landscape containing multiple open parcels and access routes.",
   "confidence": "high",
   "evidence": "Levels 1–3: extensive textured green/red cover across all cells, central openings and the curved road in bottom_left and bottom_center."
  },
  {
   "claim": "Some openings are likely managed fields, pasture, meadow or maintained clearings.",
   "confidence": "medium",
   "evidence": "Levels 2–4: smoother surfaces, locally angular margins and track access in middle_left, middle_center and bottom_center."
  },
  {
   "claim": "Crop species, grazing, mowing and crop rotation cannot be determined.",
   "confidence": "high",
   "evidence": "Levels 2–4: the open parcels lack diagnostic resolved crop patterns, and only one acquisition is available."
  },
  {
   "claim": "Some eastern woody blocks may represent managed forest stands.",
   "confidence": "medium",
   "evidence": "Levels 2 and 4: angular boundaries and relatively uniform dark-green/dark-maroon blocks in top_right and middle_right."
  },
  {
   "claim": "Much of the woody vegetation carries substantial live foliage, without a clearly resolved widespread canopy-loss signature.",
   "confidence": "medium",
   "evidence": "Levels 1 and 4: extensive strong false-color red across top_left, top_center, middle_left and bottom_left."
  },
  {
   "claim": "The pale and brown openings could reflect cutting, harvest, senescence or sparse ground cover, but a recent clearing or harvest event is not established.",
   "confidence": "low",
   "evidence": "Levels 3–4: brown/gray-green surfaces in middle_left and pale muted-red vegetation in bottom_center; no temporal comparison."
  },
  {
   "claim": "The woodland is locally interrupted by open parcels and narrow corridors, while remaining visually connected around much of the patch.",
   "confidence": "high",
   "evidence": "Levels 1–3: central openings and eastern and southwestern corridors embedded within contiguous woody cover."
  },
  {
   "claim": "There is no convincing resolved evidence of a settlement, irrigation installation, active erosion or flooding.",
   "confidence": "medium",
   "evidence": "Levels 2–4: no coherent roof cluster, diagnostic irrigation geometry, obvious sediment plume, broad inundated surface or clear erosion channel."
  },
  {
   "claim": "A temperate forest–open-land mosaic is plausible, but biome and climate cannot be assigned reliably.",
   "confidence": "low",
   "evidence": "Levels 1 and 4: extensive green woody vegetation and interspersed open parcels are compatible with this setting but are not geographically diagnostic."
  }
 ],
 "uncertainties": [
  "Medium confidence: cover percentages may differ by several percentage points because woody margins, sparse vegetation and open parcels have mixed pixels.",
  "High confidence: the 10 m source resolution does not reliably resolve individual buildings, narrow road widths, small drainage channels, fences or isolated trees.",
  "Low confidence: the eastern winding corridor's function is ambiguous; a track, drainage feature or combination remains possible.",
  "Medium confidence: dark eastern blocks could differ in vegetation composition, canopy density or illumination; shadow cannot be separated cleanly.",
  "Medium confidence: the large brownish opening may contain exposed soil, dry herbaceous material or recently managed vegetation.",
  "Medium confidence: the central openings may form connected parcels rather than the separate units used in the feature inventory.",
  "High confidence: feature dimensions are approximate visual estimates; display upsampling adds no spatial information.",
  "High confidence: narrow bright features may be widened or rendered discontinuous by mixed pixels, resampling and display contrast.",
  "Medium confidence: no obvious cloud or broad cloud-shadow feature is visible, although subtle illumination effects cannot be excluded.",
  "High confidence: season, management history and active change cannot be established from a single paired rendering."
 ],
 "summaries": {
  "w25": "High confidence: woodland dominates a mosaic of central openings and pale corridors. Medium confidence: openings include managed grassland or fields; management timing remains unresolved.",
  "w50": "High confidence: woodland dominates, interrupted by central openings and narrow pale corridors. Medium-confidence estimates assign 67% woody cover, 21% herbaceous cover, 10% sparse vegetation or soil, and 2% corridors. Strong false-color red indicates abundant live foliage. Field use and eastern corridor function remain uncertain.",
  "w100": "High confidence: this is a predominantly wooded mosaic with central open parcels and pale linear corridors. Medium-confidence cover estimates are 67% woody vegetation, 21% herbaceous vegetation, 10% sparse vegetation or soil, and 2% corridors. Strong false-color red indicates abundant live foliage, while brown and gray-green openings likely contain less green vegetation; medium confidence. A curved southwestern road is clear; the eastern winding corridor has uncertain function. Managed fields or grassland and forestry blocks are plausible; medium confidence. Crop identity, moisture, recent harvest, season and climate remain unresolved.",
  "w200": "High confidence: the patch is a predominantly wooded landscape interrupted by central openings and narrow corridors. Medium-confidence visual estimates assign 67% to woody vegetation, 21% to herbaceous vegetation, 10% to sparse vegetation or exposed soil, and 2% to corridors.\n\nHigh confidence: a large brownish opening lies left of center, greener openings occupy the center, and an elongated pale parcel extends southward. A curved road crosses the southwest. A thin winding corridor runs approximately north–south in the eastern third; its interpretation as a track or drainage feature has low confidence.\n\nStrong false-color red across much of the canopy indicates abundant live foliage; high confidence. Muted red and gray-green openings likely have less green cover or more soil and dry vegetation; medium confidence. Darker eastern vegetation blocks may reflect different stand composition, density or management; medium confidence.\n\nManaged open parcels and possible forestry are medium-confidence interpretations. Crop types, rotation, recent harvest, irrigation and moisture cannot be established. No clearly resolved water body supports turbidity analysis; high confidence. A temperate woodland setting is plausible with low confidence. At 10 m, small structures, narrow channels and individual trees remain unresolved.",
  "w400": "High confidence: this 1.2 km square patch depicts a predominantly wooded mosaic with irregular open parcels and narrow pale corridors. Medium-confidence visual estimates assign 67% to dense woody vegetation, 21% to open herbaceous vegetation, 10% to sparse vegetation or exposed soil, and 2% to roads, tracks and other corridors. The boundaries between these classes remain approximate.\n\nHigh confidence: the upper-left edge contains a partly clipped clearing, and a larger brownish opening occupies the area left of center. Greener, smoother openings extend through the central column. A pale elongated parcel occupies bottom_center, while a smaller angular opening lies farther east. Woody vegetation surrounds these openings and remains connected around much of the patch. Some eastern blocks have comparatively straight boundaries and uniform dark tones; medium confidence that these represent distinct stands.\n\nA curved pale road or substantial track crosses bottom_left and exits through bottom_center; high confidence. A probable connecting track crosses middle_center toward an approximately north–south winding corridor in the eastern third; medium confidence. The eastern corridor is clearly visible, but its identification as a road, drainage feature or combination has low confidence. Its width is unresolved.\n\nHigh confidence: much of the woody cover is dark green in true color and strongly red in the near-infrared composite, consistent with abundant live foliage. The large brownish opening becomes gray-green, suggesting reduced green cover, exposed soil or dry vegetation; medium confidence. Dark-maroon eastern blocks could differ in canopy density, vegetation composition or illumination. Their color alone does not establish poor vegetation health; high confidence.\n\nManaged grassland, fields or maintained clearings are plausible, as is some forest management; medium confidence. Specific crops, rotation, grazing and recent cutting remain unidentified. There is no convincing resolved evidence of settlement, irrigation, flooding or active erosion; medium confidence. No clear open-water body permits turbidity assessment; high confidence.\n\nA temperate forest–open-land setting is plausible with low confidence, but climate and season are not diagnostic. High confidence: one paired acquisition cannot establish change, and 10 m mixed pixels plus display resampling limit small-feature classification, dimensions and quantitative spectral interpretation."
 }
}
```

#### Patch 9931d1e8-7c27-4411-bb57-415b841dbb3c

```json
{
 "global": {
  "cover": [
   {
    "class": "Woodland and dense woody vegetation; medium confidence",
    "percent": 48
   },
   {
    "class": "Green agricultural fields and other managed herbaceous vegetation; medium confidence",
    "percent": 34
   },
   {
    "class": "Bare, sparsely vegetated, or senescent agricultural parcels; medium confidence",
    "percent": 14
   },
   {
    "class": "Roads, tracks, and associated narrow margins; medium confidence",
    "percent": 3
   },
   {
    "class": "Other small open or disturbed patches; low confidence",
    "percent": 1
   }
  ],
  "landscape": "A rural agricultural–woodland mosaic, with fields concentrated toward the upper and left portions and extensive woody cover on the right and below; high confidence.",
  "season_cues": "Strong vegetation responses coexist with pale, weakly vegetated parcels, indicating contrasting growth or management stages; high confidence. An active growing period is plausible, but the calendar season is unresolved; medium confidence.",
  "heterogeneity": "High spatial heterogeneity: angular fields, irregular woodland edges, narrow connecting vegetation strips, and a sinuous pale corridor; high confidence."
 },
 "grid": [
  {
   "cell": "top_left",
   "cover": "Predominantly agricultural vegetation with several pale parcels; high confidence.",
   "texture": "Relatively smooth field interiors with abrupt tonal boundaries; high confidence.",
   "features": "Elongated pale strips alternate with green parcels, mostly aligned northwest–southeast; medium confidence."
  },
  {
   "cell": "top_center",
   "cover": "Mixed green and weakly vegetated agricultural parcels; high confidence.",
   "texture": "Smooth rectangular and tapered patches with contrasting spectral responses; high confidence.",
   "features": "Several long narrow fields and a pale diagonal corridor meeting field boundaries near the lower part of the cell; medium confidence."
  },
  {
   "cell": "top_right",
   "cover": "Woodland mixed with a small cluster of agricultural openings; high confidence.",
   "texture": "Coarse, mottled woody cover surrounding smoother pale parcels; high confidence.",
   "features": "Irregular woodland edge encloses short rectangular and angular fields; high confidence."
  },
  {
   "cell": "middle_left",
   "cover": "Green agricultural land above, transitioning into woody vegetation below; high confidence.",
   "texture": "Broad smooth patches become darker and more mottled southward; high confidence.",
   "features": "Indented field–woodland boundary and a pale parcel clipped by the western edge; high confidence."
  },
  {
   "cell": "middle_center",
   "cover": "Fields, woody vegetation, and a curved herbaceous strip; medium confidence.",
   "texture": "Strong contrast between smooth agricultural surfaces and irregular dark vegetation; high confidence.",
   "features": "A narrow pale corridor bends southward beside a brighter green strip; pale agricultural parcels lie immediately east; medium confidence."
  },
  {
   "cell": "middle_right",
   "cover": "Predominantly woodland, with agricultural land at the upper-left edge; high confidence.",
   "texture": "Coarse and mottled, with irregular bright and dark vegetation patches; high confidence.",
   "features": "A sharply indented western woodland boundary borders pale fields; high confidence."
  },
  {
   "cell": "bottom_left",
   "cover": "Mostly woody vegetation with small open margins; medium confidence.",
   "texture": "Dark, irregular vegetation interrupted by pale linear and patchy surfaces; high confidence.",
   "features": "A winding corridor and a locally wider pale opening near the lower-center portion of the cell; medium confidence."
  },
  {
   "cell": "bottom_center",
   "cover": "Woody vegetation with one conspicuous pale field and scattered small openings; high confidence.",
   "texture": "Predominantly mottled, with a smooth angular field and narrow linear breaks; high confidence.",
   "features": "A pale polygonal field lies beside a strong bend in the corridor; small light patches occur farther east and south; medium confidence."
  },
  {
   "cell": "bottom_right",
   "cover": "Predominantly woodland; high confidence.",
   "texture": "Continuous coarse vegetation texture with variable brightness; high confidence.",
   "features": "A faint diagonal linear opening or vegetation boundary extends northeast–southwest near the lower portion; low confidence."
  }
 ],
 "boundaries": "Angular, relatively sharp boundaries separate agricultural parcels across the top and upper-left portions; high confidence. Irregular woodland margins dominate the transition into the right and lower portions; high confidence. A sinuous central corridor follows parts of this transition; medium confidence. Increasing coarse texture toward the east and south indicates greater woody cover, while tonal differences within woodland could reflect canopy structure, illumination, or species composition; medium confidence.",
 "features": [
  {
   "type": "Agricultural parcel mosaic; high confidence",
   "cell": "top_left, top_center, top_right, middle_left, middle_center",
   "size_m": "Individual visible parcels approximately 40–180 m wide and 100–400 m long; medium confidence",
   "shape": "Rectangular, elongated, trapezoidal, and irregular edge parcels",
   "orientation": "Mostly north–south to northwest–southeast, with some east–west parcels",
   "count": 22
  },
  {
   "type": "Extensive woodland complex with connected lobes; high confidence",
   "cell": "top_right, middle_right, bottom_right, bottom_center, bottom_left, middle_left",
   "size_m": "Extends approximately 1,200 m along the eastern edge, with lobes several hundred meters across; medium confidence",
   "shape": "Irregular continuous mass with indented agricultural margins",
   "orientation": "Broadly north–south on the east, spreading westward across the south",
   "count": 1
  },
  {
   "type": "Likely road or track corridor; medium confidence",
   "cell": "top_left, top_center, middle_center, bottom_center, bottom_left",
   "size_m": "Approximately 1,200–1,600 m of potentially connected visible route; apparent width roughly 10–20 m, unresolved at native resolution; low confidence",
   "shape": "Narrow diagonal upper segment becoming sinuous, with pronounced lower bends",
   "orientation": "Generally north–south, bending southwest toward the bottom-left",
   "count": 1
  },
  {
   "type": "Curved green field margin or herbaceous strip; medium confidence",
   "cell": "middle_center",
   "size_m": "Approximately 200–280 m long and 25–60 m wide; medium confidence",
   "shape": "Crescent-like strip bordering the curved corridor",
   "orientation": "North–south with an eastward bulge",
   "count": 1
  },
  {
   "type": "Pale agricultural parcel group beside the central bend; high confidence",
   "cell": "middle_center, middle_right",
   "size_m": "Group approximately 180–260 m across; medium confidence",
   "shape": "Adjacent bands and tapered polygons",
   "orientation": "Approximately east–west to northwest–southeast",
   "count": 3
  },
  {
   "type": "Isolated pale agricultural field; high confidence",
   "cell": "bottom_center",
   "size_m": "Approximately 80–110 m wide and 120–160 m long; medium confidence",
   "shape": "Angular quadrilateral tapering toward the corridor",
   "orientation": "Northeast–southwest",
   "count": 1
  },
  {
   "type": "Pale open area beside a lower corridor bend; medium confidence",
   "cell": "bottom_left",
   "size_m": "Approximately 80–150 m across; low confidence",
   "shape": "Irregular, locally branching patch",
   "orientation": "Approximately northwest–southeast",
   "count": 1
  },
  {
   "type": "Possible narrow woodland track or linear opening; low confidence",
   "cell": "bottom_center, bottom_right",
   "size_m": "Approximately 350–500 m long; width unresolved; low confidence",
   "shape": "Faint, slightly curving linear feature",
   "orientation": "Northeast–southwest",
   "count": 1
  }
 ],
 "spectral": {
  "true_color": "Fields range from deep green through muted green to pale tan. Woodland is generally dark green with coarse brightness variations. The central curved strip is unusually bright green, while narrow corridors and some small openings are pale gray or beige. These visible contrasts have high confidence.",
  "false_color": "Many agricultural parcels are vivid red, while pale true-color fields become cyan, gray-green, or muted gray. Woodland ranges from dark burgundy to brighter mottled red. The central curved strip is pink-red, supporting vegetation cover. These observed patterns have high confidence.",
  "implications": [
   {
    "claim": "Vivid red fields contain substantial photosynthetically active vegetation, although display intensity alone does not measure physiological health.",
    "confidence": "high"
   },
   {
    "claim": "Cyan and muted agricultural parcels likely have less green vegetation, with exposed soil, dry residue, or recently worked surfaces contributing to their appearance.",
    "confidence": "medium"
   },
   {
    "claim": "The bright green curved strip is vegetated rather than an obvious open-water surface, because it also appears pink-red in false color.",
    "confidence": "high"
   },
   {
    "claim": "Darker woodland red can result from canopy structure, shadow, or vegetation differences; it does not establish poorer vegetation condition than adjacent fields.",
    "confidence": "high"
   },
   {
    "claim": "Moisture differences may contribute to tonal variation, but wet and dry surfaces cannot be separated reliably from these displays alone.",
    "confidence": "low"
   },
   {
    "claim": "No clearly identifiable open-water body is visible, so water turbidity cannot be assessed.",
    "confidence": "medium"
   },
   {
    "claim": "Pale narrow surfaces may include roads or tracks, but their colors do not reliably distinguish pavement, gravel, bare soil, or small roofs.",
    "confidence": "medium"
   }
  ],
  "not_revealed": "These rendered composites do not provide calibrated reflectance, quantitative vegetation indices, soil or canopy moisture, water turbidity, crop identity, canopy height, tree species, surface material, or change through time; high confidence. Display stretching and mixed pixels limit direct radiometric comparisons; high confidence."
 },
 "interpretation": [
  {
   "claim": "Agriculture is a major land use, organized into multiple managed parcels.",
   "confidence": "high",
   "evidence": "Levels 2–3; top_left, top_center, middle_left, middle_center: angular boundaries, smooth interiors, and repeated elongated field shapes."
  },
  {
   "claim": "The field mosaic contains contrasting vegetation or management stages.",
   "confidence": "high",
   "evidence": "Levels 1 and 4; top_left, top_center, middle_center: adjacent vivid-red vegetated parcels and cyan or muted low-cover parcels."
  },
  {
   "claim": "Some pale parcels may be harvested, fallow, recently cultivated, or carrying dry vegetation; these alternatives cannot be distinguished.",
   "confidence": "medium",
   "evidence": "Levels 2 and 4; top_center, middle_center, bottom_center: smooth pale fields with weak red responses."
  },
  {
   "claim": "Specific crop types and crop rotation sequences are unresolved.",
   "confidence": "high",
   "evidence": "Levels 2 and 4; agricultural cells: parcel geometry and a single spectral snapshot provide no diagnostic crop or temporal sequence."
  },
  {
   "claim": "Woody habitat is extensive and appears broadly connected across the eastern and southern portions.",
   "confidence": "medium",
   "evidence": "Levels 1–3; middle_right, bottom_right, bottom_center, bottom_left: adjoining coarse-textured vegetation masses."
  },
  {
   "claim": "Agricultural boundaries create substantial woodland edge habitat, but biodiversity and ecological condition cannot be evaluated.",
   "confidence": "medium",
   "evidence": "Levels 2–3; middle_left, middle_center, top_right: indented interfaces between fields and woody cover."
  },
  {
   "claim": "The sinuous pale corridor is likely a rural access route, although some segments remain ambiguous.",
   "confidence": "medium",
   "evidence": "Levels 2–3; top_center, middle_center, bottom_center, bottom_left: narrow pale continuity, field access geometry, and repeated bends."
  },
  {
   "claim": "No substantial settlement or large industrial development is apparent; isolated structures cannot be excluded.",
   "confidence": "medium",
   "evidence": "Levels 2–3; all cells: no clear building blocks or extensive impervious areas, but several small pale patches remain unresolved."
  },
  {
   "claim": "The images do not establish active irrigation, flooding, erosion, logging, or recent clearing.",
   "confidence": "high",
   "evidence": "Levels 3–4; all cells: no diagnostic infrastructure, inundation pattern, erosion network, or unequivocal fresh clearing is resolved."
  },
  {
   "claim": "A vegetated woodland–agriculture setting compatible with a temperate growing season is plausible, but biome and climate cannot be assigned securely.",
   "confidence": "low",
   "evidence": "Levels 1 and 4; all cells: extensive woody cover and vigorous agricultural vegetation, without diagnostic geographic or climatic indicators."
  }
 ],
 "uncertainties": [
  "Cover percentages are visual estimates, not a classified pixel inventory; several percentage points of uncertainty per major class are plausible, with medium confidence.",
  "The agricultural parcel count is approximate because some boundaries are faint or clipped; medium confidence. Separately listed field features are subsets of the parcel mosaic.",
  "Feature dimensions are estimated from the 1,200 m patch width; medium confidence. Apparent widths near 10 m are especially unreliable.",
  "Grassland, forage crops, other green crops, and low woody vegetation can overlap spectrally and spatially at this resolution; high confidence.",
  "Bare soil, dry residue, senescent crops, and lightly vegetated surfaces are ambiguous within pale fields; high confidence.",
  "Roads, drainage lines, and narrow clearings may resemble one another where they cross dark vegetation; medium confidence.",
  "Small buildings, vehicles, individual trees, ditches, fences, and narrow hedgerows cannot be resolved reliably; high confidence.",
  "Dark patches may combine canopy shading, illumination effects, and vegetation differences; medium confidence. Terrain relief cannot be determined securely.",
  "Upsampling, interpolation, image stretching, and mixed pixels may exaggerate smoothness, width, or color at boundaries; high confidence.",
  "No obvious cloud cover is visible, with high confidence; individual dark areas cannot confidently be labeled cloud shadows.",
  "Season, species, management history, and active change remain uncertain without temporal observations or additional measurements; high confidence."
 ],
 "summaries": {
  "w25": "An agricultural–woodland mosaic dominates (high confidence), with contrasting field vegetation and a winding access corridor (medium confidence). Season and crop identities remain unresolved.",
  "w50": "Fields dominate the upper and left portions, while woodland occupies much of the east and south (high confidence). Bright-red vegetation contrasts with pale, weakly vegetated parcels (high confidence). A sinuous corridor likely provides access (medium confidence). Crop identities, season, moisture status, and recent management remain unresolved.",
  "w100": "This 1.2 km patch contains an agricultural–woodland mosaic (high confidence), estimated as 48% woody cover, 34% green managed vegetation, 14% pale agricultural surfaces, and 4% corridors or other openings (medium confidence). Angular fields dominate the upper and left portions; woodland is concentrated eastward and southward. Strong red responses indicate abundant active vegetation, while cyan parcels suggest lower green cover (medium confidence). A winding pale corridor likely represents rural access, beside a curved vegetated strip (medium confidence). No clear open-water body or substantial settlement is apparent (medium confidence). Crop types, season, moisture, and management history remain unresolved.",
  "w200": "The patch is a heterogeneous agricultural–woodland mosaic (high confidence). Visually estimated cover comprises 48% woodland, 34% green managed vegetation, 14% bare or weakly vegetated agricultural parcels, and 4% corridors and other openings (medium confidence). Smooth angular fields occupy much of the upper and left portions, while coarse woody vegetation dominates the east and south. Irregular woodland margins create extensive field edges (high confidence).\n\nA narrow pale corridor crosses the upper fields and becomes sinuous through the center and lower-left portion; interpretation as a road or track has medium confidence. A conspicuous curved green strip beside its central bend also appears pink-red in false color, supporting vegetation cover (high confidence). An isolated pale field lies beside a lower bend.\n\nVivid-red fields indicate substantial active vegetation, while pale cyan or muted parcels likely contain less green cover (medium confidence). Harvest, fallow, cultivation, and senescence are possible explanations, but none is uniquely established. Woodland darkness may reflect canopy structure and shadow rather than reduced vigor (high confidence).\n\nNo clear water body or substantial settlement is apparent (medium confidence). Crop identity, rotation, moisture, forestry activity, climate, and recent change cannot be established from these renderings.",
  "w400": "This 1.2 × 1.2 km Sentinel-2 patch shows a heterogeneous agricultural–woodland mosaic (high confidence). Approximate visual cover estimates are 48% woodland or dense woody vegetation, 34% green agricultural or managed herbaceous vegetation, 14% pale agricultural surfaces, 3% access corridors and margins, and 1% other openings (medium confidence). These estimates describe apparent cover rather than a measured classification.\n\nThe upper-left and upper-center cells contain smooth, angular parcels with contrasting vegetation responses. Several elongated fields run north–south to northwest–southeast. Small agricultural openings also indent the upper-right woodland edge. Across the middle-left and middle-center cells, broad fields transition into irregular woody cover. Woodland dominates the middle-right and much of the bottom row, forming an apparently connected eastern and southern complex (medium confidence).\n\nA narrow pale corridor runs diagonally through the upper fields, curves southward near the center, and continues through pronounced lower bends. A road or track interpretation has medium confidence; continuity and individual segments remain uncertain. Beside the central bend, a curved bright-green strip becomes pink-red in false color, indicating vegetation (high confidence). Pale agricultural bands lie immediately east. A smaller angular pale field borders a lower bend, and an irregular opening occurs farther southwest. Tiny light patches cannot reliably be identified as buildings.\n\nTrue color separates dark-green woody cover, green fields, and pale tan parcels. False color emphasizes vivid-red vegetation in many fields and darker, mottled red in woodland. This supports abundant active vegetation, but red brightness alone cannot quantify vigor or health (high confidence). Cyan or muted fields probably have reduced green cover, potentially representing exposed soil, residue, senescence, or recent cultivation (medium confidence). Differences in woodland brightness may reflect canopy structure, species, or illumination.\n\nNo clearly identifiable open-water body or substantial settlement is apparent (medium confidence). Water turbidity and moisture status cannot be assessed reliably. The images do not establish active irrigation, flooding, erosion, logging, or recent clearing. Agriculture is clear, but crop species, rotation, and forestry management remain unresolved. A temperate growing-season setting is plausible only with low confidence; biome and climate are not diagnostic. Native 10 m resolution, mixed pixels, display stretching, and upsampling limit narrow-feature dimensions and small-object identification (high confidence)."
 }
}
```

### A2. Retest caption summaries

#### Patch fc0369af-2c2d-404e-af72-f649b48fd310

- **w25** (24 words): High confidence: dark water surrounds irregular vegetated islands. Medium confidence: pale shores are partly rocky. Season, management and coastal versus inland setting remain unresolved.

- **w50** (46 words): High confidence: this is an island-rich waterscape with irregular vegetated landmasses and connected channels. Medium-confidence cover estimates are 58% water, 35% vegetation and 7% exposed or sparse surfaces. Red false-color interiors indicate vegetation. Pale shores may be rocky. Season, biome, water quality and management remain unresolved.

- **w100** (89 words): High confidence: interconnected dark water surrounds irregular vegetated islands and edge-clipped landmasses. Medium-confidence estimates assign 58% to water, 35% to vegetation and 7% to exposed or sparsely vegetated surfaces. High confidence: vegetation appears extensively red in false color, with pale shoreline fringes and internal patches. Northern channels open into broader central and southern water. Medium confidence: woody vegetation and rocky shores are plausible. No clear fields, roads or settlement are resolved. High confidence: these renderings cannot establish coastal versus inland water, quantitative vegetation health, season or active landscape change.

- **w200** (171 words): High confidence: the patch depicts an island-rich waterscape with irregular vegetated landmasses, narrow channels and broader central and southern water. Medium-confidence area estimates are 58% open water, 35% vegetation and 7% exposed or sparsely vegetated surfaces.

High confidence: a large northern landmass extends southward, a fragmented island complex occupies the northwest, and broad land enters from the southeast. Smaller islands punctuate the middle, while an elongated island lies near the southwestern margin. Pale fringes outline many shores.

High confidence: land interiors are muted green-brown in true color and extensively red in false color, supporting substantial vegetation cover. Medium confidence: some cover is woody and some pale surfaces are exposed rock. Shoreline shallows and mixed pixels are alternative explanations for the fringes.

High confidence: no organized agricultural pattern is evident. Medium confidence: roads, settlement and forestry management are not demonstrated. High confidence: vegetation stress, moisture, water depth, salinity and active erosion cannot be established. A temperate or boreal setting is plausible with low confidence. Coastal archipelago and island-rich lake interpretations remain unresolved.

- **w400** (325 words): High confidence: this 1.2 km square patch is an island-rich waterscape dominated by connected dark water and irregular vegetated land. Medium-confidence visual estimates are 58% water, 35% vegetation and 7% exposed or sparsely vegetated surfaces; these fractions are approximate.

High confidence: the northern half is relatively land-rich. Its largest landmass continues beyond the upper edge and tapers southward between water passages. A lobed island complex occupies the northwest. An elongated island lies near the center, a compact island lies farther east, and broad vegetated land enters from the southeastern edge. Southwestern edge-connected land and a separate east–west island border the lower water area. The bottom-center cell is almost entirely open water. Medium-confidence measurements place the visible northern landmass at approximately 450 × 580 m and the central island at approximately 180 × 230 m.

High confidence: water is dark blue-green in true color and nearly black in false color. Land interiors are mottled green-brown in true color and predominantly red to crimson in false color, demonstrating substantial vegetation with appreciable near-infrared reflectance. Medium confidence: larger interiors contain woody cover, while pale shoreline fringes and internal patches include exposed or sparsely vegetated substrate. Rock, shallow water and mixed pixels cannot be separated consistently. No extensive bright sediment plume is evident, but water clarity is not established.

High confidence: no regular cultivated parcels or irrigation geometry is visible. Medium confidence: settlement, roads and active forestry management are not demonstrated. Vegetation appears broadly active with medium confidence, but species, canopy structure and ecological health remain unresolved. A temperate or boreal rocky setting is plausible with low confidence; the imagery provides insufficient evidence for a firm biome or climate assignment.

High confidence: coastal archipelago and island-rich lake interpretations both remain viable. The renderings cannot quantify moisture, turbidity, water depth or salinity, establish a specific season, or diagnose recent erosion, flooding, harvest or clearing. Native 10 m resolution limits small-feature identification, and display stretching and shoreline mixing complicate spectral interpretation.

#### Patch bcec7e27-2d53-4111-b902-8fca173ba248

- **w25** (23 words): High confidence: wooded land surrounds western and southeastern waters. Medium confidence: bright shoreline vegetation marks wetter ground. Land use and season remain unresolved.

- **w50** (48 words): High confidence: this patch contains wooded land, a broad western water body narrowing into a passage, and southeastern water. Medium-confidence estimates assign 23% to water and 73% to vegetation. Bright shoreline bands may indicate wet vegetation. Agriculture, settlement, exact season and active disturbance remain unconfirmed with high confidence.

- **w100** (89 words): High confidence: the patch is a heterogeneous wooded shoreline landscape with western open water narrowing into a sinuous passage and a smaller southeastern water area. Medium-confidence cover estimates are 23% water, 61% woody vegetation, 12% low vegetation and 4% sparse or exposed ground. Widespread false-color red indicates live foliage with high confidence; brighter marginal bands plausibly mark wet vegetation with medium confidence. Pale corridors and tiny reflective spots have uncertain origins. With high confidence, these images cannot establish crop types, forestry management, water quality, precise season or recent change.

- **w200** (186 words): High confidence: the 1.2 km patch shows a wooded aquatic landscape. Broad western water narrows into a sinuous passage, while a separate visible water area occupies the southeast. Their connectivity beyond the patch is unknown with high confidence. Medium-confidence visual estimates allocate 23% to water, 61% to woody vegetation, 12% to low vegetation and 4% to sparse or exposed ground.

High confidence: terrestrial texture is irregular and mottled, contrasting with smooth dark water. Bright green bands in true color become pink-red in false color, particularly across the lower center and along eastern margins. These bands plausibly represent wetter low vegetation with medium confidence. Their brightness does not independently establish superior plant health, with high confidence.

Low-confidence features include a possible diagonal track or exposed ridge in the northeast, possible small islets and tiny reflective patches that could be structures or natural surfaces. A temperate or boreal wooded setting is plausible with low confidence. With high confidence, no crop type, rotation, active harvest, irrigation, flooding or recent clearing can be established. Native 10 m resolution, display stretching and the single acquisition limit material identification and process inference.

- **w400** (350 words): High confidence: this 1.2 km square patch contains a heterogeneous wooded shoreline landscape. The western water body is broad at the top, narrows beside a wooded projection in the middle-left cell and continues southward as a curved passage. Another water area occupies the bottom-right cell and extends beyond the lower edge. Their external connectivity and full dimensions are unknown with high confidence. Medium-confidence visual cover estimates are 23% open water, 61% woody vegetation, 12% low vegetation and 4% sparse or exposed ground.

High confidence: the upper-center and upper-right cells have coarse, irregular vegetation texture. The middle cells contain scattered muted openings and a curved inland boundary. The bottom-center cell contains conspicuously bright, lobed vegetation bands extending toward the southeastern shore. Their position and spectral appearance support a wetland or drainage-margin interpretation with medium confidence, although a distinct flowing stream is not established.

High confidence: true color is dominated by dark green and olive land and nearly black or blue water. False color shows widespread red foliage, with stronger pink-red bands along portions of the lower-center and eastern margins. Live vegetation is therefore well supported. Differences in red brightness may reflect vegetation structure, density or illumination with medium confidence; plant health cannot be diagnosed from these differences alone with high confidence. Local cyan water margins could indicate shallows, bottom visibility, suspended material or mixed pixels with low confidence.

Low-confidence local interpretations include a narrow diagonal track or exposed ridge in the northeast, small possible islets and two tiny reflective patches near the middle-right cell. No settlement or forestry operation is securely identified. With high confidence, the images cannot establish agricultural crops, rotation, active irrigation, harvest, erosion, flooding or recent clearing. A temperate or boreal wooded lake setting is plausible with low confidence, but biome and climate are not diagnostic.

High confidence: native 10 m resolution prevents reliable identification of individual trees, small structures and narrow channels. Exposed rock, soil and sparse vegetation remain partly ambiguous with medium confidence. Display stretching limits quantitative spectral comparisons, and a single acquisition cannot reveal disturbance timing, seasonal water-level changes or management history, with high confidence.

#### Patch ffa692a8-0cb0-4c93-9fd4-d57a22d5294f

- **w25** (24 words): Probable snow-covered terrain alternates with mottled, possibly wooded patches (medium confidence). Strong red vegetation is absent (high confidence); land use remains unresolved (high confidence).

- **w50** (47 words): Probable snow cover forms bright openings among mottled, possibly wooded terrain (medium confidence). Dark curved margins separate patches (high confidence). Strong red vegetation is absent (high confidence), but vigor cannot be diagnosed. Roads, water and buildings are unresolved; season, biome and active processes remain uncertain (high confidence).

- **w100** (91 words): The patch likely depicts snowy open ground interspersed with darker, possibly wooded terrain (medium confidence). Apparent cover is estimated at 45% bright snow-like surfaces, 48% mottled terrain and 7% dark margins (low confidence in exact fractions). A mottled belt trends northwest–southeast, while large bright openings occupy northern and lower-central areas (high confidence). False color shows little strong red, limiting evidence for exposed vigorous vegetation (high confidence). Snow masking and display effects prevent health diagnosis. Roads, buildings, water, management and active processes are unresolved (high confidence). Cloud contamination remains possible (low confidence).

- **w200** (190 words): The leading interpretation is a snow-covered mosaic of open ground and darker, possibly wooded patches (medium confidence). Visual estimates assign 45% to bright white/cyan cover, 48% to mottled gray-blue terrain and 7% to dark margins; exact fractions have low confidence.

A connected mottled belt runs from the upper center through the middle toward the lower right (high confidence). Bright openings occur along the northern edge, across the middle left, and prominently in the lower center. Curved dark bands border several openings, including a long southern margin (high confidence). Their interpretation as vegetation edges or shadows has low confidence.

Both composites retain extensive white and cyan areas. Strong bright-red vegetation is absent (high confidence), but this cannot establish poor vegetation health because snow masking and rendering choices may suppress the signal. Woody cover is plausible from texture (medium confidence); crop identity, forestry management and settlement remain unresolved (high confidence).

Cold-season or persistent elevated snow is plausible (medium confidence), but the biome and climate are poorly constrained. No diagnostic evidence establishes flooding, irrigation, erosion, harvesting or recent clearing (high confidence). Snow/cloud ambiguity, saturation, mixed pixels and display upsampling limit interpretation (high confidence).

- **w400** (365 words): The patch most plausibly shows snow-covered open ground interspersed with mottled, possibly wooded terrain (medium confidence). Approximate apparent cover is 45% bright white/cyan surfaces, 48% mottled gray-blue terrain and 7% dark margins. These are visual estimates with low confidence in exact fractions, rather than measured land-cover proportions.

The spatial pattern is strongly heterogeneous (high confidence). In the upper left, a bright northern opening adjoins a granular terrain block bounded by a dark curved edge. The upper center contains mottled terrain continuing southeast through the middle into the lower right. The upper right is mainly bright white/cyan, with some darker terrain near the image edge. A broad bright corridor crosses the middle left, and a large irregular opening spans the lower-left and lower-center cells. Dark, scalloped margins trace parts of its southern boundary (high confidence for these visible patterns).

The largest bright openings and terrain blocks extend several hundred meters. Narrow dark bands have apparent widths of tens of meters, although their dimensions are sensitive to mixed pixels and display smoothing (medium confidence). They cannot confidently be assigned to roads, streams or other infrastructure. No buildings or open-water bodies are unambiguously resolved (high confidence).

True color is dominated by near-white brightness, cyan transitions and blue-gray texture. False color retains these patterns without conspicuous bright-red vegetation (high confidence). This supports limited evidence for exposed vigorous vegetation, but does not demonstrate stress or dormancy. Snow masking, canopy structure and display stretching could explain the weak red response (medium confidence). Moisture, wet snow, water turbidity, bare soil and built surfaces cannot be reliably diagnosed (high confidence).

Possible woody cover is suggested by granular texture and connected darker patches (medium confidence). Agricultural use, crop rotation, plantation management and settlement are unresolved (high confidence). Uneven snow exposure might reflect shading, canopy interception or ground-cover differences (low confidence); active melting, irrigation, flooding, erosion, harvest and recent clearing are not established (high confidence).

A seasonally snowy temperate, boreal or montane setting is plausible (low confidence), but snow persistence, acquisition season and climate cannot be selected from these views. Cloud contamination remains an alternative explanation for some brightness (low confidence). Saturation, uncertain spectral stretches, mixed pixels and upsampling constrain all finer interpretation (high confidence).

#### Patch 354d86b7-c2c2-4231-854d-767edc81e5bb

- **w25** (25 words): Apparently rugged terrain has brown surfaces, dark corridors, and scattered vegetation. Vegetation identification has high confidence; terrain-shadow interpretation has medium confidence. Land use remains unresolved.

- **w50** (47 words): Apparently rugged terrain contains brown exposed or sparsely vegetated surfaces, extensive dark corridors, and scattered green patches. Vegetation identification has high confidence because patches turn red in false color. Terrain-shadow interpretation has medium confidence, with water unresolved. No clear field system, settlement, or active management is identifiable.

- **w100** (86 words): The patch likely shows rugged, sparsely vegetated terrain with extensive topographic shadow, at medium confidence. Approximate display-area fractions are 55% brown ground or sparse vegetation, 35% darkness, and 10% distinct vegetation; these estimates have medium confidence. Vegetation identification has high confidence from green-to-red correspondence between composites. Upper bands trend northwest–southeast, while the principal dark corridor trends southwest–northeast, at medium confidence. Water cannot be excluded from that corridor. No clear agricultural parcels or settlement are resolved, at medium confidence. Season, climate, management, and active disturbance remain uncertain.

- **w200** (174 words): The patch most likely depicts rugged terrain with sparse vegetation and extensive terrain shadow, at medium confidence. Visual area estimates are approximately 55% brown exposed ground or sparse vegetation, 35% very dark surfaces, and 10% distinct vegetation, at medium confidence. Darkness conceals the underlying cover.

Vegetation identification has high confidence: localized green patches become pink or bright red in the near-infrared composite. Prominent patches occur in the upper center, middle left, and lower center. Their stronger response indicates greater green cover or leaf area relative to surrounding ground, at medium confidence; physiological health and moisture are not directly measured.

Upper elongated boundaries trend northwest–southeast, and a broad central dark corridor runs southwest–northeast, at medium confidence. A separate dark lobe occupies the lower right. Shadow is favored, but water remains possible within the central corridor.

No coherent field system, road network, or settlement is apparent, at medium confidence. A dry or seasonally dry setting is plausible at low confidence. Specific biome, crop type, forestry, irrigation, erosion activity, and flooding cannot be established from these renderings.

- **w400** (350 words): The 1.2 km patch most likely shows rugged, sparsely vegetated terrain with extensive topographic shadow, at medium confidence. Approximate visual fractions are 55% brown ground or sparse vegetation, 35% very dark surfaces, and 10% distinct green vegetation, at medium confidence. These describe visible appearance; shadow obscures an unknown mixture of underlying covers.

The upper row contains mottled brown surfaces, elongated dark recesses, and a broken vegetation strip across the upper center toward the right. The middle left contains a prominent irregular vegetation cluster. Smaller patches occur near the northern edge of the central dark corridor. An isolated rounded vegetation patch occupies the upper part of the bottom-center cell. These vegetation observations have high confidence.

Upper bright–dark bands trend approximately northwest–southeast, whereas the broad central dark corridor trends southwest–northeast, at medium confidence. That corridor spans much of the patch and is roughly 120–300 m wide, at medium confidence. A separate elongated dark lobe lies in the bottom right. Curved bands across the lower center may represent slope breaks or exposed strata, at low confidence. Their appearance does not establish active erosion or terracing.

Spectrally, green patches become pink or bright red in the near-infrared rendering, providing high-confidence evidence of vegetation. The muted brown and olive background likely combines exposed substrate with sparse or dry vegetation, at medium confidence. Stronger redness can reflect greater green cover or leaf area, at medium confidence, but does not independently measure vigor or moisture. The darkest zones remain dark in both composites. Terrain shadow is favored at medium confidence, although water cannot be excluded from the central corridor; turbidity therefore cannot be assessed.

No coherent agricultural parcel pattern, settlement, or road network is resolved, at medium confidence. Natural or lightly managed cover is plausible at medium confidence. A dry or seasonally dry environment is a low-confidence possibility, but season and biome remain indeterminate. Crop identity, rotation, forestry, irrigation, harvest, flooding, clearing, and ecological degradation cannot be established. At 10 m, small structures, individual trees, narrow channels, and tracks may escape detection; upsampling and display contrast further limit feature counts, dimensions, and spectral interpretation, at high confidence.

#### Patch 3f747420-913a-44d0-8cc8-49088e9b77c8

- **w25** (21 words): High confidence: fields and settlement form a heterogeneous mosaic. Medium confidence: contrasting vegetation responses reflect different canopy cover or growth stages.

- **w50** (44 words): High confidence: large agricultural parcels dominate the west and north, while buildings, roads, and gardens concentrate eastward and southward. Strong false-color red marks live vegetation. Medium confidence: tan, muted fields have sparse canopy or dry residue. Crop identities, management timing, and moisture remain unresolved.

- **w100** (98 words): High confidence: this 1.2 km patch contains a heterogeneous agricultural–settlement mosaic. Large fields occupy the west and north; buildings, roads, and gardens concentrate in the east and lower center. Northern parcels and the broad west-central field show strong near-infrared vegetation responses. Medium confidence: tan fields contain exposed soil, dry residues, or sparse vegetation, indicating differing management or growth stages. Woody vegetation appears mainly along boundaries and within settlement. No sizeable open water is confidently identified. High confidence: these renderings cannot establish crop species, rotation, soil moisture, or active irrigation, and individual buildings are often mixed at 10 m.

- **w200** (183 words): High confidence: large agricultural fields and a substantial settlement form a heterogeneous mosaic across this 1.2 km patch. Open parcels dominate the west and north, while buildings, roads, and gardens occupy much of the east and lower center. Medium-confidence visual estimates allocate 40% to vegetated fields and grassland, 23% to low-cover fields, 20% to predominantly impervious surfaces, 10% to settlement vegetation, and 7% to woody cover and margins.

High confidence: northern fields and the broad west-central parcel show strong false-color red responses, indicating substantial live vegetation. Tan parcels in the upper-left and southwest have much weaker responses. Medium confidence: these differences reflect canopy cover, phenology, and management, potentially including exposed soil or dry residues. A north–south road organizes the eastern settlement, with diagonal connecting corridors visible at medium confidence.

Medium confidence: woody vegetation is mostly linear or scattered, and an elongated enclosure occupies the southeast. No sizeable open water is confidently identified. A temperate setting is plausible at low confidence. High confidence: crop species, rotation, quantitative moisture, and recent change cannot be determined; small buildings, hedges, and roads are affected by mixed pixels.

- **w400** (361 words): High confidence: this 1.2 km Sentinel-2 patch depicts a heterogeneous agricultural–settlement mosaic. Large fields occupy the west and north, while fine-grained buildings, roads, gardens, and small plots concentrate in the east and lower center. Medium-confidence visual cover estimates are 40% vegetated agricultural fields and open grassland, 23% bare or weakly vegetated fields, 20% predominantly impervious surfaces, 10% settlement vegetation, and 7% woody vegetation and dark margins. These estimates are approximate because many settlement pixels contain multiple covers.

High confidence: the broad west-central field is relatively smooth and strongly red in the near-infrared composite. Northern fields also show strong red responses. The upper-left tan field and southwestern parcels are more muted, consistent at medium confidence with exposed soil, dry residues, sparse crops, or senescent vegetation. Differences in displayed red intensity reflect several factors and cannot uniquely diagnose plant health or moisture.

High confidence: agricultural parcels have sharp angular boundaries, contrasting with the mottled settlement. Medium confidence: a main north–south road crosses the eastern third, diagonal lanes connect the center and southwest, and several dark field margins contain hedges or other woody vegetation. A conspicuous bright roof or hardstanding complex lies near the upper central transition into settlement. An elongated, comparatively weakly vegetated enclosure occupies the southeast. Individual building counts and narrow corridor widths cannot be resolved reliably.

Medium confidence: the scene contains managed grassland or crops alongside low-cover parcels at different growth or management stages. Specific crop types, harvest status, and rotation are unresolved. Woody cover appears concentrated along boundaries and within gardens, without extensive forest blocks. Vegetated margins may provide ecological connections, but biodiversity and habitat quality cannot be assessed. A temperate agricultural setting is plausible only at low confidence.

No sizeable open-water body is confidently visible, and water turbidity cannot be assessed. Medium confidence: there is no clear evidence of flooding, active irrigation, major erosion, or extensive recent clearing. Absence of visible evidence does not exclude small or concealed activity. High confidence: display stretching, native 10 m resolution, and mixed pixels prevent quantitative moisture or reflectance interpretation. A small saturated multicolor spot near the lower center has an uncertain origin. Single-date imagery cannot establish management timing, change rates, or climate.

#### Patch 2c3e8d2b-ff94-421d-b1fe-773fe8f2164d

- **w25** (24 words): High confidence: geometric agricultural fields dominate. Medium confidence: contrasting vegetation and pale parcels reflect different management stages, with woody cover concentrated in the northwest.

- **w50** (47 words): High confidence: this agricultural mosaic contains sharply bounded green and pale fields. Medium confidence: cover is approximately 52% vegetated agriculture, 35% bare or sparse agriculture, and 12% woody vegetation; possible infrastructure occupies 1% with low confidence. Crop identities, management timing, and climate remain unresolved with high confidence.

- **w100** (88 words): High confidence: geometric agricultural parcels dominate the patch, including a large western green field, northern strips, and an eastern L-shaped vegetated parcel. Medium confidence: cover is approximately 52% vegetated agriculture, 35% pale agricultural surfaces, and 12% woody vegetation; possible infrastructure occupies 1% with low confidence. Strong false-color red confirms vegetation with high confidence. Medium confidence: pale cyan fields represent sparse cover, exposed soil, or residue, while northwestern irregular vegetation includes woody belts. High confidence: crop species, rotation, moisture status, and active irrigation cannot be determined from these renderings.

- **w200** (190 words): High confidence: the patch is an agricultural mosaic with straight parcel boundaries and contrasting field states. Medium confidence: estimated cover is 52% vegetated agriculture, 35% bare or sparsely vegetated agricultural surfaces, and 12% woody vegetation. Possible infrastructure accounts for 1% with low confidence.

High confidence: a large green field dominates the middle-left, narrow vegetated strips meet the northern edge, and an L-shaped vegetated footprint occupies the central-right. Pale agricultural ground separates these parcels. Medium confidence: irregular northwestern vegetation forms woodland patches and field-edge belts. Curved and diagonal lines in the south may be access tracks or field margins. Small southwestern bright objects may be structures, with low confidence.

High confidence: strong red in false color identifies substantial vegetation, including fields that appear dark in true color. Medium confidence: cyan and muted parcels indicate reduced green cover or different developmental stages; harvest and fallow are possible explanations. High confidence: red brightness alone does not rank plant health, and neither soil moisture nor irrigation is established. No open water is confidently identifiable. High confidence: crop identities, rotation, exact season, and management history remain unresolved; a temperate setting is only a low-confidence possibility.

- **w400** (358 words): High confidence: this 1.2 km square patch is dominated by agriculture, with smooth geometric fields and abrupt spectral boundaries. Medium confidence: approximate cover is 52% vegetated agriculture, 35% bare, harvested, or sparsely vegetated agricultural surfaces, and 12% woody vegetation. Possible tracks and built surfaces occupy about 1%, with low confidence. These are visual estimates rather than measured classifications.

High confidence: the middle-left contains a broad green field approximately 500–550 m east–west by 380–420 m north–south. Its false-color response is vivid pink-red, and a narrow darker inset lies near its eastern margin. Northern parcels form elongated strips. A central-right vegetated footprint is L-shaped, approximately 250 by 500 m overall, although its interpretation as one parcel has medium confidence. A broad pale north–south strip separates major vegetated areas. Southern parcels include dark vegetation and a lighter, tapered field reaching the bottom edge.

Medium confidence: the northwest contains connected woody patches and belts, distinguished by irregular outlines and coarser texture. Curving lines in the southwest and a diagonal southeastern line may be tracks or field margins. Low confidence: bright objects near the southwest corner could represent small rural structures; individual buildings are unresolved.

High confidence: strong false-color red identifies substantial vegetation, including dark-green fields in true color. Medium confidence: pale tan surfaces rendered cyan contain less green cover and may comprise soil, residue, or sparse crops. Adjacent spectral contrasts plausibly reflect differing crop stages or management. High confidence: display brightness cannot establish a ranking of vegetation health, and these composites do not directly measure moisture or diagnose irrigation. No open water is confidently identifiable, preventing turbidity assessment.

High confidence: specific crop types, rotation, active harvest, flooding, erosion, and clearing history are not established. Medium confidence: woody boundaries may provide habitat connections, but biodiversity and ecological condition remain unresolved. A temperate agricultural setting is plausible with low confidence; the imagery does not establish climate or potential natural biome.

High confidence: native 10 m resolution, mixed pixels, unknown display scaling, and a single observation limit interpretation. Upsampling does not reveal additional ground detail. Small structures, narrow drainage features, machinery, and individual trees require finer observations; seasonal and management conclusions require a time series.

#### Patch 5f9d5e6f-e221-43ef-b73f-7e4f895df271

- **w25** (23 words): Probably wooded terrain contains irregular sparse or exposed openings (medium confidence). Widespread red indicates leafy vegetation (high confidence); season and management remain unresolved.

- **w50** (47 words): The patch is probably dominated by woody vegetation, with irregular pale openings concentrated toward the upper center and middle right (medium confidence). Extensive false-color red supports leafy cover (high confidence). Dark diagonal belts may reflect terrain and canopy variation (medium confidence). Season, management and climate remain unresolved.

- **w100** (91 words): The 1.2 km patch probably contains about 65% dense woody cover, 28% sparse vegetation mixed with ground, and 7% exposed soil or rock (medium confidence). Irregular openings are strongest in the upper center and central to eastern cells (high confidence). Dark southwest–northeast belts may reflect canopy and terrain illumination (medium confidence). Extensive false-color red confirms substantial leafy vegetation, while pale openings have weaker responses (high confidence). No settlement or large water body is confidently identified. Crop identity, forestry activity, season, moisture and climate cannot be established from these renderings (high confidence).

- **w200** (192 words): This 1.2 km scene is probably a wooded landscape with irregular openings and uneven terrain (medium confidence). Visual cover estimates are 65% dense woody vegetation, 28% sparse vegetation mixed with exposed ground, and 7% predominantly bare soil or rock (medium confidence).

The strongest pale clearing complex occupies the upper center, with curved margins and vegetated islands (high confidence). Additional fragmented openings cross the middle center and middle right. Lower and western cells retain more continuous vegetation (medium confidence). Several dark belts trend approximately southwest–northeast; canopy differences and terrain illumination are plausible explanations (medium confidence).

True color is dominated by greens with tan and gray openings. False color shows extensive red vegetation and weaker red responses over pale ground (high confidence). This supports differences in leafy cover, but does not independently measure health or moisture (high confidence).

No clear settlement, large water body or regular crop-field system is identified (high confidence). Natural sparse ground and managed clearings remain difficult to distinguish (low confidence). Season, crop type, forestry history, active erosion, flooding and specific biome or climate remain unresolved. Ten-meter pixels and display interpolation limit identification of small structures, tracks and channels (high confidence).

- **w400** (364 words): The paired renderings show a predominantly vegetated 1.2 km landscape with connected woody-looking cover and irregular openings (medium confidence). Approximate visual fractions are 65% dense woody vegetation, 28% sparse vegetation mixed with exposed ground, and 7% predominantly bare soil or rock (medium confidence). These are broad estimates, not measured classifications.

The upper-center cell contains the most conspicuous pale complex, roughly 250–350 m wide and 300–400 m long, with lobed margins and vegetated internal islands (medium confidence). The middle center contains another broad opening, while the middle right has elongated, fragmented pale patches (high confidence). Western and lower cells generally retain more continuous cover, punctuated by small openings (medium confidence). Boundaries are mostly irregular or diffuse, and there is no clear rectangular field network (high confidence).

Dark belts cross the western and lower portions approximately southwest–northeast (medium confidence). Their appearance is consistent with dense or shaded vegetation, potentially influenced by relief, although elevation and drainage cannot be established directly (medium confidence). Narrow pale segments and isolated bright flecks cannot be identified confidently as roads or buildings (high confidence).

True color is predominantly dark to medium green, with tan and olive-gray openings. False color shows extensive vivid red through dark crimson vegetation, whereas many pale openings become gray-green or muted reddish gray (high confidence). This supports substantial leafy cover and spatial differences in vegetation density (high confidence). Red brightness alone does not establish physiological health, and moisture cannot be isolated from illumination and cover effects (high confidence). No confidently recognized water surface permits a turbidity assessment.

Natural sparse ground, exposed substrate and managed clearings are plausible explanations for the openings, but their relative contributions remain uncertain (low confidence). Forestry, crop rotation, irrigation, harvest, recent clearing, erosion and flooding cannot be confirmed (high confidence). No settlement cluster or large water body is confidently resolved. A wooded terrestrial setting is supported, but season, specific biome and climate are undetermined (high confidence).

At 10 m resolution, mixed pixels obscure small structures, individual trees, narrow tracks and channels. Upsampling adds no ground detail, and display colors do not support quantitative reflectance measurements. Temporal imagery and additional measurements would be required to establish change, management history or ecological condition (high confidence).

#### Patch 0bf20967-86c2-4e27-96df-5220f4108873

- **w25** (24 words): High confidence: woodland dominates a rural mosaic with eastern agricultural parcels. Medium confidence: pale fields have sparse vegetation; red false-color areas indicate active foliage.

- **w50** (48 words): High confidence: this patch is a woodland-dominated rural mosaic, with angular agricultural parcels in the northeast and southeast. Medium confidence: woody cover occupies roughly 72%, and pale fields contain soil or residue. Strong false-color red indicates vegetation. Crop identity, season and recent management remain unresolved with high confidence.

- **w100** (90 words): High confidence: connected woodland dominates the western and central patch, while angular agricultural parcels occupy the northeast and southeast. Medium-confidence estimates allocate 72% to woody cover, 10% to open vegetation and 16% to sparse vegetation or exposed agricultural surfaces; corridors account for approximately 2% with low confidence. Strong false-color red supports extensive living vegetation with high confidence. Tan-to-cyan fields probably contain soil or residue, and an eastern linear feature is probably a road, both with medium confidence. Crop identity, moisture status and recent management cannot be established with high confidence.

- **w200** (183 words): High confidence: the 1.2 km patch contains a connected woodland matrix interrupted by small openings and bordered by agricultural parcels in the northeast and southeast. Medium-confidence cover estimates are 72% woody vegetation, 10% open herbaceous vegetation and 16% exposed or sparsely vegetated agricultural surfaces. Narrow corridors and disturbed surfaces contribute approximately 2%, with low confidence.

High confidence: wooded areas have mottled green true-color tones and predominantly red false-color responses, whereas pale agricultural parcels appear tan to cyan. Medium confidence: this contrast indicates substantial living foliage beside fields containing soil, dry residue or sparse vegetation. A smoother green parcel in the middle-right could be grassland, pasture or crop.

Medium confidence: a probable road follows the eastern margin, and another corridor touches the northern edge. A small rectangular clearing occurs in the middle-left; larger irregular openings occur farther south. High confidence: no unambiguous open water is resolved. Medium confidence: no settlement cluster is evident.

High confidence: these renderings cannot determine crop species, rotation, quantitative moisture, health or recent change. A temperate woodland–farmland setting is plausible with low confidence, but climate and season remain weakly constrained.

- **w400** (346 words): High confidence: this 1.2 km square is a rural woodland–agriculture mosaic. Connected woody vegetation dominates the west and center, while larger angular agricultural parcels occupy the northeast and southeast. Medium-confidence visual estimates assign 72% to woodland or scrub, 10% to open herbaceous vegetation and 16% to exposed or sparsely vegetated agricultural surfaces. An additional 2% comprises narrow corridors and disturbed surfaces with low confidence.

High confidence: the western and central cells have irregular, mottled vegetation textures. Their internal transitions are generally diffuse, whereas eastern field boundaries are sharper and more geometric. The northeastern field complex contains repeated approximately north–south bands, interpreted as managed strips with medium confidence. The southeastern complex contains several adjoining compartments and dark vegetated dividers. Medium confidence: the smoother green parcel in the middle-right may support grass, pasture or a crop.

High confidence: a small quadrilateral clearing interrupts the middle-left woodland. Its estimated dimensions are 70–90 by 50–70 m with medium confidence. Several larger irregular southern openings are approximately 100–230 m across with low confidence. Medium confidence: a probable road or track runs north-northeast–south-southwest near the eastern margin, and another possible corridor follows part of the northern edge. Small buildings cannot be reliably identified.

High confidence: green surfaces generally become red in the near-infrared composite, supporting extensive living vegetation. Medium confidence: much of this cover is actively foliated. Pale tan fields become cyan or gray-green, suggesting soil, dry residue or sparse vegetation with medium confidence. Differences in woodland redness may reflect canopy structure, composition or illumination with medium confidence; they do not uniquely diagnose health or moisture. High confidence: no unambiguous open-water surface is resolved, preventing turbidity assessment.

High confidence: agricultural use is supported by parcel geometry and repeated bands. Medium confidence: pale fields could be harvested, cultivated or fallow, but none of these states is uniquely established. High confidence: crop identity, rotation, irrigation, recent logging, erosion and flooding cannot be confirmed. A temperate woodland–farmland setting is plausible with low confidence. Native 10 m pixels, mixed surfaces, display interpolation and unknown color stretches limit fine-feature interpretation and quantitative spectral conclusions with high confidence.

#### Patch 1bb3844f-c810-4173-b6f5-2654009e3d90

- **w25** (22 words): High confidence: woodland surrounds irregular openings and pale corridors. Medium confidence: some openings are managed grassland; brown patches have less green vegetation.

- **w50** (47 words): High confidence: this is a wooded mosaic with central openings and narrow pale corridors. Medium confidence: woody cover occupies about 68%, with grassland and brown clearings comprising most remaining land. False color supports widespread active vegetation. Low confidence: individual openings represent pasture, fallow land, or recent disturbance.

- **w100** (98 words): High confidence: woodland forms a connected matrix around an irregular north–south band of openings. Medium confidence: approximately 68% is woody cover, 21% herbaceous vegetation, 9% brown or sparsely vegetated ground, and 2% pale corridors. A large brown clearing lies left of center, while smoother green and tan openings occupy the central column. High confidence: vegetation shows extensive red response in false color. Medium confidence: the southern pale line is a road or track; the eastern line is ambiguous. Low confidence: management includes pasture or forestry. Specific crops, climate, moisture, and disturbance timing cannot be established with high confidence.

- **w200** (185 words): High confidence: the patch contains a predominantly wooded landscape interrupted by central openings and narrow pale corridors. Medium confidence: visual cover estimates are 68% woody vegetation, 21% herbaceous vegetation, 9% brown or sparsely vegetated openings, and 2% exposed linear surfaces.

High confidence: a large brown clearing lies left of center, adjoining smoother central vegetation. An elongated pale opening extends through the bottom-center cell. Woodland dominates the right side, where several blocks have angular boundaries and contrasting tones. Medium confidence: a curving southern corridor is a road or track. The eastern north–south corridor could be a track or drainage feature.

High confidence: broad red responses in false color indicate extensive active vegetation. Medium confidence: weak red response in brown and pale openings reflects reduced green cover, senescence, or soil exposure. Low confidence: these openings represent pasture, fallow ground, or disturbance, and some woodland parcels reflect forestry.

High confidence: crop identity, rotation, moisture, biome, and active management cannot be established from this single image pair. Medium confidence: no settlement or broad water body is evident. High confidence: unresolved narrow features, mixed pixels, and display stretching limit interpretation.

- **w400** (348 words): High confidence: the 1.2 km patch is a woodland-dominated mosaic with irregular open parcels and narrow pale corridors. Medium confidence: estimated cover is 68% woody vegetation, 21% herbaceous vegetation, 9% brown or sparsely vegetated ground, and 2% exposed linear surfaces. These are visual estimates rather than measured classifications.

High confidence: woodland forms a broadly connected matrix around a discontinuous central band of openings. The top-left cell includes a partial brown opening at the image boundary. A larger brown clearing spans the central-left area, with angular margins and scattered dark inclusions. Medium confidence: this clearing measures roughly 300 by 400 m, and some inclusions are retained trees or shrubs. Smoother vegetated openings occupy the top-center and middle-center cells, continuing into an elongated pale parcel in bottom-center.

High confidence: the right-hand cells are mainly wooded, with darker blocks and locally angular edges. A pale line winds approximately north–south through their western portions, and a bent connector crosses the central openings. Medium confidence: a separate curving line across bottom-left and bottom-center is a road or track. Low confidence: the eastern corridor is also a track; a drainage feature or combined corridor remains plausible.

High confidence: true color contrasts dark green woodland with muted green, tan, and brown openings. False color shows extensive bright and deep red vegetation, while the large brown clearing becomes green-gray and central grassy areas are generally muted pink. Medium confidence: these differences reflect vegetation density, phenology, and soil contribution. They do not establish moisture or stress independently.

Medium confidence: some openings are managed grassland, pasture, or fallow parcels. Low confidence: woodland tonal blocks reflect forestry, or the brown clearing represents recent harvest or clearing. High confidence: crop species, rotation, disturbance timing, irrigation, flooding, and erosion cannot be confirmed. Low confidence: a mesic wooded setting is plausible, but biome and climate remain unresolved.

Medium confidence: no recognizable settlement, broad open-water body, or conspicuous cloud is evident. High confidence: original 10 m resolution limits identification of buildings, individual trees, ditch widths, and surface materials. Mixed pixels, shadow, display stretches, and the absence of temporal observations constrain all finer interpretation.

#### Patch 9931d1e8-7c27-4411-bb57-415b841dbb3c

- **w25** (24 words): High confidence: agricultural fields adjoin extensive woodland. Medium confidence: contrasting vegetation stages and pale parcels indicate varied management; a winding corridor likely provides access.

- **w50** (48 words): High confidence: this patch contains a heterogeneous agricultural–woodland mosaic, with fields concentrated northwest and woodland east and south. Medium confidence: woodland occupies about half the area, and pale parcels reflect sparse vegetation or agricultural disturbance. A winding access corridor is probable. Crop identities and seasonal timing remain uncertain.

- **w100** (88 words): High confidence: smooth agricultural parcels dominate the upper-left portion, while coarse woodland occupies much of the east and south. Medium-confidence visual estimates assign 49% to woody cover, 35% to vegetated agriculture, 13% to sparse or bare parcels and 3% to routes or disturbed surfaces. High confidence: vivid false-color red indicates abundant active vegetation. Medium confidence: neighboring tan/cyan fields represent contrasting management or phenological stages. A pale winding corridor probably provides access. High confidence: crop species, rotation, irrigation, ecological condition and precise season cannot be established from these images.

- **w200** (188 words): High confidence: the scene is a heterogeneous rural mosaic of agricultural parcels and extensive woodland. Fields are concentrated in the top row and middle-left; woodland dominates the right side and lower portion. Medium-confidence cover estimates are 49% woody vegetation, 35% vegetated agriculture, 13% bare or sparsely vegetated parcels and 3% routes or disturbed surfaces.

High confidence: fields have smooth interiors and angular boundaries, whereas woodland is coarsely mottled with irregular edges. Medium confidence: a narrow pale corridor bends through the center and continues toward winding lower-left access segments. Its broad central green margin becomes pink-red, favoring vegetation over open water.

High confidence: vivid red agricultural vegetation contrasts with cyan or muted pale parcels. Medium confidence: this indicates different vegetation densities or management stages, potentially including cutting or harvest. Darker woodland responses also reflect canopy structure and shading.

High confidence: no unambiguous water body is resolved, and the images cannot establish irrigation, crop rotation, species or quantitative vegetation health. Medium confidence: no substantial settlement is apparent. Low confidence: a temperate setting is plausible. Sizes, counts and cover fractions remain visual estimates limited by 10 m pixels and display processing.

- **w400** (344 words): High confidence: this 1.2 km square patch contains a rural agricultural–woodland mosaic. Smooth agricultural parcels dominate the upper-left and upper-center portions, while coarse woody cover occupies much of the east and south. Medium-confidence visual estimates allocate 49% to woodland or dense woody vegetation, 35% to vegetated agricultural land, 13% to bare or sparsely vegetated parcels and 3% to routes and small disturbed surfaces.

High confidence: the top row contains elongated fields with sharp tonal contrasts and predominantly angular boundaries. The middle-left transitions from open vegetation into woodland. The middle-center contains curved field bands beside a narrow pale corridor, while the middle-right is predominantly wooded. The bottom row contains extensive woody cover, small openings and winding pale segments. Medium confidence: these segments form an access network, although continuity under or beside canopy is uncertain. A wider green ribbon along the central curve has a pink-red false-color response, favoring a vegetated margin over open water.

High confidence: many agricultural parcels are vivid red in the near-infrared composite, demonstrating substantial active vegetation. Woodland is more variably maroon and red. Medium confidence: canopy structure and shading contribute to that difference. Tan or cream fields become cyan, gray-green or muted pink, consistent with sparse green cover, exposed soil, residue or senescent vegetation. Medium confidence: adjacent contrasting parcels reflect differing management or phenological stages. Specific harvest activity remains unconfirmed.

Medium confidence: woodland forms a substantial habitat matrix with numerous agricultural edges. A lower-central irregular opening could contain mixed herbaceous and woody regrowth, but a clearing or forestry interpretation has low confidence. Medium confidence: no substantial settlement is apparent; compact bright patches near lower-left routes could include small structures but are unresolved. Low confidence: a temperate agricultural–woodland setting is plausible, though the appearance does not uniquely identify a biome or climate.

High confidence: the renderings cannot determine crop species, rotations, irrigation, soil moisture, biodiversity or quantitative vegetation health. No unambiguous open-water body supports a turbidity assessment. At 10 m, narrow routes, hedges, buildings and canopy gaps suffer mixed-pixel ambiguity. Counts and dimensions are approximate, and display upsampling adds no spatial information.
