import numpy as np

_IDS = (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 19)
_TERRESTRIAL = (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14)
_NAMES = (
    "Brightness", "NDVI mean", "NDVI standard deviation",
    "Vegetated share", "NDWI mean", "Water share", "NDMI mean",
    "SWIR1 mean", "SWIR2 mean", "NIR texture",
)


def _nd(a, b):
    # Reflectance equivalent of the documented 1e-6 DN denominator floor.
    return (a - b) / np.maximum(a + b, 1e-10)


def label(desc, x):
    """Return conservative spectral evidence, not validated CORINE labels.

    Inputs: descriptors in documented units; x contains reflectance * 10000.
    No I/O, training, geographic information, or assumed class frequencies.

    'no' indicates strong contrary evidence at acquisition time. Flooding,
    seasonality, subpixel cover and CORINE boundaries can defeat these rules.
    Exclusion requires every pixel to satisfy the relevant spectral test;
    a dominant surface alone does not exclude another class in a mixed patch.

    No reliable 'yes' rule is available for these particular classes:
    spectra cannot establish water geography, management, canopy height,
    forest composition, or vegetation succession. Those distinctions abstain.

    Distribution checks:
    - 59 bank entries had Water share > .95, NDWI mean > .2, NDVI mean < .1.
      Their SWIR1/2 mean p95 values were .01527/.01391 reflectance.
      Water ceilings below are approximately twice those values.
    - Bank-wide SWIR1/2 mean p25 values were .09908/.05200 reflectance.
    - Water-like sample pixels had NIR p99=.0278; .08 allows some turbidity.
    These thresholds are heuristics, not statistically calibrated boundaries.

    Tested against all 50 samples, using matching bank descriptors:
    60 'no', 690 'abstain', zero 'yes' across 750 decisions.
    Also checked missing descriptors, NaNs, zeros and incorrect input units.
    Execution checks do not establish classification accuracy.
    """
    out = dict.fromkeys(_IDS, "abstain")
    a = np.asarray(x, dtype=np.float64)
    if a.shape != (12, 120, 120):
        raise ValueError("x must have shape (12, 120, 120)")
    if not np.isfinite(a).all() or np.any(a < 0):
        return out
    try:
        supplied = np.array([desc[k] for k in _NAMES], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return out
    if supplied.shape != (10,) or not np.isfinite(supplied).all():
        return out

    # Apply the supplied scale exactly; do not infer an additional DN offset.
    r = a / 10000.0
    blue, green, red = r[1], r[2], r[3]
    re1, re3, nir, nir_narrow = r[4], r[6], r[7], r[8]
    swir1, swir2 = r[10], r[11]
    visible = (blue + green + red) / 3.0
    ndvi = _nd(nir, red)
    ndwi = _nd(green, nir)
    ndmi = _nd(nir_narrow, swir1)
    mndwi = _nd(green, swir1)

    # Check descriptor correspondence and units; means cannot identify land use.
    measured = np.array([
        visible.mean() * 10000, ndvi.mean(), ndvi.std(),
        (ndvi > .5).mean(), ndwi.mean(), (ndwi > 0).mean(),
        ndmi.mean(), swir1.mean() * 10000, swir2.mean() * 10000,
        nir.std() * 10000,
    ])
    tolerance = np.array([2, .002, .002, .002, .002, .002, .002, 2, 2, 2])
    if np.any(np.abs(supplied - measured) > tolerance):
        return out

    # Conservative quality vetoes; these cannot replace a cloud/scene mask.
    obscured = (
        (visible < .003)
        | ((blue > .20) & (visible > .20) & (ndvi < .25))
        | ((green > .15) & (nir > .10) & (mndwi > .40))
    )
    if obscured.any() or np.any(r > 1.2):
        return out

    # Water absorbs strongly in NIR/SWIR relative to visible green.
    water = (
        (ndwi > .20) & (mndwi > .35) & (ndvi < .15)
        & (green > .01) & (nir < .08) & (nir_narrow < .08)
        & (swir1 < .031) & (swir2 < .028)
    )

    # Chlorophyll absorption and a red-edge rise support vegetation,
    # but do not separate crops, pasture, shrubs and tree canopies.
    vegetation = (
        (ndvi > .40) & (_nd(re3, re1) > .15)
        & (nir > .10) & (swir1 > .05)
    )

    # Substantial SWIR reflection contradicts an exposed open-water surface.
    nonwater_surface = (
        (swir1 > .099) & (swir2 > .052)
        & (mndwi < -.20) & (nir > .08)
    )

    # Upsampled pixels are not independent observations. No significance
    # calculation or texture-derived claim about management is made.
    if water.all():
        for class_id in _TERRESTRIAL:
            out[class_id] = "no"
    if (vegetation | nonwater_surface).all():
        out[18] = "no"
        out[19] = "no"

    # Wetlands can contain vegetation, exposed peat or open water.
    return out
