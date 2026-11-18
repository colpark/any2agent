import numpy as np
from scipy import ndimage

_CLASS_IDS = tuple(range(1, 20))


def _nd(a, b):
    return (a - b) / np.maximum(a + b, 1e-6)


def label(x):
    """Return conservative, abstention-heavy spectral exclusions.

    x: (12, 120, 120), reflectance * 10000, in the specified band order.
    'no' for water means no spectrally resolved open water in the patch;
    it does not establish absence from a CORINE reference polygon.
    These rules deliberately make no positive CORINE assignments.
    """
    x = np.asarray(x)
    if x.shape != (12, 120, 120):
        raise ValueError('Expected shape (12, 120, 120).')
    if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise TypeError('Expected real numeric reflectance DNs.')

    out = {i: 'abstain' for i in _CLASS_IDS}
    if not np.isfinite(x).all():
        return out
    r = x.astype(np.float64) / 10000.0
    if np.any(r < 0):
        return out

    # Preserve high values for quality checks; do not clip reflectance.
    # B01 and B09 are coarse atmospheric bands, not texture measurements.
    blue, green, red = r[1], r[2], r[3]
    nir, narrow_nir = r[7], r[8]
    swir1, swir2 = r[10], r[11]
    visible = (blue + green + red) / 3.0
    ndvi = _nd(nir, red)
    ndwi = _nd(green, nir)
    mndwi = _nd(green, swir1)
    whiteness = (
        np.maximum.reduce([blue, green, red])
        - np.minimum.reduce([blue, green, red])
    ) / np.maximum(visible, 1e-6)

    # Suspicion masks, not a replacement for an SCL/cloud mask.
    # Dark water is intentionally also ambiguous under the shadow guard.
    cloud = (visible > 0.20) & (whiteness < 0.30)
    snow = (green > 0.12) & (mndwi > 0.40)
    dark = (visible < 0.025) & (nir < 0.05)
    suspect = cloud | snow | dark | np.any(r > 1.2, axis=0)
    if suspect.any():
        return out

    # Strong vegetation or SWIR-bright, clearly non-water surfaces.
    # Require vegetation-index agreement with the narrow-NIR band.
    vegetation = (
        (ndvi > 0.45) & (_nd(narrow_nir, red) > 0.45)
        & (nir > 0.15) & (swir1 > 0.06)
    )
    dry_surface = (
        (swir1 > 0.18) & (swir2 > 0.12) & (nir > 0.22)
        & (ndwi < -0.25) & (mndwi < -0.30)
    )
    nonwater = vegetation | dry_surface

    # A 30 m neighborhood statistic flags strong local spectral variation.
    # Resampling does not create independent 10 m SWIR observations.
    mean = ndimage.uniform_filter(ndwi, size=3, mode='reflect')
    variance = np.maximum(
        ndimage.uniform_filter(ndwi * ndwi, size=3, mode='reflect')
        - mean * mean, 0.0
    )
    uncertain_edge = np.sqrt(variance) > 0.12

    # No minimum-area cutoff: any unresolved pixel retains abstention.
    # Water color and contact with patch edges cannot settle inland/marine.
    if np.all(nonwater) and not uncertain_edge.any():
        out[18] = 'no'
        out[19] = 'no'
    return out
