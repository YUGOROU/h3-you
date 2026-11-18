"""Corrected-only producer, isolated from byte-frozen legacy cache producer."""
from research.h3_personal.component_encode import PinnedBackend
from research.h3_personal.reference_geometry import CORRECTED, prepare_rgb_reference


class CorrectedBackend(PinnedBackend):
    def image(self, unit, *, geometry_policy=CORRECTED):
        if geometry_policy != CORRECTED:
            raise ValueError('continuation producer requires corrected reference geometry')
        from PIL import Image
        from diffusers.modular_pipelines.minimax_h3.packing_ref2va import (
            prepare_reference_image, resolve_reference_image_size,
        )
        with Image.open(unit['sources']['image']['path']) as original:
            return prepare_rgb_reference(original, resolve_reference_image_size,
                                         prepare_reference_image, policy=geometry_policy)
