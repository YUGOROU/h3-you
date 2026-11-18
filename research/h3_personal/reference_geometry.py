"""Versioned H3 reference geometry. Existing frozen caches remain legacy.

Pinned Diffusers resolve_reference_image_size returns (HEIGHT, WIDTH).
Corrected preparation must use fresh reference AND text components; it cannot be
mixed with the frozen legacy reference latent or silently applied to715cache.
"""
LEGACY = 'legacy-wh-v1'
CORRECTED = 'correct-hw-v2'


def prepare_rgb_reference(image, resolver, prepare, *, policy):
    if policy not in (LEGACY,CORRECTED):
        raise ValueError('explicit versioned reference geometry policy required')
    image=image.convert('RGB')
    resolved=resolver(image.width,image.height)
    if len(resolved)!=2 or any(type(n) is not int or n<=0 for n in resolved):
        raise ValueError('positive H/W resolver output required')
    if policy==CORRECTED:
        height,width=resolved
    else:
        width,height=resolved  # Exact historical behavior for immutable legacy use.
    return prepare(image,height,width)
