import logging

import numpy as np
from mac.types import InferenceData


def process_data(input_data: InferenceData) -> InferenceData:
    """Dummy processing step."""
    logging.info(f"Got keys: {input_data.keys()}")
    image_3d = np.zeros((1, 2052, 2456, 3))
    image_2d = np.zeros((1, 2052, 2456))

    return {
        "image": input_data['image'].astype(np.uint16),
        "virtual_stain": image_3d.astype(np.uint8),
        "mask_overlay": image_3d.astype(np.uint8),
        "mask": image_2d.astype(np.uint16),
    }
