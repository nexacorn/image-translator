from PIL import Image, ImageDraw, ImageFont
import numpy as np
from app.module.utils import is_similar_color

def expand_bbox(image, bbox, max_expand=5):

    if isinstance(image, Image.Image):
        image = np.array(image)

    x1, y1, x2, y2 = bbox
    height, width = image.shape[:2]
    last_valid_bbox = bbox
    # print('new box')
    for expand in range(1, max_expand + 1):
        # Check bounds and update last valid bbox if within bounds

        # print(expand)
        if (
            x1 - expand < 0
            or y1 - expand < 0
            or x2 + expand >= width
            or y2 + expand >= height
        ):
            return last_valid_bbox
        last_valid_bbox = (x1 - expand, y1 - expand, x2 + expand, y2 + expand)

        # Get surrounding pixels
        top_row = image[y1 - expand, x1 - expand : x2 + expand]
        bottom_row = image[y2 + expand - 1, x1 - expand : x2 + expand]
        left_col = image[y1 - expand : y2 + expand, x1 - expand]
        right_col = image[y1 - expand : y2 + expand, x2 + expand - 1]

        # Check if surrounding pixels are similar
        reference_color = image[
            y1, x1
        ]  # Reference color from the top-left corner of the bbox
        if not (
            np.all([is_similar_color(pixel, reference_color) for pixel in top_row])
            and np.all(
                [is_similar_color(pixel, reference_color) for pixel in bottom_row]
            )
            and np.all([is_similar_color(pixel, reference_color) for pixel in left_col])
            and np.all(
                [is_similar_color(pixel, reference_color) for pixel in right_col]
            )
        ):
            return last_valid_bbox

    return last_valid_bbox