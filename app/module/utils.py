import numpy as np
from PIL import Image
import io
import boto3


def to_png(jpg_image):
    # BytesIO 객체를 사용하여 메모리 내에서 이미지를 처리합니다.
    png_image_io = io.BytesIO()

    # 이미지를 PNG 형식으로 BytesIO 객체에 저장합니다.
    jpg_image.save(png_image_io, format="PNG")

    # BytesIO 객체에서 PNG 형식의 이미지를 읽어 Image 객체로 변환합니다.
    png_image_io.seek(0)
    png_image = Image.open(png_image_io)

    return png_image


def upload_image_url(image, file_name):
    s3 = boto3.client("s3")
    bucket_name = "okit"

    in_mem_file = io.BytesIO()
    image.save(in_mem_file, format="PNG")
    in_mem_file.seek(0)

    upload_response = s3.upload_fileobj(
        in_mem_file,
        bucket_name,
        "translated_" + file_name,
        ExtraArgs={"ContentType": "image/png"},  # 이미지 형식에 맞게 변경
    )

    image_url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": "translated_" + file_name},
        ExpiresIn=3600,
    )  # 유효시간 1시간

    return image_url


def is_similar_color(color1, color2, threshold=0):
    return np.all(np.abs(color1 - color2) <= threshold)


def calculate_center(box):
    # box format: [x, y, w, h]
    center_x = box[0] + box[2] / 2
    center_y = box[1] + box[3] / 2
    return np.array([center_x, center_y])


def normalize_color(color):
    return [component / 255.0 for component in color]


def color_similarity(color1, color2):
    normalized_color1 = normalize_color(color1)
    normalized_color2 = normalize_color(color2)
    return np.linalg.norm(np.array(normalized_color1) - np.array(normalized_color2))


def adjust_bbox(image_shape, bbox):
    h, w = image_shape[:2]
    x1, y1, bw, bh = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x1 + bw), min(h, y1 + bh)
    return x1, y1, x2, y2


def find_most_different_color(border_color, bbox_colors):
    max_distance = 0
    most_different_color = None

    for color in bbox_colors:
        # RGB 공간에서 유클리드 거리 계산
        distance = np.linalg.norm(border_color - color)
        if distance > max_distance:
            max_distance = distance
            most_different_color = color
    return tuple(most_different_color.astype(int))
