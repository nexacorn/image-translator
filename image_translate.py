import logging
import os
import cv2
import hydra
import numpy as np
import io
import tqdm
import json
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from PIL import Image, ImageDraw, ImageFont
import textwrap
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
from sklearn.cluster import DBSCAN
import os
import glob
from openai import OpenAI
import re
import random
from sklearn.cluster import KMeans


from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, SDSampler
from lama_cleaner.helper import (
    load_img,
    numpy_to_bytes,
    resize_max_size,
    pil_to_bytes,
)

LOGGER = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.CRITICAL)

openai_api_key = os.getenv("OPENAI_API_KEY")
ocr_lang = "ch"
image_extensions = ["png", "jpg", "jpeg", "bmp", "gif"]

model: ModelManager = None

config = Config(
    ldm_steps=25,
    # ldm_sampler=form["ldmSampler"],
    hd_strategy=HDStrategy.ORIGINAL,
    # zits_wireframe=form["zitsWireframe"],
    hd_strategy_crop_margin=128,
    hd_strategy_crop_trigger_size=128,
    hd_strategy_resize_limit=128
    # prompt=form["prompt"],
    # negative_prompt=form["negativePrompt"],
    # use_croper=form["useCroper"],
    # croper_x=form["croperX"],
    # croper_y=form["croperY"],
    # croper_height=form["croperHeight"],
    # croper_width=form["croperWidth"],
    # sd_scale=form["sdScale"],
    # sd_mask_blur=form["sdMaskBlur"],
    # sd_strength=form["sdStrength"],
    # sd_steps=form["sdSteps"],
    # sd_guidance_scale=form["sdGuidanceScale"],
    # sd_sampler=form["sdSampler"],
    # sd_seed=form["sdSeed"],
    # sd_match_histograms=form["sdMatchHistograms"],
    # cv2_flag=form["cv2Flag"],
    # cv2_radius=5,
    # paint_by_example_steps=form["paintByExampleSteps"],
    # paint_by_example_guidance_scale=form["paintByExampleGuidanceScale"],
    # paint_by_example_mask_blur=form["paintByExampleMaskBlur"],
    # paint_by_example_seed=form["paintByExampleSeed"],
    # paint_by_example_match_histograms=form["paintByExampleMatchHistograms"],
    # paint_by_example_example_image=paint_by_example_example_image,
    # p2p_steps=form["p2pSteps"],
    # p2p_image_guidance_scale=form["p2pImageGuidanceScale"],
    # p2p_guidance_scale=form["p2pGuidanceScale"],
    # controlnet_conditioning_scale=form["controlnet_conditioning_scale"],
    # controlnet_method=form["controlnet_method"],
)


def load_model(name, device):
    model = ModelManager(name=name, device=device)
    return model


def is_similar_color(color1, color2, threshold=0):
    """Check if two colors are similar within a threshold."""
    print(color1, color2)
    return np.all(np.abs(color1 - color2) <= threshold)


def expand_bbox(image, bbox, max_expand=5):
    """
    Expand the bounding box in the image and check if the surrounding pixels are of similar color.
    If expansion goes beyond image boundaries, use the last valid bounding box.

    Args:
    image (numpy.ndarray): The image array.
    bbox (tuple): The bounding box (x1, y1, x2, y2).
    max_expand (int): Maximum pixels to expand.

    Returns:
    tuple: Expanded bounding box (x1, y1, x2, y2) if expansion is possible; otherwise, original bbox.
    """
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


def calculate_center(box):
    # box format: [x, y, w, h]
    center_x = box[0] + box[2] / 2
    center_y = box[1] + box[3] / 2
    return np.array([center_x, center_y])


def cluster_boxes(boxes, eps):
    # Calculate centers
    centers = np.array([calculate_center(box) for box in boxes])

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=1).fit(centers)
    labels = clustering.labels_

    # Group boxes by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label in clusters:
            clusters[label].append(i)
        else:
            clusters[label] = [i]

    return list(clusters.values())


def contains_english(s):
    return bool(re.fullmatch("[a-zA-Z0-9\s!@#$%^&*()_+\-=[\]{ };'\":,.<>/?\\|￥]+", s))


def translate_llm(original_text_len, orginal_text, lang="ko"):
    client = OpenAI(api_key=openai_api_key)
    if "ko" in lang:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "배열로 들어온 총 "
                    + str(original_text_len)
                    + "개의 유저 메시지는 패션상품의 광고문구야 무조건 빠짐없이 모든 메시지들을 번역해야돼. 즉, "
                    + str(original_text_len)
                    + '개의 번역결과가 나와야해. 번역이 잘 안되는 것들은 그대로 순서에 맞게 반환해줘. 이것들을 패션 상품에 어울리는 한글로, 가능한 개조식으로 번역해줘. 배열 형식과 순서는 유지한채 json 형식으로, 한글로 번역해서 반환해줘. return example : { translated : ["번역 결과","번역 결과","번역 결과","번역 결과"] }',
                },
                {"role": "user", "content": orginal_text},
            ],
        )
        return completion
    elif "vi" in lang:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Trong tổng số "
                    + str(original_text_len)
                    + " tin nhắn người dùng nhập từ mảng là slogan quảng cáo cho sản phẩm thời trang, cần phải dịch tất cả mà không bỏ sót bất kỳ tin nhắn nào. Nghĩa là, phải có "
                    + str(original_text_len)
                    + ' kết quả dịch. Những tin nhắn khó dịch thì trả lại nguyên trạng theo đúng thứ tự. Hãy dịch những điều này sang tiếng Việt phù hợp với sản phẩm thời trang, càng sáng tạo càng tốt. Giữ nguyên định dạng và thứ tự của mảng dưới dạng json, trả lại bằng tiếng Việt. Ví dụ trả về: { translated : ["kết quả dịch", "kết quả dịch", "kết quả dịch", "kết quả dịch"] }',
                },
                {"role": "user", "content": orginal_text},
            ],
        )
        return completion

def adjust_bbox(image_shape, bbox):
    h, w = image_shape[:2]
    x1, y1, bw, bh = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x1 + bw), min(h, y1 + bh)
    return x1, y1, x2, y2

def extract_border_colors(image_path, bbox, expansion=1, n_clusters=1):
    # 이미지 로드
    image = Image.open(image_path)
    image = np.array(image)

    x1, y1, x2, y2 = adjust_bbox(image.shape, bbox)
    x1, y1, x2, y2 = x1 - expansion, y1 - expansion, x2 + expansion, y2 + expansion
    x1, y1, x2, y2 = adjust_bbox(image.shape, (x1, y1, x2-x1, y2-y1))

    # 경계선 색상 추출
    top_border = image[y1, x1:x2+1]
    bottom_border = image[y2, x1:x2+1]
    left_border = image[y1:y2+1, x1]
    right_border = image[y1:y2+1, x2]

    border_colors = np.vstack((top_border, bottom_border, left_border, right_border))

    # K-means 클러스터링
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(border_colors)
    return kmeans.cluster_centers_

def extract_bbox_colors(image_path, bbox, n_clusters=2):
    # 이미지 로드
    image = Image.open(image_path)
    image = np.array(image)

    x1, y1, x2, y2 = adjust_bbox(image.shape, bbox)

    # 바운딩 박스 내부 픽셀 추출
    bbox_pixels = image[y1:y2+1, x1:x2+1].reshape(-1, 3)

    # K-means 클러스터링
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(bbox_pixels)
    return kmeans.cluster_centers_

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


def image_translate(predict_config):
    key_folders = [f.path for f in os.scandir(predict_config.indir) if f.is_dir()]
    for key_folder in key_folders:
        key = os.path.basename(key_folder)
        images = []
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(key_folder, f"*.{ext}")))

        for input_image in tqdm.tqdm(images):
            img = Image.open(input_image)
            ocr = PaddleOCR(
                lang=ocr_lang, show_log=False, use_gpu=True, ocr_version="PP-OCRv3"
            )
            img_path = input_image

            result = ocr.ocr(img_path, cls=False)

            if not result or result == [None]:
                cur_outdir = os.path.join(predict_config.outdir, key)
                if not os.path.exists(cur_outdir):
                    os.makedirs(cur_outdir)

                file_name = os.path.basename(input_image)
                base_name = os.path.splitext(file_name)[0]
                img.save(os.path.join(cur_outdir, f"{base_name}_result.jpg"))
                continue

            font = predict_config.font

            boxes = [line[0] for line in result[0]]

            # OCR 결과 이미지 저장
            # txts = [line[1][0] for line in result[0]]
            # scores = [line[1][1] for line in result[0]]
            # result_np = draw_ocr(img, boxes, txts, scores, font_path=font)
            # result_img = Image.fromarray(result_np)

            # file_name = os.path.basename(input_image)
            # base_name = os.path.splitext(file_name)[0]
            # result_img.save(
            #     os.path.join(predict_config.ocr_outdir, f"{base_name}_ocr.png")
            # )

            boxes = []

            for i, r in enumerate(result[0]):
                if not contains_english(result[0][i][1][0]):
                    x1, y1 = r[0][0]
                    x2, y2 = r[0][2]
                    w, h = x2 - x1, y2 - y1

                    text, conf = r[1]

                    boxes.append([int(x1), int(y1), int(w), int(h), text, conf, i])

            # 원본 이미지 크기로 검은색 마스크 이미지 생성
            mask_img = Image.new("RGB", img.size, "black")
            mask_draw = ImageDraw.Draw(mask_img)

            # 각 박스 위치에 하얀색 사각형 그리기
            mp = 2
            width, height = mask_img.size
            for box in boxes:
                x1, y1, w, h, _, _, _ = box
                x2 = x1 + w
                y2 = y1 + h
                x1 = max(x1 - mp, 0)
                y1 = max(y1 - mp, 0)
                x2 = min(x2 + mp, width)
                y2 = min(y2 + mp, height)
                expanded_bbox = expand_bbox(mask_img, (x1, y1, x2, y2))
                # expanded_bbox = (x1,y1,x2,y2)
                mask_draw.rectangle(expanded_bbox, fill="white")

            # 마스크 이미지 저장
            # file_name = os.path.basename(input_image)
            # base_name = os.path.splitext(file_name)[0]

            # img.save(os.path.join(predict_config.ocr_outdir, f"{base_name}.png"))
            # mask_img.save(
            #     os.path.join(predict_config.ocr_outdir, f"{base_name}_mask.png")
            # )

            lama_model = load_model("lama", "cuda")

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            byte_im = buf.getvalue()

            img, alpha_channel, exif_infos = load_img(byte_im, return_exif=True)

            buf.flush()
            buf.seek(0)
            mask_img.save(buf, format="PNG")
            byte_mask = buf.getvalue()

            mask_img, _ = load_img(byte_mask, gray=True)

            mask_img = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)[1]

            config.sd_seed = random.randint(1, 999999999)
            config.paint_by_example_seed = random.randint(1, 999999999)

            interpolation = cv2.INTER_CUBIC
            size_limit = max(img.shape)

            img = resize_max_size(
                img, size_limit=size_limit, interpolation=interpolation
            )
            mask_img = resize_max_size(
                mask_img, size_limit=size_limit, interpolation=interpolation
            )

            inpainted_result = lama_model(img, mask_img, config)
            inpainted_result = cv2.cvtColor(
                inpainted_result.astype(np.uint8), cv2.COLOR_BGR2RGB
            )
            if alpha_channel is not None:
                if alpha_channel.shape[:2] != inpainted_result.shape[:2]:
                    alpha_channel = cv2.resize(
                        alpha_channel,
                        dsize=(inpainted_result.shape[1], inpainted_result.shape[0]),
                    )
                inpainted_result = np.concatenate(
                    (inpainted_result, alpha_channel[:, :, np.newaxis]), axis=-1
                )
            inpainted_result = Image.fromarray(inpainted_result)

            ocr_result = []
            for box in boxes:
                ocr_result.append([box])

            final_result = []

            for sub_result in ocr_result:
                x1 = sub_result[0][0]
                y1 = sub_result[0][1]
                x2 = sub_result[-1][0] + sub_result[-1][2]
                y2 = sub_result[-1][1] + sub_result[-1][3]

                w, h = x2 - x1, y2 - y1

                text = ""

                for r in sub_result:
                    text += r[4] + " "

                text = text.strip()

                final_result.append([x1, y1, w, h, text])

            original_texts = [item[4] for item in final_result]
            original_texts_string = json.dumps(
                [item[4].replace("，", "").strip() for item in final_result],
                ensure_ascii=False,
            )

            completion = translate_llm(
                len(original_texts), original_texts_string, predict_config.lang
            )

            result = json.loads(completion.choices[0].message.content)
            translated = next(iter(result.values()))

            if len(original_texts) != len(translated):
                print("ERROR : length error")
                continue

            for i, r in enumerate(final_result):
                final_result[i].append(translated[i].upper())

            translated_result = inpainted_result.copy()
            draw = ImageDraw.Draw(translated_result)

            for r in final_result:
                x1, y1, w, h, text_ko, translated_text = r
                # print(x1, y1, w, h, text_en)
                border_color = extract_border_colors(input_image, (x1, y1, w, h))
                bbox_colors = extract_bbox_colors(input_image,(x1,y1,w,h))
                font_color = find_most_different_color(border_color, bbox_colors)
                wrapped_text = [translated_text]

                font_size = 17
                font_ = ImageFont.truetype(font, font_size)
                ascent, descent = font_.getmetrics()
                (width, height), (offset_x, offset_y) = font_.font.getsize(
                    translated_text
                )

                while width < w and (ascent - offset_y) < h:
                    font_size += 1
                    font_ = ImageFont.truetype(font, font_size)
                    ascent, descent = font_.getmetrics()
                    (width, height), (offset_x, offset_y) = font_.font.getsize(
                        translated_text
                    )
                font_size -= 1
                font_ = ImageFont.truetype(font, font_size)
                line_height = font_.getsize("hg")[1] * 1.5

                # 각 텍스트 줄에 대해
                for line in wrapped_text:
                    # 텍스트 줄의 너비 계산
                    text_width, _ = font_.getsize(line)

                    text_x = x1 + (w - text_width) / 2
                    if text_x <= 0:
                        text_x = 0
                    text_position = (text_x, y1)

                    draw.text(
                        text_position,
                        line,
                        fill=font_color,
                        font=font_
                    )
                    y1 += line_height

            cur_outdir = os.path.join(predict_config.outdir, key)
            if not os.path.exists(cur_outdir):
                os.makedirs(cur_outdir)

            file_name = os.path.basename(input_image)
            base_name = os.path.splitext(file_name)[0]
            inpainted_result.save(
                os.path.join(cur_outdir, f"{base_name}_inpainted.png")
            )
            translated_result.save(os.path.join(cur_outdir, f"{base_name}_result.jpg"))


@hydra.main(config_path="./config", config_name="ocr.yaml")
def main(predict_config: OmegaConf):
    image_translate(predict_config)


if __name__ == "__main__":
    main()
