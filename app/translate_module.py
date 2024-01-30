import logging
import cv2
import numpy as np
import io
import json
from torch.utils.data._utils.collate import default_collate
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import boto3

from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy
from lama_cleaner.helper import (
    load_img,
    resize_max_size,
)

from app.module.masking import expand_bbox
from app.module.utils import find_most_different_color, to_png, upload_image_url
from app.module.translate_llm import contains_english, translate_llm
from app.module.clustering import cluster_boxes_by_text_height, cluster_colors, cluster_boxes_by_text_y, align_text_boxes, extract_border_colors, extract_bbox_colors

LOGGER = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.CRITICAL)

ocr_lang = "ch"
global_lama = None
global_ocr = None

model: ModelManager = None

s3 = boto3.client('s3')

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


def process_image(input_image, font, to_lang, model, ocr, file_name, openai_api_key):
    img = input_image
    
    img_format = img.format

    if img_format != 'PNG':
        img = to_png(img)
    
    result = ocr.ocr(np.asarray(img), cls=False)
    
    if not result or result == [None]:    
        return upload_image_url(img, file_name)
    font = font

    boxes = [line[0] for line in result[0]]

    boxes = []

    for i, r in enumerate(result[0]):
        if not contains_english(result[0][i][1][0]):
            x1, y1 = r[0][0]
            x2, y2 = r[0][2]
            w, h = x2 - x1, y2 - y1

            text, conf = r[1]

            boxes.append([int(x1), int(y1), int(w), int(h), text, conf, i])

    if not boxes or len(boxes) < 1:
        return upload_image_url(img, file_name)

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

    lama_model = model

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

    img = resize_max_size(img, size_limit=size_limit, interpolation=interpolation)
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
    boxes = cluster_boxes_by_text_height(boxes, 40)
    boxes = cluster_boxes_by_text_y(boxes, 10)
    boxes = align_text_boxes(boxes, input_image)
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
        len(original_texts), original_texts_string,openai_api_key, to_lang
    )

    result = json.loads(completion.choices[0].message.content)
    translated = next(iter(result.values()))

    if len(original_texts) != len(translated):
        print("ERROR : length error")
        return

    for i, r in enumerate(final_result):
        final_result[i].append(translated[i].upper())

    translated_result = inpainted_result.copy()
    draw = ImageDraw.Draw(translated_result)

    for r in final_result:
        x1, y1, w, h, text_ko, translated_text = r
        border_color = extract_border_colors(input_image, (x1, y1, w, h))
        bbox_colors = extract_bbox_colors(input_image, (x1, y1, w, h))
        font_color = find_most_different_color(border_color, bbox_colors)
        r.append(font_color)

    final_result = cluster_colors(final_result, 0.3)

    for r in final_result:
        x1, y1, w, h, original_text, translated_text, font_color = r
        wrapped_text = [translated_text]

        font_size = 17
        font_ = ImageFont.truetype(font, font_size)
        ascent, descent = font_.getmetrics()
        (width, height), (offset_x, offset_y) = font_.font.getsize(translated_text)

        while width < w and (ascent - offset_y) < h:
            font_size += 1
            font_ = ImageFont.truetype(font, font_size)
            ascent, descent = font_.getmetrics()
            (width, height), (offset_x, offset_y) = font_.font.getsize(translated_text)
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

            draw.text(text_position, line, fill=font_color, font=font_)
            y1 += line_height


    if not isinstance(translated_result, Image.Image):
        translated_result = Image.fromarray(np.uint8(translated_result))
        
    return upload_image_url(translated_result, file_name)
