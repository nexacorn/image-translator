from flask import Flask, request, jsonify
from app.translate_module import process_image, upload_image_url
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from lama_cleaner.model_manager import ModelManager
from paddleocr import PaddleOCR
from urllib.parse import urlparse, unquote
from dotenv import load_dotenv
import os
import argparse

load_dotenv()

app = Flask(__name__)  # Flask 애플리케이션 인스턴스 생성

global_lama = None
global_ocr = None
model: ModelManager = None


def load_models(device):
    global global_lama, global_ocr
    if global_lama is None or global_ocr is None:
        global_lama = ModelManager(name="lama", device=device)
        global_ocr = PaddleOCR(
            lang="ch",
            show_log=False,
            use_gpu=False if device == "cpu" else True,
            ocr_version="PP-OCRv3",
        )
    return global_lama, global_ocr


def extract_filename_from_url(url):
    parsed_url = urlparse(url)
    filename = parsed_url.path.split("/")[-1]
    decoded_filename = unquote(filename)
    return decoded_filename


@app.route("/process-images", methods=["POST"])
def process_images():
    image_urls = request.json.get("urls", [])
    to_lang = request.json.get("language", "")

    processed_urls = []
    openai_api_key = os.getenv("OPENAI_KEY")

    font = ""

    if "vi" in to_lang.lower():
        to_lang = "vi"
        font = "./font/NotoSerifCJKsc-VF.ttf"
    elif "ko" in to_lang.lower():
        to_lang = "ko"
        font = "./font/NanumGothicBold.ttf"
    else:
        return "The language requested by the user is not supported"

    for url in image_urls:
        file_name = extract_filename_from_url(url)

        response = requests.get(url, stream=True, headers={
            "Content-Type": "application/json", 
            "Accept": "application/json", 
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
        })

        if response.status_code == 200:
            image_data = BytesIO(response.content)
            image = Image.open(image_data)
        else:
            print(f"Failed to retrieve image. Status code: {response.status_code}")
            return [""]

        try:
            processed_url = process_image(
                image, font, to_lang, model, ocr, file_name, openai_api_key
            )
        except Exception as e:
            print(f"An error occurred: {e}")

        processed_urls.append(processed_url)

    return processed_urls


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="cpu", help='Device to use: "cuda" or "cpu"'
    )
    args = parser.parse_args()

    # 모델을 로드할 때 명령줄 인자를 사용합니다.
    model, ocr = load_models(args.device)

    app.run(host="0.0.0.0", port=8088, threaded=False, debug=False)
