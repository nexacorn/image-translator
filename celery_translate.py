from celery import Celery, current_task
from module.translate_module import process_image, upload_image_url
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from lama_cleaner.model_manager import ModelManager
from paddleocr import PaddleOCR
from urllib.parse import urlparse, unquote


global_lama = None
global_ocr = None
model: ModelManager = None

celery_app = Celery('translate', broker='pyamqp://guest@localhost//')
celery_app.conf.update(
    result_backend='rpc://',
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Europe/London',
    enable_utc=True,
)

def load_models():
    global global_lama, global_ocr
    print(global_lama, global_ocr)
    if global_lama is None or global_ocr is None:
        global_lama = ModelManager(name='lama', device='cuda')
        global_ocr = PaddleOCR(lang="ch", show_log=False, use_gpu=True, ocr_version="PP-OCRv3")
    return global_lama, global_ocr

def extract_filename_from_url(url):
    parsed_url = urlparse(url)
    filename = parsed_url.path.split('/')[-1]
    decoded_filename = unquote(filename)
    return decoded_filename

model, ocr = load_models()

@celery_app.task(bind=True)
def translate_image(self, image_urls):
    task_id = self.request.id
    
    processed_urls = []
    for url in image_urls:
        file_name = extract_filename_from_url(url)
        
        response = requests.get(url)
        # image = Image.open(BytesIO(response.content))
        if response.status_code == 200:
            # BytesIO를 사용하여 이미지 데이터를 바이너리 스트림으로 변환합니다.
            image_data = BytesIO(response.content)

            # PIL을 사용하여 이미지를 엽니다.
            image = Image.open(image_data)
        else:
            print(f"Failed to retrieve image. Status code: {response.status_code}")
            return

        font = '../font/NanumGothicBold.ttf'
        to_lang = 'ko'
        
        processed_image = process_image(image, font, to_lang, model, ocr, file_name, task_id)
        processed_url = upload_image_url(processed_image)
        
        processed_urls.append(processed_url)
        
    return processed_urls


