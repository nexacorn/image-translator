from flask import Flask, request, jsonify
from celery_translate import translate_image
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)
app = Flask(__name__)  # Flask 애플리케이션 인스턴스 생성

@app.route('/process-images', methods=['POST'])
def process_images():
    print(request.json)
    image_urls = request.json.get('urls', [])
    print(image_urls)
    task = translate_image.delay(image_urls)
    return jsonify({"task_id": task.id})