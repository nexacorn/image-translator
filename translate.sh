cd lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd) && export OPENAI_API_KEY="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"

LOG_NAME=auto_expand_bbox_vietnam
mkdir /home/ubuntu/OCR/lama/ocr_lama_images/$LOG_NAME
python3 bin/predict_ocr.py model.path=$(pwd)/big-lama indir=$(pwd)/images ocr_outdir=$(pwd)/ocr_lama_images/$LOG_NAME outdir=$(pwd)/output/$LOG_NAME font=$(pwd)/font/NotoSerifCJKsc-VF.ttf