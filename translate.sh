export OPENAI_API_KEY="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"
LOG_NAME=sample_korea
LANGUAGE=ko
python3 image_translate.py indir=$(pwd)/input_images outdir=$(pwd)/output/$LOG_NAME font=$(pwd)/font/NotoSerifCJKsc-VF.ttf lang=$LANGUAGE