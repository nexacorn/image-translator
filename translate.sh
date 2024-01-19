export OPENAI_API_KEY=""
input_folder=input_images
LOG_NAME=sample_vietnam_font_cluster
LANGUAGE=vi
python3 image_translate.py indir=$(pwd)/$input_folder outdir=$(pwd)/output/$LOG_NAME font=$(pwd)/font/NotoSerifCJKsc-VF.ttf lang=$LANGUAGE