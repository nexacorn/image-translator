
## SETTING
```
conda create -n translate python=3.8 -y
sh setting.sh
```
## RUN
translate.sh
```
export OPENAI_API_KEY=""
input_folder=input_images
LOG_NAME=sample_vietnam_font_cluster
LANGUAGE=vi
python3 image_translate.py indir=$(pwd)/$input_folder outdir=$(pwd)/output/$LOG_NAME font=$(pwd)/font/NotoSerifCJKsc-VF.ttf lang=$LANGUAGE
```
1. input 폴더 설정
    `translate.sh` 폴더 내부에 `input_folder`에 input으로 사용할 이미지 폴더를 적는다. `input_folder` 내부에 상품별 폴더를 두어, 그 폴더 내부에 상세페이지 이미지들을 두어야함
2. LOG NAME : output 폴더 내의 결과 폴더 이름
3. LANGUAGE : vi / ko 지원
4. RUN shell script
`sh translate.sh`
