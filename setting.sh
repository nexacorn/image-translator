pip3 install -r requirements.txt
python -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install "paddleocr>=2.0.1" 
# GPU
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
# CPU
# conda install pytorch torchvision torchaudio -c pytorch -y
pip install pytorch-lightning==1.2.9
pip install Pillow==9.5.0