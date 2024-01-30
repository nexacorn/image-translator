pip3 install -r requirements.txt
pip3 install paddlepaddle
pip3 install "paddleocr>=2.0.1" 
# GPU When CUDA Version 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CPU
# conda install pytorch torchvision torchaudio -c pytorch -y
pip install pytorch-lightning==1.2.9
pip install Pillow==9.5.0