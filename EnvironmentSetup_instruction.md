Step by Step Environment Setup 

1. Create a virtual environment with python > 10.0
2. go to root folder and  pip install -e .
3. If you have gpu do this : 
         pip install "torch==2.6.*" --index-url https://download.pytorch.org/whl/cu118
   Otherwise do this : 
         pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
