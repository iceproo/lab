FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3

RUN echo "Current directory:" && pwd && ls -al

WORKDIR /workspace
COPY requirements_yolonas.txt requirements_yolonas.txt
# Install Jupyter and other packages



RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install matplotlib