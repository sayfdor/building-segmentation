FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN sed -i 's/opencv-python$/opencv-python-headless/' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

COPY src/      ./src/
COPY configs/  ./configs/
COPY train.py  .
COPY evaluate.py .
COPY visualize.py .
COPY vectorize.py .
COPY app.py .

CMD ["python", "train.py", "--help"]