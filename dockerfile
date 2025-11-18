FROM python:3.10

WORKDIR /app

# Install system dependencies needed for image/video (if required)
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

COPY requirements.txt .

# Install PyTorch CPU version
RUN pip install --no-cache-dir torch==2.2.0+cpu torchvision==0.17.0+cpu torchaudio==2.2.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]