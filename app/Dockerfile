FROM python:3.10.10-slim-bullseye

WORKDIR /app
RUN mkdir -p /app/classified
COPY ./classified /app/classified
COPY requirements.txt /app/requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
COPY . .