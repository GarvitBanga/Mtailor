FROM python:3.10-bookworm
RUN apt-get update && apt-get install -y dumb-init
RUN update-ca-certificates

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN python convert_to_onnx.py
RUN rm -rf pytorch_model_weights.pth
EXPOSE 8000
CMD ["dumb-init", "--", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 