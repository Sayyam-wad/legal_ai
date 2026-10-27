# CPU image for the web app. Build the indexes first (or mount them):
#   docker build -t legal-ai .
#   docker run -p 5000:5000 -v $PWD/faiss_indexes:/app/faiss_indexes legal-ai
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 PORT=5000 HF_HOME=/app/.cache/huggingface

WORKDIR /app
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt gunicorn

COPY . .
RUN python preprocess.py > /dev/null

EXPOSE 5000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')"
CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "1", "--threads", "4", "--timeout", "300", "app:app"]
