# ARROW decode HTTP service (TI-ONE and WeChat 云托管).
#
# The process listens on 0.0.0.0:${PORT:-8501}. In the WeChat console set
# 监听端口 to 8501 (the form default is 80). On TI-ONE set 端口 to 8501.
# Do not use 8500 (gRPC) or the reserved ports 8502-8510, 6006, 9092.
#
# The image is self-contained (code + ARROW_clean_pro.png). Do not mount a
# model directory over /opt/arrow, and do not rely on /data/model — TI-ONE
# overwrites /data/model when a CFS/COS model source is selected.
#
# This file is byte-identical in three places:
#   Dockerfile
#   services/decode_api/Dockerfile
#   wxcloudrun/Dockerfile
# wxcloudrun/ holds a copy of the sources this file COPYs, so either build
# context works. CI checks the copies match.
#
# Context = repository root (WeChat 目标目录 left empty, Dockerfile name Dockerfile):
#   docker build -t arrow-decode-api .
#   docker build -f services/decode_api/Dockerfile -t arrow-decode-api .
# Context = wxcloudrun/ (WeChat 目标目录 = wxcloudrun, Dockerfile name Dockerfile):
#   docker build -t arrow-decode-api wxcloudrun
#
# Run:
#   docker run --rm -p 8501:8501 arrow-decode-api

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/opt/arrow \
    PORT=8501 \
    DECODE_WARMUP=1

WORKDIR /opt/arrow

# opencv-python-headless still needs the system glib/gomp loaders on slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY services/decode_api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY arrow_decode.py arrow_wm.py arrow_inner.py ARROW_clean_pro.png /opt/arrow/
COPY services/__init__.py /opt/arrow/services/__init__.py
COPY services/decode_api /opt/arrow/services/decode_api

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /opt/arrow
USER appuser

EXPOSE 8501

# start-period covers SIFT template load. /health is intentionally unauthenticated.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8501'), timeout=4)"

# 进程内解码由一把锁串行（OpenCV matcher 不做重入保证），所以单 worker 的
# 吞吐上限约 1/解码耗时。要多核并行就设 WEB_CONCURRENCY=N：每个 worker 进程
# 各自持有完整解码器，内存约为单进程的 N 倍。默认 1 与原行为一致。
CMD ["sh", "-c", "exec python -m uvicorn services.decode_api.app:app --host 0.0.0.0 --port ${PORT:-8501} --workers ${WEB_CONCURRENCY:-1}"]
