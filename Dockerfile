FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy requirements（context = monorepo 根，路径带服务前缀）
COPY rag-finance-qa/requirements.txt .
RUN pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple/ -r requirements.txt

# Copy app 自身代码
COPY rag-finance-qa/ .

# Copy 共享配置层：配合下方 PYTHONPATH，使 from shared.config 生效
COPY shared/ ./shared/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 5679

CMD ["python", "src/web_app.py"]
