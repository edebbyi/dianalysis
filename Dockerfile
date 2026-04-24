FROM python:3.11-slim
ARG EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG TORCH_VERSION=2.5.1+cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DIANALYSIS_CONFIG=configs/base.toml
ENV DIANALYSIS_PROFILE=
ENV HF_HOME=/root/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
ENV DIANALYSIS_EMBED_MODEL=${EMBED_MODEL}

WORKDIR /app

RUN apt-get update && \
    apt-get install -y make && \
    rm -rf /var/lib/apt/lists/*
ARG REQUIREMENTS_FILE=requirements-docker.txt
ARG PRELOAD_EMBED_MODEL=1
COPY requirements.txt requirements-docker.txt ./
COPY setup.py .
COPY Makefile .
COPY train.py .
COPY experiments ./experiments
COPY dianalysis ./dianalysis
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} "torch==${TORCH_VERSION}" && \
    pip install --no-cache-dir -r ${REQUIREMENTS_FILE}
RUN if [ "${PRELOAD_EMBED_MODEL}" = "1" ]; then \
      python -c "import os; from sentence_transformers import SentenceTransformer; model_name = os.getenv('DIANALYSIS_EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'); SentenceTransformer(model_name); print(f'Preloaded embedding model: {model_name}')" \
    ; fi

# Copy only runtime files needed by the Streamlit app.
COPY app.py .
COPY .streamlit ./.streamlit
COPY artifacts ./artifacts
COPY data ./data
COPY configs ./configs

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
