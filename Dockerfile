FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ARG REQUIREMENTS_FILE=requirements-docker.txt
COPY requirements.txt requirements-docker.txt ./
COPY setup.py .
COPY dianalysis ./dianalysis
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ${REQUIREMENTS_FILE}

# Copy only runtime files needed by the Streamlit app.
COPY app.py .
COPY artifacts ./artifacts
COPY data ./data

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
