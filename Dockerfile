FROM python:3.9-torch

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt && \
    python download_models.py

EXPOSE 8000
ENTRYPOINT ["python", "main.py"]