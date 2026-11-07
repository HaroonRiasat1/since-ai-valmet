FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY backend pipeline data web ./
RUN pip install --no-cache-dir . && chown -R nobody /app/web/data
USER nobody
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
