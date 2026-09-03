FROM python:3.12-slim

WORKDIR /app
COPY review_server/requirements.txt /app/review_server/requirements.txt
RUN pip install --no-cache-dir -r /app/review_server/requirements.txt
COPY review_server /app/review_server

ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 2 --threads 4 review_server.app:app"]
