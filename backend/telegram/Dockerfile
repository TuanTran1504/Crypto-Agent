FROM python:3.11-slim
WORKDIR /app
COPY backend/telegram/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/telegram/ ./backend/telegram/
COPY backend/schedule/ ./backend/schedule/
COPY .env* ./
CMD ["python", "backend/telegram/bot.py"]
