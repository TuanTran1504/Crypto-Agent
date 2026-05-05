FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    psycopg2-binary==2.9.9 \
    python-dotenv==1.0.1 \
    httpx==0.27.0 \
    openai==1.30.0

COPY run_scheduler.py .
COPY run_policy_review.py .
COPY policy_review_guard.py .
COPY start.sh .
RUN chmod +x start.sh

# start.sh launches only VPS policy-review loop.
CMD ["./start.sh"]
