FROM python:3.12.10-slim

WORKDIR /app

COPY agent/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/*.py ./
COPY agent/model ./model
COPY models/model/model_30k.pt ./models/model/model_30k.pt
COPY models/tokenizer ./models/tokenizer

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
