FROM python:3.10-slim

# Install git
RUN apt-get update && apt-get install -y git && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY .env ./

CMD ["python", "src/cli.py"]
