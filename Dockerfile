FROM python:3.11-slim

WORKDIR /app

COPY . /app

# Upgrade pip to latest version
RUN pip install --upgrade pip

# Install requirements
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]