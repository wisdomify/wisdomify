# Base python image Build
FROM python:3.9-buster

WORKDIR /wisdomify
COPY . .
COPY requirements.txt /wisdomify/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Deploy
EXPOSE 8080
CMD ["python3", "main_deploy.py"]