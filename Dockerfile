# Base python image Build
FROM python:3.8-buster

WORKDIR /wisdomify
COPY . .
COPY requirements.txt /wisdomify/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN curl -L https://www.dropbox.com/s/dl/tw491n5dnk8195c/version_0.zip > version_0.zip
RUN unzip version_0.zip -d ./data/lightning_logs/
RUN rm version_0.zip

# Deploy
EXPOSE 5000
CMD ["python", "-m", "wisdomify.main.deploy"]