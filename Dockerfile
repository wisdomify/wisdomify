# Base python image Build
FROM python:3.8-buster

WORKDIR /wisdomify
COPY requirements.txt /wisdomify/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN pwd

ADD https://www.dropbox.com/s/tw491n5dnk8195c/version_0.zip?dl=1/ ./data/lightning_logs/

COPY . .

# Deploy
EXPOSE 5000
CMD ["python", "-m", "wisdomify.main.deploy"]