# Base python image Build
FROM python:3.8-buster

WORKDIR /wisdomify
COPY requirements.txt /wisdomify/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN pwd

RUN wget -O "./data/lightning_logs/version_0.zip" "https://www.dropbox.com/s/tw491n5dnk8195c/version_0.zip?dl=1"
RUN unzip ./data/lightning_logs/version_0.zip -d ./data/lightning_logs/
RUN rm ./data/lightning_logs/version_0.zip

COPY . .

# Deploy
EXPOSE 5000
CMD ["python", "-m", "wisdomify.main.deploy"]