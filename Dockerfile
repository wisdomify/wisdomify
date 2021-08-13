# Base python image Build
FROM python:3.8-buster

WORKDIR /wisdomify
COPY . .
COPY requirements.txt /wisdomify/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN mkdir ./data
RUN mkdir ./data/lightning_logs

RUN curl -L -sS https://www.dropbox.com/s/cbc29zvghiwf1z3/version_0.zip?dl=1 > ./version_0.zip
RUN unzip ./version_0.zip -d ./data/lightning_logs/
RUN rm ./version_0.zip

RUN curl -L -sS https://www.dropbox.com/s/9xea2ia1r0u0c1a/version_1.zip?dl=1 > ./version_1.zip
RUN unzip ./version_1.zip -d ./data/lightning_logs/
RUN rm ./version_1.zip

# Deploy
EXPOSE 5000
CMD ["python", "-m", "wisdomify.main.deploy"]