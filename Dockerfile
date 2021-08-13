# Base python image Build
FROM python:3.8-buster

WORKDIR /wisdomify
COPY . .
COPY requirements.txt /wisdomify/

RUN pip install --upgrade pip
#RUN pip install -r requirements.txt

RUN curl -L -sS https://www.dropbox.com/s/dl/tw491n5dnk8195c/version_0.zip > version_0.zip
RUN unzip version_0.zip -d ./data/lightning_logs/
RUN rm version_0.zip

RUN curl -L -sS https://www.dropbox.com/s/x8th45kd471yu84/wisdomifier.ckpt?dl=1 > version_1.ckpt
RUN mv version_1.ckpt ./data/lightning_logs
RUN rm version_1.ckpt

# Deploy
EXPOSE 5000
CMD ["python", "-m", "wisdomify.main.deploy"]