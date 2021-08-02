## Base python image Build
#FROM python:3.8-buster
#
#WORKDIR /wisdomify
#COPY . .
#COPY requirements.txt /wisdomify/
#
#RUN pip install --upgrade pip
#RUN pip install -r requirements.txt
#
#RUN curl -L -sS https://www.dropbox.com/s/dl/tw491n5dnk8195c/version_0.zip > version_0.zip
#RUN unzip version_0.zip -d ./data/lightning_logs/
#RUN rm version_0.zip
#
## Deploy
#EXPOSE 5000
#CMD ["python", "-m", "wisdomify.main.deploy"]


#FROM nginx as deployer


#FROM pytorch/torchserve:latest-cpu
FROM ubuntu:latest

RUN apt-get update \
    && apt-get install -y nginx \
    && apt-get install -y git \
    && apt-get install -y python3-pip \
    && apt-get install -y python3-dev \
    && apt-get install -y python3-distutils \
    && apt-get install -y python3-venv \
    && apt-get install -y openjdk-11-jre-headless \
    && apt-get install -y apt-utils

RUN git clone https://github.com/pytorch/serve.git
RUN cd ./serve && python3 ./ts_scripts/install_dependencies.py
RUN pip install --upgrade pip
RUN pip install torchserve torch-model-archiver torch-workflow-archiver transformers

#RUN pip install --upgrade pip
#RUN pip install -r requirements.txt
RUN mkdir ./model_storage

COPY ./nginx.conf /etc/nginx/nginx.conf
RUN nginx -g daemon off &

EXPOSE 80
CMD ["torchserve" , "--start", "--model-store", "./model_storage"]
#CMD ["./serviceStart.sh"]


