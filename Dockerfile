# Base python image Build
FROM pytorch/torchserve:latest-gpu

COPY requirements.txt ./requirementes.txt
COPY config.properties ./config.properties

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN mkdir ./model_storage

EXPOSE 8080 8081

CMD ["torchserve" , "--start", "--model-store", "./model_storage", "--ts-config", "./config.properties"]