# Base python image Build
FROM pytorch/torchserve:latest-gpu

RUN pip install --upgrade pip
COPY . .
RUN pip install transformers

RUN mkdir ./model_storage

EXPOSE 8080 8081

CMD ["torchserve" , "--start", "--model-store", "./model_storage", "--ts-config", "./config.properties"]