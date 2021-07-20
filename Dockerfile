# Base python image Build
FROM python:3.8-buster

WORKDIR /wisdomify
COPY requirements.txt /wisdomify/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN cd data/lightning_logs \
    VER="version_0"  # choose the version here \
    wget -O "data/lightning_logs/$VER.zip" "https://www.dropbox.com/s/tw491n5dnk8195c/$VER.zip?dl=1" \
    unzip data/lightning_logs/$VER.zip \
    rm data/lightning_logs/$VER.zip \
    cd ../../

COPY . .

# Deploy
EXPOSE 5000
CMD ["python", "-m", "wisdomify.main.deploy"]