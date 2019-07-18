FROM ubuntu:16.04

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev

COPY . /

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt


CMD ["python3", "-u", "main.py"]
