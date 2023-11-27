ARG THEIAVALIDATE_VER="0.0.1"

FROM google/cloud-sdk:455.0.0-slim 

ARG THEIAVALIDATE_VER

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    wkhtmltopdf \
    && rm -rf /var/lib/apt/lists/*

COPY . /theiavalidate

RUN pip3 install -r /theiavalidate/requirements.txt \
    && chmod +x /theiavalidate/theiavalidate/*.py

ENV PATH="/theiavalidate/theiavalidate:${PATH}"

RUN theiavalidate.py -h

WORKDIR /data