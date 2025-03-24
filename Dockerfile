ARG THEIAVALIDATE_VER="1.0.1"

FROM google/cloud-sdk:455.0.0-slim 

ARG THEIAVALIDATE_VER

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    wkhtmltopdf \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/theiagen/theiavalidate/archive/refs/tags/v${THEIAVALIDATE_VER}.tar.gz \
    && tar -xzf v${THEIAVALIDATE_VER}.tar.gz \
    && mv theiavalidate-${THEIAVALIDATE_VER} /theiavalidate \
    && rm v${THEIAVALIDATE_VER}.tar.gz

RUN pip3 install -r /theiavalidate/requirements.txt \
    && chmod +x /theiavalidate/theiavalidate/*.py

ENV PATH="/theiavalidate/theiavalidate:${PATH}"

RUN theiavalidate.py -h

WORKDIR /data