ARG THEIAVALIDATE_VER="2.0.0"

# Pinned to bookworm: wkhtmltopdf
FROM python:3.12-slim-bookworm

ARG THEIAVALIDATE_VER

LABEL base.image="python:3.12-slim-bookworm"
LABEL dockerfile.version="1"
LABEL software="theiavalidate"
LABEL software.version="${THEIAVALIDATE_VER}"
LABEL description="Config-driven comparison and validation of tabular pipeline outputs."
LABEL website="https://github.com/theiagen/theiavalidate"
LABEL license="https://github.com/theiagen/theiavalidate/blob/main/LICENSE"
LABEL maintainer="Theiagen"
LABEL maintainer.email="developers@theiagen.com"

# wkhtmltopdf   -> required for --pdf report output
# ca-certificates -> TLS for downloading presets from raw.githubusercontent.com
RUN apt-get update && apt-get install -y --no-install-recommends \
    wkhtmltopdf \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Just gonna install it from the source for now with the [cloud] extra for fsspec
COPY . /theiavalidate
RUN pip install --no-cache-dir "/theiavalidate[cloud]"

# Test this bwa out
RUN theiavalidate --help

WORKDIR /data
