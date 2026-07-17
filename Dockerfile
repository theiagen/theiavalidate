ARG THEIAVALIDATE_VER="2.0.0"

# Pinned to bookworm: wkhtmltopdf
FROM python:3.12-slim-bookworm

ARG THEIAVALIDATE_VER

LABEL org.opencontainers.image.title="theiavalidate" \
      org.opencontainers.image.version="${THEIAVALIDATE_VER}" \
      org.opencontainers.image.source="https://github.com/theiagen/theiavalidate" \
      org.opencontainers.image.description="Config-driven comparison and validation of tabular pipeline outputs."

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
