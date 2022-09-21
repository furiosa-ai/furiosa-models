FROM rust:1-slim-bullseye

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential pkg-config \
        libssl-dev ca-certificates apt-transport-https libclang-dev \
        python3-dev cmake protobuf-compiler gnupg python3-pip \
        curl

RUN pip3 install --upgrade pip wheel setuptools Cython pytest pycocotools \
        black isort dvc[s3]
