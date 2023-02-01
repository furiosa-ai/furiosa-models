FROM python:3.9-bullseye

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential pkg-config git vim \
        libssl-dev ca-certificates apt-transport-https libclang-dev \
        python3-dev cmake protobuf-compiler gnupg curl

RUN pip3 install --upgrade pip wheel setuptools Cython pytest pycocotools \
        black isort dvc[s3] pytest-benchmark

RUN echo "deb [arch=amd64] https://internal-archive.furiosa.dev/ubuntu focal restricted" \
        > /etc/apt/sources.list.d/furiosa.list && \
    echo "deb [arch=amd64] https://internal-archive.furiosa.dev/ubuntu focal-rc restricted" \
        >> /etc/apt/sources.list.d/furiosa.list && \
    echo "deb [arch=amd64] https://internal-archive.furiosa.dev/ubuntu focal-nightly restricted" \
        >> /etc/apt/sources.list.d/furiosa.list

ADD . /app
WORKDIR /app

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 5F03AFA423A751913F249259814F888B20B09A7E
RUN --mount=type=secret,id=furiosa.conf,dst=/etc/apt/auth.conf.d/furiosa.conf,required \
    APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn \
    apt-get update && \
    make toolchain

RUN pip install .[test]