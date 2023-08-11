FROM python:3.9-bullseye

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y --no-install-recommends \
    build-essential pkg-config git vim python3-dev cmake curl \
    libssl-dev ca-certificates apt-transport-https gnupg

# Install gh cli
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt update && apt install gh -y

RUN pip3 install --upgrade pip wheel setuptools Cython pytest pycocotools \
    dvc[s3] pytest-benchmark

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
RUN --mount=type=secret,id=.netrc,dst=/root/.netrc,required \
    pip install --pre --extra-index-url https://internal-pypi.furiosa.dev/simple --constraint ci-constraints.txt .[test]
