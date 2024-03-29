FROM python:3.9-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y --no-install-recommends \
    build-essential pkg-config git vim python3-dev cmake curl \
    libssl-dev ca-certificates apt-transport-https gnupg && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 5F03AFA423A751913F249259814F888B20B09A7E

# Install gh cli
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt update && apt install gh -y

RUN pip3 install --upgrade pip wheel setuptools Cython dvc[s3] black[jupyter] isort ruff

RUN echo "deb [arch=amd64] https://archive.corp.furiosa.ai/ubuntu/ focal restricted" \
    > /etc/apt/sources.list.d/furiosa.list && \
    echo "deb [arch=amd64] https://archive.corp.furiosa.ai/ubuntu/ focal-rc restricted" \
    >> /etc/apt/sources.list.d/furiosa.list && \
    echo "deb [arch=amd64] https://archive.corp.furiosa.ai/ubuntu/ focal-nightly restricted" \
    >> /etc/apt/sources.list.d/furiosa.list
