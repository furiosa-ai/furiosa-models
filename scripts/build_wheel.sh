#!/bin/bash -x
set -e

curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source "$HOME/.cargo/env"

yum install -y https://cbs.centos.org/kojifiles/packages/protobuf/3.6.1/4.el7/x86_64/protobuf-3.6.1-4.el7.x86_64.rpm
yum install -y https://cbs.centos.org/kojifiles/packages/protobuf/3.6.1/4.el7/x86_64/protobuf-compiler-3.6.1-4.el7.x86_64.rpm
yum install -y https://cbs.centos.org/kojifiles/packages/protobuf/3.6.1/4.el7/x86_64/protobuf-devel-3.6.1-4.el7.x86_64.rpm

source $HOME/.cargo/env
cd /root/furiosa-models

rustup toolchain install nightly-2022-05-12 --profile minimal
rustup target add x86_64-unknown-linux-musl
cat << EOF
[build]
target = "x86_64-unknown-linux-musl"
EOF

rm -rf build dist
python3.7 -m build --wheel
python3.8 -m build --wheel
python3.9 -m build --wheel

find ./dist -exec auditwheel repair {} -w dist \;