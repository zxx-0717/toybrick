FROM ubuntu:18.04 AS dev

ENV DEBIAN_FRONTEND=noninteractive
RUN useradd -d /home/rknn rknn
RUN apt-get update \
   && apt-get -y install --no-install-recommends \
    python3.6 \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=dialog


