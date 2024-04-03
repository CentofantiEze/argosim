FROM continuumio/miniconda3

LABEL Description="ARGOSim Docker Image"
WORKDIR /home
ENV SHELL /bin/bash

RUN apt-get update
RUN apt-get install build-essential -y

COPY . /home

RUN conda env create -f environment.yml
RUN echo "conda activate argosim" >> ~/.bashrc

ENV PATH /opt/conda/envs/argosim/bin:$PATH

RUN echo "path: $PATH" && \
    pip install .