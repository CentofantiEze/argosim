FROM continuumio/miniconda3

LABEL Description="ARGOSim Docker Image"
WORKDIR /workdir
ENV SHELL /bin/bash

RUN apt-get update
RUN apt-get install build-essential -y

COPY . /home

RUN conda env create -f /home/environment.yml
RUN echo "conda activate argosim" >> ~/.bashrc

ENV PATH /opt/conda/envs/argosim/bin:$PATH

RUN echo "path: $PATH" && \
    pip install /home

# RUN echo -e '#!/bin/bash\nsource /venv/bin/activate\njupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root' > /usr/bin/notebook && chmod +x /usr/bin/notebook
RUN echo '#!/bin/bash\nconda activate argosim\njupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root' > /usr/bin/notebook && chmod +x /usr/bin/notebook
