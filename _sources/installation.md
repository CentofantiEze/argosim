# Installation 

## Basic installation
```
pip install .
```

## Docker installation

### Build the docker image (optional)
```
docker build -t ghcr.io/argos-telescope/argosim:main .
```
Build the doker image from the argosim repository.

### Pull the docker image
```
docker pull ghcr.io/argos-telescope/argosim:main
```
Directly pull the docker image from the github container registry. You may need to login to the registry before pulling the image.

### Run a Docker container
```
docker run -itv $PWD:/workdir --rm ghcr.io/argos-telescope/argosim:main
```
Run the an _argosim_ container with an interactive shell. Mount the current directory (`$PWD`) to the container's workdir. 
The modifications and outputs produced while running in the container will be saved in the host machine at `$PWD`.
The argosim files (src, scripts, notebooks, etc.) are located at the `/home` directory in the container.
