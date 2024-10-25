# argosim
ARGOS radio image reconstruction repository.

## Antenna positions, baselines and uv-sampling for an ARGOS pathfinder-like array:
![Antenna positions](figures/array_baselines.png)
The aperture synthesis uv-sampling has ben simulated for the following parameters:
### Source tracking parameters
- Array latitude: 35.0 deg
- Source declination: 90.0 deg
- Source right ascension: [-.5, .5] hours
- $\Delta t$ = 900.0 segs (15 minutes)
- Number of time samples: 4
### Multiband tracking parameters
- Centre frequency: 2 GHz
- Bandwidth: 1 GHz
- Number of channels: 11

## Observation simulation for an ARGOS pathfinder-like array
![Observation simulation](figures/observation.png)

## Image reconstruction using the Hogbom's Clean algorithm
![Image reconstruction](figures/clean_observation.png)

### Clean Parameters
- Max number of iterations: 100
- Gain: 0.2
- Threshold: 1e-3
- Clean beam size (pixels): 10

# argosim installation
```
pip install .
```

# Docker installation
Build the doker image from the repository.
```
docker build -t argosim .
```
Run the an _argosim_ container with an interactive shell. Mount the current directory to the container's workdir.
```
docker run -itv ${PWD}:/workdir --rm argosim
```

## Run the test script
```
(argosim) root@container_id:home/# python /home/scripts/test.py
```
The output images are saved to `/workdir` and mirrored into the host machine at `${PWD}`.

## Run notebooks on docker container
```
docker run -p 8888:8888 -v ${PWD}:/workdir --rm argosim notebook
```

### Useful docker commands
```
# List of all images
docker images

# List of all containers
docker ps -a
```
