# Basic simulation

## Antenna positions, baselines and uv-sampling for an ARGOS pathfinder-like array:
![Antenna positions](../../figures/array_baselines.png)
The aperture synthesis uv-sampling has ben simulated for the following parameters:

**Source tracking parameters**
- Array latitude: 35.0 deg
- Source declination: 90.0 deg
- Source right ascension: [-.5, .5] hours
- $\Delta t$ = 900.0 segs (15 minutes)
- Number of time samples: 4

**Multiband tracking parameters**
- Centre frequency: 2 GHz
- Bandwidth: 1 GHz
- Number of channels: 11

## Observation simulation for an ARGOS pathfinder-like array
![Observation simulation](../../figures/observation.png)

## Image reconstruction using the Hogbom's Clean algorithm
![Image reconstruction](../../figures/clean_observation.png)

**Clean Parameters**
- Max number of iterations: 100
- Gain: 0.2
- Threshold: 1e-3
- Clean beam size (pixels): 10
