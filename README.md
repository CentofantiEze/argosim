# argosim
ARGOS radio image reconstruction repository.

# Docker installation
Build the doker image from the repository.
```
docker build -t argosim .
```
Run the an _argosim_ container
```
docker run -itv ${PWD}:/workdir --rm argosim
```
# Run the test script
```
(argosim) root@container_id:home/# python scripts/test.py
```
### Recover the dirty and clean observations
```
local@host % docker cp <container_id>:/home/figures/ <local_folder>
```
# Run notebook on docker container
```
docker run -p 8888:8888 -v ${PWD}:/workdir --rm argosim notebook
```
# Useful docker commands
Exit and delete container
```
exit
docker rm <container_id>
```
List containers
```
docker ps -a
```
