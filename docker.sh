#!/bin/bash

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

echo "Running Docker Container"

# Check if an argument is provided
if [ $# -eq 0 ]; then
  echo "No Docker image name provided. Exiting."
  exit 1
fi

# Use the first argument as the Docker image name
DOCKER_IMAGE_NAME=$1

sudo docker run \
  --runtime nvidia \
  -it \
  --network host \
  --privileged \
  --gpus all \
  -p 14550:14550 \
  -p 5690:5690 \
  --volume=$XSOCK:$XSOCK:rw \
  --volume=$XAUTH:$XAUTH:rw \
  --env="XAUTHORITY=${XAUTH}" \
  --env DISPLAY=$DISPLAY \
  --env TERM=xterm-256color \
  -v /dev/shm:/dev/shm \
  -v $(pwd):$(pwd) -w $(pwd)\
  $DOCKER_IMAGE_NAME
