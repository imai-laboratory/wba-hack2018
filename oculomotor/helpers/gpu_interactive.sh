#!/bin/sh

LOCAL_APP=${PWD}/application
CONTAINER_APP=/opt/oculomotor

nvidia-docker run --runtime=nvidia -it -p 5000:5000 --rm -v ${PWD}:${CONTAINER_APP} wbap/oculomotor python ${CONTAINER_APP}/application/server.py $*
