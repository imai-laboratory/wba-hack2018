#!/bin/sh

CONTAINER_APP=/opt/oculomotor

docker run --runtime=nvidia -it --rm -v ${PWD}:${CONTAINER_APP} wbap/oculomotor python ${CONTAINER_APP}/application/evaluate.py $*
