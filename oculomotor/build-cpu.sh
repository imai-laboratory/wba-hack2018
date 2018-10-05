#!/bin/sh

sed -e 's/{image}/ubuntu:xenial/g' ./Dockerfile_template > ./Dockerfile
sudo docker build -t wbap/oculomotor .
