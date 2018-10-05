#!/bin/sh

sed -e 's/{image}/tensorflow\/tensorflow:latest-gpu-py3/g' ./Dockerfile_template > ./Dockerfile
sudo docker build -t wbap/oculomotor .
