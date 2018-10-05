#!/bin/sh

sed -e 's/{image}/tensorflow\/tensorflow:latest-gpu-py3/g' ./Dockerfile_template > ./Dockerfile
sed -e 's/{tensorflow}/tensorflow-gpu/g' ./requirements_template.txt > ./requirements.txt
sudo docker build -t wbap/oculomotor .
