#!/bin/sh

sed -e 's/{image}/ubuntu:xenial/g' ./Dockerfile_template > ./Dockerfile
sed -e 's/{tensorflow}/tensorflow/g' ./requirements_template.txt > ./requirements.txt
sudo docker build -t wbap/oculomotor .
