#!/bin/sh

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=$1" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=$1" -o $2
