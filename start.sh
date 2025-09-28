#!/bin/sh

docker build -t my-node-server .

docker run --rm -it -p 8080:8080 my-node-server