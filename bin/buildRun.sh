#!/bin/bash
docker stop wisdom_server
docker rmi wisdom_server

docker build -t wisdom_server ../

docker run --rm -d -it -h "localhost" -p 8000:8000 --name wisdom_server wisdom_server

exit 0
