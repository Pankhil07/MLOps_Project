FROM ubuntu:latest
LABEL authors="pankhil"

ENTRYPOINT ["top", "-b"]