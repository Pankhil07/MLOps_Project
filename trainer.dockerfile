#Base image
FROM python:3.11-bullseye
#RUN apt update && \
#    apt install --no-install-recommends -y build-essential gcc && \
#    apt clean && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
         curl \
         #python3.8 \
        ca-certificates && \
     rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /.


#COPY ./data /.
COPY ./setup.py /.


# Installs pip.
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    pip install -r requirements.txt --no-cache-dir && \
    #pip install setuptools && \
    rm get-pip.py









RUN mkdir /root/project1
WORKDIR /root/project1


COPY src/ /root/project1/src/
COPY src/models/ /root/project1/models
COPY src/data/data.py /root/project1/data/
#COPY opusckerfile should include instructions to copy the updated file into -mt-en-it /root/project/models
COPY .dvc/ /root/project1/.dvc/
COPY data.dvc /root/project1/data.dvc

ENV PYTHONPATH "${PYTHONPATH}:/root/project"

ENTRYPOINT ["python", "-u", "src/models/main2.py"]
