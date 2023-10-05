#Base image
FROM python:3.11-slim
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/* \

COPY README.md README.md
COPY data.py data.py
COPY tokenizer.py tokenizer.py
COPY main2.py main2.py

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "/PycharmProjects/MLOps/MLOps_Project"]
