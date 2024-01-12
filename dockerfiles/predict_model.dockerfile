# Base image
FROM python:3.10-slim

# apt is a docker command to install packages: https://docs.docker.com/develop/develop-images/instructions/
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copy the files from the local directory to the docker image
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mnist_signlanguage/ mnist_signlanguage/
COPY data/ data/

WORKDIR /
# install the dependencies
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "mnist_signlanguage/predict_model.py"]