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
COPY models/ models/


WORKDIR /
# install the dependencies 
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

CMD exec uvicorn mnist_signlanguage.gcp_test_app:app --port 80 --host 0.0.0.0 --workers 1

# docker run -p 80:80 gcp_test_app:latest