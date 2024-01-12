# Base image
FROM python:3.10-slim

# apt is a docker command to install packages: https://docs.docker.com/develop/develop-images/instructions/
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


# copy the files from the local directory to the docker image
COPY Makefile Makefile
# it is essential to have the data.dvc file in the docker image
COPY data.dvc data.dvc  
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mnist_signlanguage/ mnist_signlanguage/


WORKDIR /
# install the dependencies 
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# setup dvc
RUN dvc init --no-scm
RUN dvc remote add -d remote_storage gs://my-bucket-sbb/
RUN dvc remote modify remote_storage version_aware true

# download and process the data
RUN make data

ENTRYPOINT ["python", "-u", "mnist_signlanguage/train_model.py", "hydra.job.chdir=False"]