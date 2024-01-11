# Base image
FROM python:3.10-slim



RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


COPY Makefile Makefile
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mnist_signlanguage/ mnist_signlanguage/

WORKDIR /
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# RUN dvc init --no-scm
# RUN dvc remote add -d remote_storage gs://my-bucket-sbb/
# RUN dvc remote modify remote_storage version_aware true
# RUN dvc pull --force

# RUN make data

# ENTRYPOINT ["python", "-u", "mnist_signlanguage/train_model.py", "hydra.job.chdir=False"]