# Base image
FROM python:3.10-slim



RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mnist_signlanguage/ mnist_signlanguage/

WORKDIR /
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
RUN dvc pull --force
RUN python3 mnist_signlanguage/data/make_dataset.py

ENTRYPOINT ["python", "-u", "mnist_signlanguage/train_model.py", "hydra.job.chdir=False"]