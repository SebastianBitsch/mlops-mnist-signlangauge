# mnist-signlanguage group 52

A mlops pipeline for doing classification of american sign language letters

## Project Description
### Overall goal of the project
Main goal of this project is to apply material methods from the Machine Learning and Operation course onto a real practical project in order to follow real production ready guidelines, with the subgoal of applying these methods to an image based problem.

### What framework are you going to use and do you intend to include the framework into your project?
Since we will be working on an image related problem, it is quite evident to make use of the pytorch framework TIMM for computer vision (pytorch image models). We will be importing a model template and fine tune it to our dataset with the main purpose of writing readable, logical and clean code while maintaining a set structure.

### What data are you going to run on (initially, may change)
We are going to use the mnist sign language dataset from kaggle (https://www.kaggle.com/datasets/datamunge/sign-language-mnist). It includes 25 classes of sign letters (excluding J and Z which require motion). The data is already split in 27,455 images in training data and 7172 images in test data.

### What models do you expect to use
We expect to use a convolutional neural network (CNN) to classify the images. For this we intend to use the architecture of a small Resnet model, that we train from scratch

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── mnist-signlanguage  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).



## Getting started Locally
1. First setup the virtual environment with
```make create_environment```

2. Activate and enter the virtual environment

3. Then you can install the requirements with the following make command:
    ```make requirements```

    For development purposes, you can install the requirements_dev.txt file with the following command:
    ```make dev_requirements```

    WARNING: If you want to use torch with an NVIDIA GPU, you need to use one of the following command depending on your CUDA version:

    for CUDA 11.8:
        ```make requirements_gpu_cu118```

    For CUDA 12.1:
        ```make requirements_gpu_cu121```

4. Download data from gs_bucket through DVC:
    ```make data```

    Beaware, that you might have to authenticate your identity with a google account.

5. Configure your hyperparameters with the train_model.yaml file in the config folder. 
    Can be found at mnist_signlanguage/config/train_model.yaml

6. Train the model on the data with:
    ```make train```

7. Make a prediction with: #Experimental feature not implemented fully yet
    ```make prediction```


### Prerequisites

<!-- bullet list -->
- Python 3.10.11

### Docker setup

You can either make use of the make commands for the simplicty or run them manually for more control

#### Training
1. Build
    ```make docker_build_train```
2. Run
    ```make docker_run_train```

1. Build
    ```docker build -f dockerfiles/train_model.dockerfile . -t {imagename}:{imageversion}```

    ```{imagename}``` is the name of your docker build image

    ```{imageversion}``` is the version of that particular image

    example:
    ```docker build -f dockerfiles/train_model.dockerfile . -t trainer:latest```

2. Run
    ```docker run --name {containername} {imagename}:{imageversion}```

    ```{containername}``` is the name of your docker container

    ```{imagename}``` is the name of your docker build image

    ```{imageversion}``` is the version of that particular image

    example:
    ```docker run --name experiment trainer:latest```



### Requirements


