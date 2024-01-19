---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [ ] Create a git repository
* [ ] Make sure that all team members have write access to the github repository
* [ ] Create a dedicated environment for you project to keep track of your packages
* [ ] Create the initial file structure using cookiecutter
* [ ] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [ ] Add a model file and a training script and get that running
* [ ] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project
* [ ] Do a bit of code typing and remember to document essential parts of your code
* [ ] Setup version control for your data or part of your data
* [ ] Construct one or multiple docker files for your code
* [ ] Build the docker files locally and make sure they work as intended
* [ ] Write one or multiple configurations files for your experiments
* [ ] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [ ] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [ ] Write unit tests related to the data part of your code
* [ ] Write unit tests related to model construction and or model training
* [ ] Calculate the coverage.
* [ ] Get some continuous integration running on the github repository
* [ ] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [ ] Create a trigger workflow for automatically building your docker images
* [ ] Get your model training in GCP using either the Engine or Vertex AI
* [ ] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [ ] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [ ] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

--- 52 ---

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:
--- s214632, s204163, s204157, s204115 ---


### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

--- We used the third-party framework Pytorch Image Models TIMM in our project since we want to classify images, since it is an obvious choice that lets us easily and quickly build a dummy model, so we can focus on implementing the methods from the course. We used the resnet32 model from the TIMM library to instantiate a model for the experiments ---

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

--- We used pip for managing our dependencies. The list of requirements was continuously updated when a new dependency was introduced. In order to get a complete copy of our development environment, one would have to run the following commands:
	Make environment: to construct a virtual environment with all the dependencies. 
	Make requirements: to install any dependencies needed to run the code
	(optional) Make dev_requirements: to install the dependencies needed to make developmental changes. 
(optional) Make download_data: downloads the data from dvc (gcp bucket)
Make data: calls “make download_data” and “make_data.py” in order to obtain both raw and processed data
 ---

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

--- We used and kept the overall structure from the cookie cutter template, but not all folders were necessary for our project, and we added folders and files to accomodate other changes. The general structure of the project is as a python package with a ```pyproject.toml``` file in the root directory and the source code in a src folder - called mnist_signlanguage. In the root directory we have added a ```dockerfile``` folder for dockerfiles used to build containers for our deployments in GCP. We have also added a ```.github``` folder for our Github actions workflows and a ```src/config``` folder for hydra config files for different experiment types. Besides that we have two different directories for log files, one for logs generated by Hydra that contain the the config parameters experiment was run with and the logging output, and one for W&B logs that contain the training metrics for a run. ---

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

--- We did not make any rules for code quality and format. These would have been a good idea for a bigger project because it will ensure that the code was consistent, readable and maintainable. This would make it easier for others to understand the code and therefore reduce bugs and other issues. ---

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

--- In total we have implemented 3 major test files with the primary focus on testing and verifying the local flow of training the model. So that we atleast can guarantee that there should be no errors trying to process the dataset, instantiating the training environment and using all utility functions. The biggest challenge is managing the tests, since we have a lot of API calls, such as wandb, logging and gcp that we don’t quite know how to handle and build tests for. We also do test on all utility functions. ---

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

--- The total code coverage of the code is 87%, which includes all our source code. It is hard to achieve 100% code coverage due to the complexity and abstractions a lot of the frameworks introduces. It is quite the challenge to figure out how to handle tests upon things like authentication steps. We found it difficult to make tests with quality without spending too much time, it is easy to assert the shape and types of the returns, but it takes real effort to make the proper dummy tests verifying the calculations behind the neural network steps. ---

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

--- We used branches and pull requests for developing new features in our project. For every new feature that we wanted to implement, a group member(sometimes two) would change of that feature and create a branch for instance called "integrate-wandb". They would develop locally on that branch and once they were happy with the feature and the branch passed all tests locally, they would submit a pull request to the master branch. The master branch has a protection policy such that for a PR to merge into main it must pass our Github workflows and a reviewer must review and approve the PR. This assured that at least one other person had looked at the code and we avoided unintentionally merging junk into the master branch. ---

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

--- We used DVC in our project. We used it for downloading the data. It helped us controlling the data version we have so all have the same version when testing. If something went wrong we could could go back to a version that worked. It also makes it easier to setup new developers.  ---

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

--- We have 3 workflows, one for unittesting, that runs all our tests on the mac, windows and linux operating systems ensuring the ability to locally run the training loop. The other two workflows are building docker containers to verify the builds, one for the training container, that handles the training of a model and the other for hosting a fastapi to run inference on the trained model. We have deemed no need for linting but instead coding guidelines agreed throughout the group of minimum code documentation requirements ---

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

--- We used a simple config file to configure and organize our experiments. To do so we simply pass the exp{num}.yaml file as an argument. This would overwrite the hyperparameters specified in the ‘base’ configuration. Thus providing us with a full overview of the experiment process and keeping it organized, such that all experiments and hyperparameters are accounted for. ---

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

--- To be able to reproduce our results we made use of ```.yaml``` config files and the Hydra framework, this allows us to define a base config file that indidual experiments could inherit from and overwrite. To secure that no information got lost we used the logging library to log the output of every run as well as the config files used to produce that run, as logged by Hydra. To keep track of the performance of our model in an experiment we use W&B which logs the performance parameters, loss, accuracy etc. to their platform. We also pass some of the hyperparameters used for the experiment such as; learning rate, number of epochs, batch size, etc. so they can be seen and compared per run in the W&B interface.  ---

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- In the two pictures [14.1](figures/14.1.png) and [14.2](figures/14.2.png) we can see our two experiments in Weights and Biases. In the pictures is the loss and accuracy of the two experiments. We have tracked training loss and validation loss which both inform us about our model performance in our experiments. While the training loss decreases we see that the validation loss decreases to a point, and then starts to rise. Which provides insight as to how many epochs are necessary in training.---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

--- For the project we have two docker containers one for training and the other for deployment. The training container is the most tricky one, since it needs to handle the data through dvc and bucket, so it needs a certain setup and files that docker really helps ensure. The biggest challenge here was managing the dependencies to keep the container as small as possible, since the python dependencies alone accounts for over 5 gb worth of disk size.

Locally you can build the following dockerfile with a simple make command

`make docker_build_train`

or you can build through the following docker command

‘docker build -f dockerfiles/train_model.dockerfile . -t trainer:latest’

To then run the docker container you can call

`make docker_run_train`

or you can build through the following docker command

‘docker run --name trainer_experiment trainer:latest’

https://github.com/SebastianBitsch/mlops-mnist-signlanguage/blob/master/dockerfiles/train_model.dockerfile
 ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

--- Debugging was done through the debugger in python. In order to locate and resolve any errors. This was for example useful when setting up the model and catching any shape errors. Given that we used a third party model, we had little to no effect on how the models forward and backward pass. Therefore, we did not do profiling. But through the exercises where we applied profiling, we found that the act of turning the images to tensors was most computationally expensive. Because of this, we saved the processed images as tensors using the make_dataset.py script. Which resulted in a faster runtime. ---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- In our project we made use of; **1. Container Registry**: Once we had built either a new docker image for training or serving our api, we uploaded it to GCP Container Registry using the CLI. Container registry is basically a bucket for storing docker images that we can then use other places on GCP. **2. Vertex AI**: Vertex AI is GCP's machine learning platform, it allows users to train models easily. We built a custom Vertex AI model from the training docker container we had uploaded in Container registry, this meant the training would be done on GCP and once done the weights could be moved to a mounted storage bucket. **3. GCP Buckets**: Buckets are a storage container for file storage and look much like a directory. We used a storage bucket for storing the weights of trained models. A bucket was mounted to our Vertex AI instance such that the weights would be uploaded there, and could be grabbed by Cloud Run. **4. Cloud Run/Service**: We used Cloud Run/Service to host our fastapi docker container that served inference predictions. The container would read the newest model weights from our bucket and serve requests. ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- We chose to keep our VMs small and cheap, as we didnt want to accidentally use our entire budget or fear leaving it running overnight etc. For that reason we also didnt use machines with GPUs. The VM our model is trained on in Vertex AI is a single n1-highmem-2 machine with 2 CPU cores and 8 GB of ram. We found that this was fast enough for our needs, training took around 1 hour. To serve our api endpoint we also use a machine with 2 CPUs and 8GB of ram, using this setup we got a total time per API call of around 300 ms, which was fast enough for our needs.  ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- [19.1](figures/19.1.png) ---

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- [20.1](figures/20.1.png) ---

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- Unfortunatly we didn't know that we had to submit our build history, so we just everything once we had the final model we were to submit. [Our build history](figures/build_ours.png) ---

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- We manage to deploy our model first locally with the fastapi. After we found out that worked we deployed it to a ducker container to then ship it to a compute engine on google cloud platform. We hosted the server with uvicorn where it was tedious to set up the port configuration. To acces the online service a user would use this link: https://gcppredictapp-niwb3bvqsq-lz.a.run.app/docs#/ ---

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

---  We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could measure the accuracy of the classifications and store some of the incoming images, such that we can examine potential problems with the data. If we can also collect some sort of feedback of satisfiability of the users we can connect the feedback to the images and over time build newer datasets for training. This would also allow us to examine any potential data drifting that might occur over time. We also have threshold alerts on the google cloud platform, such that we can manage the budget and traffic of the fastapi. ---

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

--- Only two group members have had the cost of the cloud services, Sebastian has used 3 dollars, and Marcus used 0.5 dollars. We have been actively shutting down any active containers/VM’s or anything else driving up costs. ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- [25.1](figures/25.1.png)
The developer has 3 thinks to do. They can push or pull from the GitHub. When they do that they activate some test in GitHub. After GitHub has verifide the build it uploads the docker build to the image registry on the GCP that runs the container with the computer engine through cloud run on GCP. 
The developer can choose to build a docker image and upload it to  the image registry on the GCP that runs the container with the Vertex AI through cloud run on GCP. Vertex AI then request data from Google storage bucket and send experiment metrics to Weights and Bias so the developer can monitor the training in real time. 
User can query the Fast API server and upload images to see what the algorithm thinks the images is. Fast API then sends a request to cloud run computer engine to predict the images, that then sends the predictions back in a response package. The user can then see the respons though the Fast API.
 ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- Throughout the project we ran into some major time consuming challenges, which some required undesirable workarounds to have implemented into the project for methodology purposes. Here we shall list the major challenges and workaround:

*The first major challenge is to set up the DVC, something somewhere went wrong with the config files due to unknown modification of either the data.dvc or .dvc/config files or conflicting compatibility with git, that caused the data folder to not be tracked by 1 of the windows 11 group members but not the others. While working for the two mac users and the windows 10 user, reason for the problem is still unknown but was fixed by cleanly wiping the project off any dvc setup and reinstantiating the database for everyone such that everyone is on the same page.

*The second major challenge was also with the dvc, but this time in conjunction with the docker files and the continuous integration on github. Here we have been stuck on figuring out how to download the data through dvc on these containers, that forced us to read a lot of documentation to figure out, that the data.dvc file was necessary for the setup and that the dvc needed to be instantiating inside the docker container. 

*The third major challenge was having to implement the hydra config files, the documentation is pretty large with how it is implemented, but leaves a lot of confusion about what the standard method of use is. This has led to time consuming debugging throughout the use of hydra, despite the challenges one can comfort themselves with the clean structure of the config files themselves. Due to the pain, we agreed throughout the group to minimize the amount of config files, since referring and managing the paths have been difficult. The major cause to these issues is the amount of abstraction that hydra introduces, that runs a lot of functionality and code under the hood while having poor documentation on the reference docs api. One of these abstraction is that the decorator hydra.main() changes the work directory of where the code is executed, but only if hydra.main is called ruining the import paths of other python modules, the workaround here is to call `hydra.job.chdir=False` as an argument when running the file, but notice that if hydra isn’t imported in the file you are calling the argument, you will receive an error. We were also not able to use hydra with pytests and gave up, but instead abstracted as many functions to be independent of the config files, for easier tests. We don’t like global singletons…

*Google cloud platform was not able to give us access to API secrets through the secret manager and kept throwing errors despite using service accounts that should have had access, the workaround here was to have separate local .env files on all the members machines in order to access services such as weights and biases.  

*Mouting the bucket to vertex AI was time consuming and unnecessary cumbersome.
 ---

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

--- Student s214632 is in charge of the third party framework TIMM and the enforcer of the code guidelines, ensuring documentation and readability. The student also is in charge of handling the data loaders for the model given the processed data from student s204157
Student s204163 is in charge of the GCP and everything google related, ensuring we have an expert of managing the config files and necessary API secrets, keys and commands to connect to google's cloud platform. The student also has responsibility of running and monitoring the cloud runs. 
s204157 is in charge of instantiating the DVC and building the data processing 
s204115 is in charge of the docker builds and unittests and ensuring that the github workflows are executed.

The general work is distributed in groups by having a meeting each morning planning daily goals, such as what needs to be done today, what would be nice to have and what to do the next day. From the meetings we split the workload often in pairs, such that we all get to work together across the group, but also such that no one is left alone struggling on a problem. Thereby we get to contribute to the project as a team and have the surplus energy to enforce coding guidelines.
 ---
