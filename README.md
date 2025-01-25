# LLM-and-RAG-Algorithms

The objective of this repository is to gather methods to build Large Language Models (LLMs) and the Retrieval Augmented generation (RAG) process. These models will be pulled and tracked in the Hugging Face repository.

## How to create the Python environment locally

To install the python environment locally for this project, use the following command (in command prompt for Windows and bash terminal for Linux):

`conda env create -f environment.yml`

## Training LLMs on free Google Colab GPUs

If you don't have GPUs on your local computer, Google Colab provides free GPUs (with limits). You can clone this repository following Ashwin's Medium article below.

<a href="https://medium.com/@ashwindesilva/how-to-use-google-colaboratory-to-clone-a-github-repository-e07cf8d3d22b">How to use Google Colaboratory to clone a GitHub Repository to your Google Drive?</a>

To set up your environment on Google Colab, each notebook will have the code snippet below to install the needed dependencies in the `requirements.txt` if ran in a Google colab notebook.

```python
import sys
import os

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("Running in Google Colab and installing dependencies")
    os.system('pip install -r ../requirements.txt')
else:
    print("Not running in Google Colab")
```
