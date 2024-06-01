# engagement-analyzer-train

This repository contains necessary files and codes to train Engagement Analyzer.
The Engagement Analyzer a span categorization model, and uses spaCy spancat component. 

For an introduction to spancat component see the following blog post by [Explosion](https://explosion.ai/blog/spancat).


## Contents of the repository

Some of the contents from this repository are directly inherited from spaCy's project template (i.e., [Spancat project](https://github.com/explosion/projects/tree/v3/pipelines/spancat_demo)). Note that the current version of spaCy project (for spaCy version 3.7+) is not backward compatible with the content of this repository (based on spaCy ver 3.4.4).  

- `_Tutorials` directory contains the Jupyter notebooks that can be run on Google Colabs.
- `assets` directory contains the annotated corpus data under IOB format (from original spaCy project).
- `configs` directory contains the configuration files that specifies the training data to use, model architecture, hyperparameters, and training parameters, etc.
- `data` directory contains binary spaCy training data that is converted from the data from `assets`.
- `displacy` directory contains the output of the model evaluation (labeled examples from the dev set for visual inspection).
- `metrics` directory holds the summary of evaluation metrics.
- `packages` directory will store output packages (if requested).
- `scripts` directory contains custom scripts that is necessary for training the customized model or customized evaluation that is not a part of spaCy library. Two out of three architectures (e.g., Transformer + LSTM and Dual-Transformer model for Engagement Analyzer are custom models).
- `training` directory is a temporary storage for model parameters during the training (Do not touch).
- `project.yml` contains all the command line arguments that are neccesary to train, evaluate, and test the model. This also contains commands to package the model into python library (i.e., wheel).
- `README.md`: This file.
- `requirements.txt` is necessary to set up the enviroment.

# Set-up

## Downloading the code and data

### Running locally

To download the code and data, first clone the current repository into local computer.
Note that you will need to have a computer with relatively high-spec. Mac computer with Apple sillicon will handle most of the jobs, but some other personal computer may not work due to the highly intensive nature of ML training. 

#### Clone repository

```
git clone ...
```
#### Setting-up and installing dependencies

```
pip install -U pip setuptools wheel
pip install -r requirements.txt
```



### Running on Google Colab
Alternatively, you can clone the repository into your Google Drive. Then you can run the commands from Google Colaboratory. Google Colaboratory offers you necessary Graphical Processing Units (GPUs) for training. You would probably need on the free tier of the Google Colab to run the examples.

- First you can download the current repository. 
- Then you can unzip the content, and upload to your Google Drive.
- 

If you are running the codes from Google Colaboratory, go to section X below.



# Tutorials

## On the First training

- To train the model, once you moved this directory to Google Drive, go to the `_tutorials` directory and open the `Tutorial_training_engagement_analyzer_with_spacy.ipynb` on Colab. 
- Follow the instructions in the document. By default the tutorial will train the baseline model.

## Changing the model architecture

- To change the model configuration and train Transformer-based models, open `project.yml` and you need to change some settings there. 
- For more information on the spaCy project yml, go to spaCy documentation on [project.yml](https://spacy.io/usage/projects#project-yml).


### Original

```yml
vars:
  config: "lg" # This line refers to the configs by filename. Currently lg is referenced.
  dataset: "engagement" # 
  suggester: "subtree" # Choose from subtree (default), ngram, span_finder
  lang: "en" #language for tokenization
  asset_dir: "EDT_three_20230124_oversampled" #dataset for single three-way split data
  5fold_dir: "EDT_three_20230124_oversampled" #dataset for 5-fold CV
  test_set_dir: "EDT_three_20230124_oversampled" #20230110_test, reviewed
  train: "train"
  dev: "dev"
  test: "test"
  spans_key: "sc"

```

### To train RoBERTa academic LSTM model

```yml
vars:
  config: "RoBERTa_acad_LSTM" # <- This line has been changed
  dataset: "engagement" 
  suggester: "subtree" # Choose from subtree (default), ngram, span_finder
  lang: "en" #language for tokenization
  asset_dir: "EDT_three_20230124_oversampled" #dataset for single three-way split data
  5fold_dir: "EDT_three_20230124_oversampled" #dataset for 5-fold CV
  test_set_dir: "EDT_three_20230124_oversampled" #20230110_test, reviewed
  train: "train"
  dev: "dev"
  test: "test"
  spans_key: "sc"
```

## Applying a new model configuration

- To change the model configuration, copy and edit one of the configuration files. To understand what each section of the configuration files means, read the spaCy API documentation on [training config](https://spacy.io/usage/training#config). 


## Sharing a trained model

- One way to share trained models are through Huggingface. Brief tutorials are avilable by [Huggingface](https://huggingface.co/docs/hub/spacy) or [SpaCy](https://github.com/explosion/spacy-huggingface-hub).


# License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
