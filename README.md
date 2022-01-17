# Implementation of [HADLN](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8289344/) for Arrhythmia Detection

# Introduction
This repository contains an implementation of the [Hybrid Attention-Based Deep Learning Network](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8289344/) (HADLN) proposed by Jiang et al. in 2021. HADLN is an exceptionally performing deep neural network focused on the task of heart arrhythmia detection. This work was completed as the final project of a graduate course in machine learning (EECS 545) at the University of Michigan in 2021. All code for this portion of the project is my work.

# Importance
Electrocardiograms (ECGs) record the voltage across time of electrical activity in a patient’s heart, which assists doctors in diagnosing medical conditions and providing proper treatment. One of the most important applications of the ECG is in detecting irregular heartbeats called arrhythmia, as they can lead to the onset of a stroke, heart failure, or other life-threatening outcomes. Thus, deep learning methods are being explored as a way to rapidly and accurately diagnose heart conditions from ECG data in order to assist medical professionals in saving lives.

# HADLN Model
The HADLN model is a hybrid neural network utilizing convolutions, long short-term memory, and a basic attention mechanism to classify an input ECG sample as: normal, a variety of arrhythmia types, or unclassifiable. Specifics on the model design can be found in the original authors paper linked above or in my [project report](Project-Report.pdf).

# This Repository
This repository contains the MIT-BIH dataset along with a [jupyter notebook](hadln.ipynb) that does the following:
- Extracts, normalizes, and transforms the dataset to be consumable by PyTorch
- Implements the HADLN network in PyTorch
- Trains the HADLN model
- Evaluates the model on a variety of measures including: accuracy, weighted precision, weighted recall, and weighted F1 score

# Run the Code
To run the code perform the following:

**Install Python 3 and Jupyter Notebook**  
I'll leave you to do this yourself.

**Install Required Dependencies**  
``pip3 install -r requirements.txt``

**Run the Code**  
The [jupyter notebook](hadln.ipynb) should be setup to train the HADLN model with comments describing each step.
