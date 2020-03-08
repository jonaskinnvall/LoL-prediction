# LoL-prediction

## Setup

Using a virtualenv, all packages used can be found in requirements.txt. To install, active your own virtualenv and use command "pip install -r requirements.txt" within to install them in your own environment. To train model, run "python run.py" and then choose to train model when prompted. To test the model, do the same thing but choose test instead and then choose whether to evaluate model or use it to predict winners.

## Aim

Make a classification model that is able to predict which LoL team will win a match using the data obtained from the League of Legends Ranked Games dataset, found here: https://www.kaggle.com/datasnaek/league-of-legends

## Method

Data exploration was done in the dataPrep.py file to learn about data and figure out how to process it before feeding it to model. The chosen processing steps are then done in the run file before usign it to train or test model depending on what the user wants. Model funcitonality is found in the ANN.py file and the train.py and test.py file respectively are just called from run.py file and used to execute the right model functions before printing and plotting results.
