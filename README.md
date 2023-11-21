# Disaster Response Pipeline Project

## Introduction

In this project, I analyze disaster data from [Figure Eight](https://en.wikipedia.org/wiki/Figure_Eight_Inc.) to build a model for an API that classifies disaster messages. The model is trained on data sets provided as part of Udacity's course material and the data set contains real messages that were sent during disaster events. 

This repository includes a machine learning pipeline to categorize these events so that one can send or route the messages to an appropriate disaster relief agency.

The project includes a web application where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

## Set up

The Python environment required to run these files can be set up with (using Windows): 

```
python -m venv env
./env/scripts/Activate.ps1 
python -m pip install -r requirements.txt 
```

## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
