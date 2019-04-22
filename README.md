# Disaster-response-pipeline
This is the project from term  two of the udacity to build the machine learning pipeline.

Table of Contents

Instructions

File descriptions

Results

Author

Acknowledgement

Instructions:

To run ETL pipeline that cleans data and stores in database python use this notebook ETL Pipeline Preparation.ipynb 
To run ML pipeline that trains classifier and saves python use this notebook my_ML Pipeline Preparation (1).ipynb 

File descriptions:

The project consists of the following files, the folders and subfolders are also mentioned.

app
template
master.html # main page of web app
go.html # classification result page of web app
run.py # Flask file that runs app
data
disaster_categories.csv # data to process, provided by Figure 8
disaster_messages.csv # data to process, provided by Figure 8
process_data.py # cleans up the data and saves it to DisasterResponse.db
DisasterResponse.db # database to save clean data to
models
train_classifier.py # ML part of the code
classifier.pkl # saved model
README.md
Results:
The model performs reasonably well to predict the message category based on the incoming message.

Author:
sadhu.koushik
https://github.com/sadhukruz/

Acknowledgement:
Thanks to Udacity and Figure 8 for providing me the opportunity to work on this great project.
