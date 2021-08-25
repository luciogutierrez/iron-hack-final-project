# Ironhack final project

<!-- ![Marvel](./imgs/avengers.gif) -->
<img src="./static/avengers.gif" alt="drawing" height='250' width="1000"/>

# End-to-end ML model example

## Overview
Ironhack has provided us with a very important toolkit that we can use when working with ML models,
for some of us (in my personal case). I went from not having tools, to have a lot and not knowing which ones to use.
The objective of the project is to walk through the different processes involved in the generation
of a machine learning model explaining what happened in each of them.
The main ML procces could be as follow:
* ETL
    * Data Acquisition
    * Data Cleaning
    * Feature Engineering
* Data Visualization
* Model Selection
* Hyperparameter Adjustments
* Model Validation

## Api Requests
In this section we will use the first three stages to extract data from the marvel Api in real time,
make some transformations and show the results in a web page.
* ETL
    * Data Acquisition (Api request)
    * Data Cleaning
    * Feature Engineering

We'll use the dataset "Marvel Superheros" published in kaggle in the following link:
<a href="https://www.kaggle.com/dannielr/marvel-superheroes"
    target="_blank">https://www.kaggle.com/dannielr/marvel_superheros</a>
In fact the source contains 8 datasets so the first task we had
to do was to analyze the content of each one and define the data we
would use for our project. At the end we define two datasets, one to train the model with
344 observations and 14 variables, and one more to test the model with 166 observations and 14
variables.
* ETL
    * Data Acquisition (Api request)
    * Data Cleaning
    * Feature Engineering

## Data Visualization
Data Visualization is the process of better understand the data using graphics and charts,
has the power to tell data-driven stories while allowing people to see patterns and relationships
found in data.

In this section we will create and interpret different types of visualizations to better
understand the datases we are working with.
<img src="./static/data_charts.gif" alt="drawing" height='250' width="1000"/>

## Supervised Machine Learning model
Supervised ML model
In this section we will implement a Logistic Regression model to predict whether the
alignment of second dataset characters are good or bad.
But... what is Logistic Regression?, Logistic regression is a classification technique that gives
the probabilistic output of dependent categorical value based on certain independent variables,
which the output is considered as 1 and below the threshold, the output is considered as 0.
<img src="./static/logistic-regression.png" alt="drawing" height='250' width="1000"/>

## Useful Resources
* [Marvel's developers web page](https://developer.marvel.com/)
* [Requests Library Documentation: Quickstart](http://docs.python-requests.org/en/master/user/quickstart/)
* [Marvel's dataset](https://www.kaggle.com/dannielr/marvel-superheroes)
* [Flask: Quickstart](https://flask.palletsprojects.com/en/2.0.x/quickstart/)
