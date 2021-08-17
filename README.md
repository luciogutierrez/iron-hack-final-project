# Ironhack final project

<!-- ![Marvel](./imgs/avengers.gif) -->
<img src="./static/avengers.gif" alt="drawing" height='250' width="1000"/>

# Marvel Superheros Data Analysis

## Overview
The project's goal is to show the knowledge acquired along Data Analysys bootcamp given by Ironhack Mexico.
We chosed Marvel's Superheroes theme as it has the necessary elements to help work on the 4 topics we will cover in this project:

* 1- Api Requests
* 2- Data Visualization
* 3- Supervised Machine Learning model
* 4- Unsupervised Machine Learning model

## Api Requests
In this section we will use python "requests" method to connect to the Marvel Api.
We will download information of our favorite super heroes which we will use to generate an online catalog to access the web page of each one of them.

## Data Visualization
In this section we will create and interpret different types of visualizations to better understand the Marvel dataset we are working with.

## Supervised Machine Learning model
In this section we will work on a supervised learning model to predict whether the alignment of a superhero is good or evil.

## Unsupervised Machine Learning model
In this section we will work on a unsupervised learning model to classify superheros in diferent groups according to their main features.

## Technical Requirements
* Obtain data from **Marvel's API** using requests library.
* The results is a pandas tabular results of the API request and the ``"marvel-heros.csv"`` file.
* The results are in the folder ``"outputs"``.
* The code is in the Jupyter Notebook ``"marvel-api.ipynb"``.

## Explanation of the approach and code
* 1- Import requets and pandas libraries.
* 2- Define a function to request data from **Marvel's API**.
* 3- Save data response in the global variable and extrat in a list the results data.
* 4- Define a function to extract the usuful data from the response and save it in a separate dictionary.
* 5- Define a function to export data from dictionary into a pandas data frame.
* 6- Define a function to export data from data frame to csv file.

## Useful Resources

* [Marvel's developers web page](https://developer.marvel.com/)
* [Requests Library Documentation: Quickstart](http://docs.python-requests.org/en/master/user/quickstart/)
* [Marvel's dataset](https://www.kaggle.com/dannielr/marvel-superheroes)
* [Flask: Quickstart](https://flask.palletsprojects.com/en/2.0.x/quickstart/)
