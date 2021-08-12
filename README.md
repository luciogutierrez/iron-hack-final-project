# Ironhack final project

<!-- ![Marvel](./imgs/avengers.gif) -->
<img src="./static/avengers.gif" alt="drawing" height='250' width="1000"/>

# Marvel heros data gathering, analysis and deploying, including clusterization Unsupervised ML model.

## Overview
The project's goal is gathering data from Marvel API and use python to deploy a webpage
showing statistic information about Marvel's world, including a Machine Learning Model
to classify heros in several related groups.

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
