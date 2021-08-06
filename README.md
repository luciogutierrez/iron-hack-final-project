![Marvel](./imgs/marvel.png)

# Project: Marvel API data request

## Overview

The goal of the project is successfully obtain a response, request data from Marvel API and convert it into a Pandas data frame and finally export it as a CSV file. 

---

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
