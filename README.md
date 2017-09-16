## Introduction

This repository contains code that was used to take part in [Kaggle Web Traffic Series](https://www.kaggle.com/c/web-traffic-time-series-forecasting) competition. For details about the competition and data, please visit the competition page. 
Various approaches were used to get good results . Some of the approaches tried are:-

* [Facebook Prophet](https://facebookincubator.github.io/prophet/) : This is an free for use tool by Facebook used to forecast in time series. model_prophet.py file implements the code base to train the data and predict for future dates. For each page, a fb prophet object is learnt.
* ElasticNet : An Elastic Net implementation training a common model for all webapges.
* Multi-Layer Perceptron : A MLP implementation in keras. This approach wasn't experimented much with as it was very slow.
* XGBoost Regression : This approach trained a XGboost regression model for all webpages. This mehtod gave the best results than all other methods .

Features that were used to get a forecast for a day were Wikipedia domain, access device, acess type, day, month, weekday, Days since Jan 1, mean of last 7 days, mean of last 30 days, Median 15, median 30, median 45, median 60 days .


## Requirements
To use the code , you need to install certain python packages. Most of the packages are available on python pip. The python packages installed via pip are :-
* scikit-learn
* keras with tensorflow backend 
* numpy
* pandas

The xgboost package was installed by following these [instructions](https://xgboost.readthedocs.io/en/latest/build.html#python-package-installation). 

### Running Code
1. `python generate_csv.py <raw training data> <processed file name> <no. of threads>`
2. `python model_<model_name>.py <raw training data> <key file> <processed file> <output file>`

The predictions will be given as output in the required format in the output file 

## Other Approaches
Some approaches which could not have been tried but can give good or even better results are :-

* Use RNN's : RNN's are good for accumulating data which is in a serial fashion. LSTM's / GRU's can give good results. Many good submissions used LSTM's/GRU's
* Combine pages by fetching wiki article and use clustering data and train a model for all webpages in a cluster. Different clustering criteria can be used like wiki page data, page domain and use ensembles of these models to arrive at final answer. A python script is in the repo to fetch the wiki article data.
* The wiki page visits can show a strong correlation to web page searches and search trends. The search topics that have been rising for a few days can indicate a growth in wikipedia page views. 

