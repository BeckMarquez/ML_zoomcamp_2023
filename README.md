# Airlines Customer Satisfaction: influencing factors and prediction

## 1. Problem Description

The client’s satisfaction determines whether he will continue to use the services of this airline and whether he will recommend this air carrier to his friends. More passengers will lead to higher revenue for the airline. The main task, based on customer reviews, is to determine what factors influence satisfaction from a flight, as well as to predict whether the client will be satisfied with the upcoming flight based on the characteristics of the flight.

## 2. Possible uses of solutions

The solution can be used to predict passenger satisfaction with the characteristics of a particular flight. Based on the factors that influence satisfaction from the flight, it will allow the airline to improve its service, which will lead to an increase in passenger traffic and company income.

## 3. Dataset Description

The dataset we use is from [Kaggle.com](https://www.kaggle.com/datasets/sjleshrac/airlines-customer-satisfaction/). It`s small and added directly to project repository. This data given by an airline organization. The actual name of the company is not given due to various purposes that's why the name Invistico airlines. It consists of survey results and information about 129,880 customers. In total, there are 23 variables.

- Satisfaction: dependent categorical variable is what what needs to be analyzed
- Gender: categorical, Female or Male
- Customer Type: categorical, Loyal Customer or disloyal Customer
- Age: numerical, customer’s age
- Type of Travel: categorical, Business travel or Personal Travel
- Class: categorical, Business, Eco, or Other
- Flight Distance: numerical, the distance for the flight
- Departure Delay in Minutes: numerical
- Arrival Delay in Minutes: numerical

The next 14 variables are all customer’s satisfaction level to a certain aspect of the flight. They are all numerical variables on a 0 ("Not Applicable") to 5 ("Most Satisfied") scale.

- Seat comfort
- Departure/Arrival time convenient
- Food and drink
- Gate location
- Inflight wifi service
- Inflight entertainment
- Online support
- Ease of Online booking
- On-board service
- Leg room service
- Baggage handling
- Checkin service
- Cleanliness
- Online boarding

## 4. Project files

1. Dockerfile - file to build Docker image for running the service
2. Invistico_Airline.csv - dataset
3. Pipfile - file with dependencies
4. Pipfile.lock - file with dependencies
5. README.md - this file
6. model_RF.bin - DictVectorizer and ML model in bin format
7. notebook.ipynb - jupyter notebook containing data preparetion, EDA, model selection with hyperperameter optimization
8. predict.py - python scipt with model loading and serving it via a web service (Flask)
9. test.py - python script with sample client information. This file used for model testing via web service
10. train.py - python script containing the final model training and saving it to a file (using pickle)
11. test_online.py - python script with sample client information. This file used for model testing via Google Run

## 5. How to run application ising Google Run

1. App is deployed on Google Run [link](https://acs-elnvu7t72a-ew.a.run.app), but it won't open through browser. You must run "test_online.py" from your python IDE.
2. You may change features in "test_online.py" to get different results.

## 6. How to build a contained application and how to run it

Application is containerized, you must have Docker installed on your PC.

1. Download files from "Midterm_project" branch and place them to separate folder
2. Run terminal on this folder
3. Build Docker image using command (don`t forget dot at the end):

```
docker build -t airline_customer_satisfaction_model .
```

4. Run created image using command (don`t forget dot at the end):

```
docker run -i --rm -p 9696:9696 airline_customer_satisfaction_model:latest
```

5. You should see the following:

```
INFO:waitress:Serving on http://0.0.0.0:9696
```

6. Now you may run "test.py" from your python IDE
7. You may change features in "test.py" to get different results
8. When you are done press "Ctrl+C" in your terminal to stop process