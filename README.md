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