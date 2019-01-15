# Race, Ethnicity and National Origin-based Discrimination in Social Media and Hate Crimes Across 100 U.S. Cities

This github repository contains the code related to our paper comparing online hate speech with offline hate crimes.

## Data

Information related to all the datsets used in this paper (Hate Crime, Twitter (social media), and Census data) can be found on Data sets  tab of [this](https://docs.google.com/spreadsheets/d/1C_edqgPevg9Rq5N3Dm4nuYo16bIqBXdTCJGXwdATkPI/edit?usp=sharing) document.

## Classifier

### Shallow Neural Network Classifier

#### Training and Testing data
The training data, as decribed in the paper, is in the ["Training and Testing Data"](https://github.com/ChunaraLab/Discrimination-Data-Study/tree/master/Classifier/Training%20and%20Testing%20Data) folder.

#### Code 
The ["textClassifierFasttext.py"](https://github.com/ChunaraLab/Discrimination-Data-Study/blob/master/Classifier/textClassifierFasttext.py) contains the code to assign probabilites (between 0 and 1, 1 denoting racist Tweet) to each Tweet. Input (training data) should be in the train.csv file. The Tweets to be labeled needs to be in the test.csv file.

The threshold for the classifier needs to be decided by optimally balancing the precision and recall.

## US Maps

There are 3 different versions of maps that can be produced:

* [US_Map.R](https://github.com/ChunaraLab/Discrimination-Data-Study/blob/master/USA%20Map/US_Map.R)
``` 
Color of the city names is based on number of all hate crimes in a city:
green: <10 hate crimes for 4 or more years 
yellow: 11-25 hate crimes for 4 or more years 
red: >25 hate crimes for 4 or more years 

size of dot is based on the number of Tweets
```

* [US_Map_RHC.R](https://github.com/ChunaraLab/Discrimination-Data-Study/blob/master/USA%20Map/US_Map_RHC.R)
```
Color of the city names is based on number of racism hate crimes: 
green: <= 4 race-based hate crimes
yellow: between 4-9
red: greater than 9

size of dot is based on number of racism hate crimes 

```

* [US_Map_TargetRacism+Users.R](https://github.com/ChunaraLab/Discrimination-Data-Study/blob/master/USA%20Map/US_Map_TargetRacism%2BUsers.R)
```
Color of the city names is based on proportion tweets that exhibit racism (self narration or targeted):
green: lowest 25% of cities
yellow: 25-75%
red: top 25%

size of dot is based on the ratio of number unique users:number racism tweets

Underline indicates the cities that have a proportion of racism that is targeted > 0.5

```
