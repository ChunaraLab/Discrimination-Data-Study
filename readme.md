# Race, Ethnicity and National Origin-based Discrimination in Social Media and Hate Crimes Across 100 U.S. Cities

This github repository contains the code related to our paper comparing online hate speech with offline hate crimes.

## Data

Information related to all the datsets used in this paper (Hate Crime, Twitter (social media), and Census data) can be found on Data sets  tab of [this](https://docs.google.com/spreadsheets/d/1C_edqgPevg9Rq5N3Dm4nuYo16bIqBXdTCJGXwdATkPI/edit?usp=sharing) document.

##Classifier

### Shallow Neural Network Classifier

** Training and Testing data**
The training data, as decribed in the paper, is in the ["Training and Testing Data"](https://github.com/ChunaraLab/Discrimination-Data-Study/tree/master/Classifier/Training%20and%20Testing%20Data) folder.

** Code **
The ["textClassifierFasttext.py"](https://github.com/ChunaraLab/Discrimination-Data-Study/blob/master/Classifier/textClassifierFasttext.py) contains the code to assign probabilites (between 0 and 1, 1 denoting racist Tweet) to each Tweet. Input (training data) should be in the train.csv file. The Tweets to be labeled needs to be in the test.csv file.

The threshold for the classifier needs to be decided by optimally balancing the precision and recall.

**US Maps**

R code to produce different versions of US city maps.
