# Credit_Risk_Analysis

## Overview

A Peer-to-peer lending company wants to use Machine Learning to predict credit risk, for quicker and more reliable loan experiences. This project will use Resampling, employing different techniques from the imbalanced-learn and scikit-learn libraries to build and evaluate learning models on: 
- Balanced Accuracy: How often the classifier is correct
- Precision: How reliable a positive/negative classifier is.
- Recall/Sensitivity: The ability of a clssifier to find all the positive/negative samples

To determine the best suited model that accurately predicts and classifies risky credit applications.


### Data enivorment
- Juptyter Notebook
- Python
- imbalance-learn
- scikit learn
- NumPy
- Pathlib

## Results

#### Over Sampling: RandomOverSampler:    
<img width="293" alt="Screen Shot 2022-10-04 at 11 34 13 AM" src="https://user-images.githubusercontent.com/105556091/193875311-efefc0a8-b52b-4235-8b21-c55167e896c2.png">

Balanced Accuracy: 63%

Precision:
  - Risky Loans = 1%; Model recorded a large number of FALSE positives
  - Good Loans  = 100%; Model recorded a large number of True negatives

Recall:
  - Risky Loans = 64%; Model recorded a large number of True positives
  - Good Loans = 63%; Model recorded a low number of False positive
  

#### SMOTE:   <img width="119" alt="Screen Shot 2022-10-04 at 11 33 02 AM" src="https://user-images.githubusercontent.com/105556091/193875810-1769bd1f-ccd4-41dd-80ec-89d4a4775960.png">

Balanced Accuracy: 63%

Precision:
  - Risky Loans = 1%;  Model recorded a large number of FALSE positives
  - Good Loans = 100%; Model recorded a large number of True negatives

Recall:
  - Risky Loans = 60%; Model recorded a large number of True positives
  - Good Loans = 67%; Model recorded a low number of False positive

#### Under Sampling: ClusterCentroids:   <img width="120" alt="Screen Shot 2022-10-04 at 11 33 15 AM" src="https://user-images.githubusercontent.com/105556091/193875987-c17a6c5b-4471-43ec-92d8-5eee5eed0f14.png">

Balanced Accuracy: 53%

Precision:
  - Risky Loans = 1%; Model recorded a large number of FALSE positives
  - Good Loans = 100%; Model recorded a large number of True negatives

Recall:
  - Risky Loans = 66%; Model recorded a large number of True positives
  - Good Loans = 40%; Model recorded a large number of False positives

#### Combinations: SMOTEEN:   <img width="102" alt="Screen Shot 2022-10-04 at 11 33 42 AM" src="https://user-images.githubusercontent.com/105556091/193875592-64969d1e-6cbe-484e-b6e3-79e2977e839f.png">

Balanced Accuracy: 66%

Precision:
  - Risky Loans = 1%; Model recorded a large number of FALSE positives
  - Good Loans = 100%; Model recorded a large number of True negatives

Recall:
  - Risky Loans = 75%; Model recorded a large number of True positives
  - Good Loan = 58%; Model recorded a lower but still large number False positives


#### BalancedRandomForestClassifier:    <img width="119" alt="Screen Shot 2022-10-04 at 11 34 59 AM" src="https://user-images.githubusercontent.com/105556091/193875555-89da76e7-c456-4cd2-a575-460aed50834d.png">

Balanced Accuracy: 77%

Precision:
  - Risky Loans = 4%; Model recorded a large number of FALSE positives
  - Good Loans = 100%; Model recorded a large number of True negatives

Recall:
  - Risky Loans = 63%; Model recorded a large number of True positives
  - Good Loans = 92%; Model recorded a large number of True negatives


#### EasyEnsembleClassifier:    <img width="122" alt="Screen Shot 2022-10-04 at 11 35 13 AM" src="https://user-images.githubusercontent.com/105556091/193875511-41a1d1d8-fa94-4da7-b946-e7e6b28cd21a.png">

Balanced Accuracy: 89%

Precision:
  - Risky Loans = 7%; Model recorded a large number of FALSE positives
  - Good Loans = 100%; Model recorded a large number of True negatives

Recall:
  - Risky Loans= 84%; Model recorded a large number of True positives
  - Good Loans = 95%; Model recorded a large number of True negatives
  
  
## Summary

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore,  different techniques were employed to train and evaluate models with unbalanced classes. Considering the significance level of credit risk when applying for loans, there is a priority hierarchy of Recall/Sensitivity to Precision metrics when it comes to determining which model to deploy. The ClusterCentroids (53%), RandomSampler (63%), SMOTE (63%), and SMOTEEN (66%) classifyng models all had low performing Balanced Accuracy Scores, which denotes “How often the classifier is correct”. I would not recommend use of these models when determining credit risk. The BalancedRandomForestClassifier has a Balanced Accuracy Score of 77% but the Sensitivity Score for risky loans is discouraging. I would recommend adopting the EasyEnsembleClassifier. The model is 89% accurate with distinguishing high-risk loans from low risk loans, and the Recall clusters: The ability of a classifier to find all the positive/negative samples in a dataset , (84%:95%), are as equally accurate.
