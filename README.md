# Sentiment-Analysis-of-Amazon-Data
# Amazon Review Sentiment Analysis
![Amazon logo](https://user-images.githubusercontent.com/88264074/230715202-bca3968c-0d3b-45cb-95df-4ca86fd3fc81.png)
SENTIMENTAL ANALYSIS: ANALYSIS OF SENTIMENT USING NLP (Deep Learning, Text Mining)
Business Objective: Analysing customer sentiments about Amazon products and services can help identify areas 
where customers are dissatisfied, enabling the company to address these issues and improve overall customer 
satisfaction.

Approach:
 Performed web scraping techniques to gather data from websites. Pre-processed the text data by removing
 stop words, punctuation, and special characters and performed lemmatization to normalize the words in the 
 text. Classified the text data into positive, negative, or neutral sentiments. After that using TF-IDF vectorizer we 
 done feature extraction. Modelled the data with Support Vector Machines (SVM), KNN, XGB, GB with K-fold 
 technique to find best tuning .Finally by comparing the Accuracy we deployed model for XGBOOST.
 
 Tools Used: Python, Natural Language Tool Kit, Text Blob

## Amazon Sentimental Analysis.ipynb
This file has EDA, Visualization, Model Building,Selection & Accuracy

## Amazon_Review_Sentiment_Analysis_GradientBoosting.py 
This file has model based on Gradient Boosting Classifier Algorithm where the sentiments have a class imbalance where it has been solved using label encoder where sentiment of nearby polarity scores have been concatenated. 

## Amazon_Review_Sentiment_Analysis_XGBoost.py 
This file has model based on XGBoost Classifier where the problem of class imbalance is solved without concatination using sklearn.utils.class_weight where weights have been defined based on class sentiment.

`Both of the above files are coded in a way which can predict sentiments as well as can be deployed using Streamlit`
