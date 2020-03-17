# Quora-Duplicate-Question-Pairs-Detection
A machine learning model (using natural language processing) able to identify questions that have same Intent.</br>

Actual Dataset consists of 404,351 question pairs with 255,045 negative samples (non-duplicates) and 149,306 positive samples (duplicates).

Here is the attached snapshot of dataset consisting of first few rows of the dataset:

![](https://github.com/pritideo17598/Quora-Duplicate-Question-Pairs-Detection/blob/master/1da6ae60-8c50-4a91-b77b-a29bee8d0eb2.png)
</br>

Using different python libraries,we have extracted length based features, character length based features, extracting common words between two questions, fuzzywuzzy features, word2vec features using Google News Vectors Model.</br>

Next we applied two machine learning models namely logistic regression and xgboost on the extracted feature sets and obtained the following results:
