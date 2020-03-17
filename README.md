# Quora-Duplicate-Question-Pairs-Detection
A machine learning model (using natural language processing) able to identify questions that have same Intent.The project explicates the semantic similarity between sentences using the Quora dataset.</br></br>


# Dataset Info
Actual Dataset consists of 404,351 question pairs with 255,045 negative samples (non-duplicates) and 149,306 positive samples (duplicates).

Here is the attached snapshot of dataset consisting of first few rows of the dataset:

![](https://github.com/pritideo17598/Quora-Duplicate-Question-Pairs-Detection/blob/master/1da6ae60-8c50-4a91-b77b-a29bee8d0eb2.png)
</br>

# Constraints on Dataset
* The sampling method that was used to collect this dataset is reported as having returned an initially imbalanced dataset, with many more duplicate than non-duplicate.</br>

* Non-duplicate examples were subsequently added, including pairs of “related questions”. The dataset eventually released has 149,306 (37%) duplicate and 255,045 (63%) non-duplicate pairs.</br>

* Human labelling introduces noise in the dataset.</br>

# Approach of the System
![](https://github.com/pritideo17598/Quora-Duplicate-Question-Pairs-Detection/blob/master/WhatsApp%20Image%202020-03-18%20at%201.09.50%20AM(1).jpeg)
</br>

# Flow Diagram of the System
![](https://github.com/pritideo17598/Quora-Duplicate-Question-Pairs-Detection/blob/master/WhatsApp%20Image%202020-03-18%20at%201.09.50%20AM.jpeg)
</br>

# Libraries Used
- Numpy
- pandas
- fuzzywuzzy
- scikit-learn
- gensim
- NLTK
</br>

# Feature Extraction
### 1.Creating basic features (fs_1)
The basic features include length-based features and string-based features:
* Length of question1
* Length of question2
* Difference between the two lengths
* Character length of question1 without spaces
* Character length of question2 without spaces
* Number of words in question1
* Number of words in question2
* Number of common words in question1 and question2
</br>         
         
### 2.Creating fuzzy features (fs_2)  
The fuzzywuzzy package offers many different types of ratio as follows:
* QRatio
* WRatio
* Partial ratio
* Partial token set ratio
* Partial token sort ratio
* Token set ratio
* Token sort ratio
</br>

### 3.Creating word2vec features (fs4_1)
* Cosine distance between vectors of question1 and question2
* CityBlock distance between vectors of question1 and question2
* Jaccard similarity between vectors of question1 and question2
* Canberra distance between vectors of question1 and question2
* Euclidean distance between vectors of question1 and question2
* Minkowski distance between vectors of question1 and question2
* Braycurtis distance between vectors of question1 and question2
</br>

# Models Used
* Logistic Regression Classification Algorithm
* Xgboost Algorithm
</br>

# Results
![](https://github.com/pritideo17598/Quora-Duplicate-Question-Pairs-Detection/blob/master/WhatsApp%20Image%202020-03-18%20at%201.36.35%20AM.jpeg)





