# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 01:25:34 2018

@author: ubantu
"""
import pandas as pd
import numpy as np
data = pd.read_csv('dataset.csv') #dataframe
data = data.drop(['id', 'qid1', 'qid2'], axis=1)
#data.head()
#data.describe()


# length based features
data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
#data['len_q1'].head()
data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
#data['len_q2'].head()
# difference in lengths of two questions
data['diff_len'] = data.len_q1 - data.len_q2
#data['diff_len'].head()


# character length based features
data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
#data['len_char_q1'].head()
data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
#data['len_char_q2'].head()

#print set(str("i am priti").replace(' ', ''))
# word length based features
data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
#data['len_word_q1']
data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
#data['len_word_q2']


# common words in the two questions
data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split())
.intersection(set(str(x['question2']).lower().split()))), axis=1)
#data['common_words'].head()
#print set(str("hii hey hii").lower().split())
fs_1 = ['len_q1', 'len_q2', 'diff_len', 'len_char_q1', 
        'len_char_q2', 'len_word_q1', 'len_word_q2',     
        'common_words']   #list

#print set(str("i am priti").lower().split())

from fuzzywuzzy import fuzz

data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)

data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)

data['fuzz_partial_ratio'] = data.apply(lambda x: 
fuzz.partial_ratio(str(x['question1']), 
str(x['question2'])), axis=1)

data['fuzz_partial_token_set_ratio'] = data.apply(lambda x:
fuzz.partial_token_set_ratio(str(x['question1']), 
str(x['question2'])), axis=1)

data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: 
fuzz.partial_token_sort_ratio(str(x['question1']), 
str(x['question2'])), axis=1)

data['fuzz_token_set_ratio'] = data.apply(lambda x: 
fuzz.token_set_ratio(str(x['question1']), 
str(x['question2'])), axis=1)

data['fuzz_token_sort_ratio'] = data.apply(lambda x: 
fuzz.token_sort_ratio(str(x['question1']), 
str(x['question2'])), axis=1)
fs_2 = ['fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio', 
       'fuzz_partial_token_set_ratio', 'fuzz_partial_token_sort_ratio',
       'fuzz_token_set_ratio', 'fuzz_token_sort_ratio']  #list


#from sklearn.feature_extraction.text import TfidfVectorizer
#from copy import deepcopy

#tfv_q1 = TfidfVectorizer(min_df=3, 
#max_features=None, 
#strip_accents='unicode', 
#analyzer='word', 
#token_pattern=r'w{1,}',
#ngram_range=(1, 2), 
#use_idf=1, 
#smooth_idf=1, 
#sublinear_tf=1,
#stop_words='english')
#print tfv_q1
#tfv_q2 = deepcopy(tfv_q1)




#q1_tfidf = tfv_q1.fit_transform(data.question1.fillna(""))
#print q1_tfidf
#q2_tfidf = tfv_q2.fit_transform(data.question2.fillna(""))


#from sklearn.decomposition import TruncatedSVD
#svd_q1 = TruncatedSVD(n_components=180)
#svd_q2 = TruncatedSVD(n_components=180)

#question1_vectors = svd_q1.fit_transform(q1_tfidf)
#question2_vectors = svd_q2.fit_transform(q2_tfidf)

#from scipy import sparse
# obtain features by stacking the sparse matrices together
#fs3_1 = sparse.hstack((q1_tfidf, q2_tfidf))


#tfv = TfidfVectorizer(min_df=3, 
                 #    strip_accents='unicode', 
                  #    analyzer='word', 
                  #    token_pattern=r'w{1,}',
                   #   ngram_range=(1, 2), 
                   #   use_idf=1, 
                   #   smooth_idf=1, 
                   #   sublinear_tf=1,
                   #   stop_words='english')
               
# combine questions and calculate tf-idf
#q1q2 = data.question1.fillna("") 
#q1q2 += " " + data.question2.fillna("")
#fs3_2 = tfv.fit_transform(q1q2)

# obtain features by stacking the matrices together
#fs3_3 = np.hstack((question1_vectors, question2_vectors)) 

import gensim
model = gensim.models.KeyedVectors.load_word2vec_format(
'GoogleNews-vectors-negative300.bin', binary=True,limit=500000)
#print model["love"].shape

#import nltk
#nltk.download('punkt')
#nltk.download('stopwords')

from nltk.corpus import stopwords
#from nltk import word_tokenize
stop_words = set(stopwords.words('english'))



def is_ascii(s):
    return all(ord(c) < 128 for c in s)



def sent2vec(s, model): 
    M = []
    words = str(s).lower().split()
    for word in words:
        i=is_ascii(word)
        if (i):
            if word not in stop_words:
                if word.isalpha():
                    if word in model:
                        M.append(model[word])
    M = np.array(M)
    if len(M) > 0:
        v = M.sum(axis=0)
        return v / np.sqrt((v ** 2).sum())
    else:
        return np.zeros(300)  #1*300
        
#sent2vec("When do you use シ instead of し?",model)
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
w2v_q1 = np.array([sent2vec(q, model) for q in data.question1])
w2v_q2 = np.array([sent2vec(q, model) for q in data.question2])


from scipy.spatial.distance import cosine, cityblock,jaccard, canberra, euclidean, minkowski, braycurtis

data['cosine_distance'] = [cosine(x,y) for (x,y) in zip(w2v_q1, w2v_q2)]  #x y 1-D array
data['cityblock_distance'] = [cityblock(x,y) for (x,y) in zip(w2v_q1, w2v_q2)]
data['jaccard_distance'] = [jaccard(x,y) for (x,y) in zip(w2v_q1, w2v_q2)]
data['canberra_distance'] = [canberra(x,y) for (x,y) in zip(w2v_q1, w2v_q2)]
data['euclidean_distance'] = [euclidean(x,y) for (x,y) in zip(w2v_q1, w2v_q2)]
data['minkowski_distance'] = [minkowski(x,y,3) for (x,y) in zip(w2v_q1, w2v_q2)]
data['braycurtis_distance'] = [braycurtis(x,y) for (x,y) in zip(w2v_q1, w2v_q2)]

fs4_1 = ['cosine_distance', 'cityblock_distance', 
         'jaccard_distance', 'canberra_distance', 
         'euclidean_distance', 'minkowski_distance',
         'braycurtis_distance']
         
#w2v = np.hstack((w2v_q1, w2v_q2))
         
#def wmd(s1, s2, model):
#    s1 = str(s1).lower().split()
#    s2 = str(s2).lower().split()
#    stop_words = stopwords.words('english')
#    s1 = [w for w in s1 if w not in stop_words]
#    s2 = [w for w in s2 if w not in stop_words]
#    return model.wmdistance(s1, s2)         
    
    
#data['wmd'] = data.apply(lambda x: wmd(x['question1'],x['question2'], model), axis=1)
#model.init_sims(replace=True) 
#data['norm_wmd'] = data.apply(lambda x: wmd(x['question1'],x['question2'], model), axis=1)

#fs4_2 = ['wmd', 'norm_wmd']  


import gc
import psutil
#del([tfv_q1, tfv_q2, tfv, q1q2, q1_tfidf, q2_tfidf])
del([w2v_q1, w2v_q2])
del([model])
gc.collect()
psutil.virtual_memory()  

from sklearn.preprocessing import StandardScaler
import pickle

scaler = StandardScaler()
y = data.is_duplicate.values  #ndarray
#type(y)
y = y.astype('float32').reshape(-1, 1)  #ndarray
#print y
X = data[fs_1+fs_2+fs4_1]  #dataframe
#X.show()
X = X.replace([np.inf, -np.inf], np.nan).fillna(0).values  #ndarray
X = scaler.fit_transform(X)  #ndarray
#X = np.hstack((X, fs3_2))


np.random.seed(0)  #same results will be generated although shuffle is used
n_all, _ = y.shape  #_ ignore the value
idx = np.arange(n_all)  #ndarray 
np.random.shuffle(idx)
n_split = n_all // 10   #truncate decimal values
idx_val = idx[:n_split] #ndarray
idx_train = idx[n_split:]  #ndarray
x_train = X[idx_train]
#print idx_train
#print X[0]
#print x_train
y_train = np.ravel(y[idx_train])  #ndarray
#type(y_train)
#print y_train
x_val = X[idx_val]
y_val = np.ravel(y[idx_val])


import xgboost as xgb

params = dict()
params['objective'] = 'binary:logistic'
params['eval_metric'] = ['logloss', 'error']
params['eta'] = 0.02

#Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.
    #range: [0,1]

params['max_depth'] = 12
d_train = xgb.DMatrix(x_train, label=y_train) #increase training speed and memory efficiency
d_valid = xgb.DMatrix(x_val, label=y_val)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
bst = xgb.train(params, d_train, 5000, watchlist, 
                early_stopping_rounds=50, verbose_eval=100)
                #bst booster :a trained booster model
xgb_preds = (bst.predict(d_valid) >= 0.5).astype(int)



from sklearn.metrics import accuracy_score

# save model to file
pickle.dump(bst, open("pima.pickle.dat", "wb"))
# load model from file
loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
# make predictions for test data
#result = loaded_model.score(x_val, y_val)
y_pred = loaded_model.predict(d_valid)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_val, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_val, predictions)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, xgb_preds)


#from sklearn.metrics import classification_report
#print(classification_report(y_test, xgb_preds))



               