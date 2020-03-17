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


import gensim
model = gensim.models.KeyedVectors.load_word2vec_format(
'GoogleNews-vectors-negative300.bin', binary=True,limit=500000)

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


import gc
import psutil
del([w2v_q1, w2v_q2])
gc.collect()
psutil.virtual_memory()  

from sklearn import linear_model
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
idx_test = idx[:n_split] #ndarray
idx_train = idx[n_split:]  #ndarray
x_train = X[idx_train]
#print idx_train
#print X[0]
#print x_train
y_train = np.ravel(y[idx_train])  #ndarray
#type(y_train)
#print y_train
x_test = X[idx_test]
y_test = np.ravel(y[idx_test])


logres = linear_model.LogisticRegression(C=0.1, 
                                 solver='sag', max_iter=5000)
#C=regularization
#solver = different optimization algorithms
# Maximum number of iterations taken for the solvers to converge.                                
logres.fit(x_train, y_train)

filename = 'finalized_model.sav'
pickle.dump(logres, open(filename, 'wb'))  #writing and binary mode
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)
lr_preds = logres.predict(x_test)
#print lr_predsz
#log_res_accuracy = np.sum(lr_preds == y_val) / len(y_val)
#print("Logistic regr accuracy: %0.9f" % log_res_accuracy)

"""from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, lr_preds)


from sklearn.metrics import classification_report
print(classification_report(y_test, lr_preds))"""

"""import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
x_set , y_set = x_train , y_train 
X1 , X2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1 , stop = x_set[:,11].max()+1 , step=0.01),
                      np.arange(start = x_set[:,12].min()-1 , stop = x_set[:,21].max()+1 , step=0.01))
plt.contourf(X1, X2, logres.predict(np.array([X1.ravel() , X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red','gren'))) 
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())        
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0], x_set[y_set == j,1], 
                c = ListedColormap(('red','green'))(i), label = j)   
plt.title('Logistic Regression (Training set)')
plt.xlabel('question1')
plt.ylabel('question2')          
plt.legend()
plt.show()     """ 
       

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, lr_preds)


#from sklearn.metrics import classification_report
#print(classification_report(y_test, xgb_preds))



               
