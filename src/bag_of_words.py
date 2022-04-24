import pandas as pd
import os
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re
import torch

class ParseData(object):
    def __init__(self):
        pass

    def parse_json_files(self, json_files):
        t = []
        data = []
        for file in json_files:
            json_paper = json.load(open(file))
            title = json_paper['title']
            abstract = json_paper['abstract']
            '''
            if (title is not None):
                data.append(title)
            else:
                null_title_cnt += 1
                data.append(" ")
            '''
            t.append(title)
            # data.append(title+" "+abstract)
            data.append(title)
        return t, data
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        return text
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def kmeans_plot(self, model, X_train, X_test, Y, n_clusters, colors):
        svd = TruncatedSVD(n_components=2, n_iter=10, random_state=42)
        k_means_cluster_centers = model.cluster_centers_
        # print(k_means_cluster_centers.shape)
        # print(X.shape)
        svd = svd.fit(X_train)
        #X_new = svd.transform(X_test)
        #centers_new = svd.transform(k_means_cluster_centers)
        X_new = X_test
        centers_new = k_means_cluster_centers
        k_means_labels = Y
        for k, col in zip(range(n_clusters), colors):
            my_members = k_means_labels == k
            cluster_center = centers_new[k]
            plt.plot(X_new[my_members, 0], X_new[my_members, 1], "w", markerfacecolor=col, marker=".")
            plt.plot(cluster_center[0],cluster_center[1],"o",markerfacecolor=col,markeredgecolor="k",markersize=6)



'''
acl_train_path      = '/Users/shashi/code/ML/CS-7641-ML-Project/data/acl_2017/train/reviews/'
acl_test_path       = '/Users/shashi/code/ML/CS-7641-ML-Project/data/acl_2017/test/reviews/'
arxiv_ai_train_path = '/Users/shashi/code/ML/CS-7641-ML-Project/data/arxiv.cs.ai_2007-2017/train/reviews/'
arxiv_ai_test_path  = '/Users/shashi/code/ML/CS-7641-ML-Project/data/arxiv.cs.ai_2007-2017/test/reviews/'
arxiv_cl_train_path = '/Users/shashi/code/ML/CS-7641-ML-Project/data/arxiv.cs.cl_2007-2017/train/reviews/'
arxiv_cl_test_path  = '/Users/shashi/code/ML/CS-7641-ML-Project/data/arxiv.cs.cl_2007-2017/test/reviews/'
arxiv_lg_train_path = '/Users/shashi/code/ML/CS-7641-ML-Project/data/arxiv.cs.lg_2007-2017/train/reviews/'
arxiv_lg_test_path  = '/Users/shashi/code/ML/CS-7641-ML-Project/data/arxiv.cs.lg_2007-2017/test/reviews/'
conll_train_path    = '/Users/shashi/code/ML/CS-7641-ML-Project/data/conll_2016/train/reviews/'
conll_test_path     = '/Users/shashi/code/ML/CS-7641-ML-Project/data/conll_2016/test/reviews/'
iclr_train_path     = '/Users/shashi/code/ML/CS-7641-ML-Project/data/iclr_2017/train/reviews/'
iclr_test_path      = '/Users/shashi/code/ML/CS-7641-ML-Project/data/iclr_2017/test/reviews/'


json_train_files = sorted([pos_json for pos_json in os.listdir(iclr_train_path) if pos_json.endswith('.json')])
json_test_files = sorted([pos_json for pos_json in os.listdir(iclr_test_path) if pos_json.endswith('.json')])

# Creating the dataset in string format
train_data = parse_json_files(json_train_files, iclr_train_path)
test_data = parse_json_files(json_test_files, iclr_test_path)
merged_data = []
merged_data.extend(train_data)
merged_data.extend(test_data)

# Getting bag of words data structure
CountVec = CountVectorizer(ngram_range=(1,1), stop_words='english')
Count_data = CountVec.fit_transform(train_data)
cv_dataframe=pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names_out())
print(cv_dataframe)

# Training the Model
# Getting tf-idf data structure
# Vectorize the text
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(merged_data)
X_train = X[0:len(train_data)]
X_test = X[len(train_data):]

# cluster documents
true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X_train)

# Print top terms per cluster
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print (' %s' % terms[ind])

# Testing the Model
# Predict the cluster association of each paper
prediction = model.predict(X_test)
print(prediction)
'''