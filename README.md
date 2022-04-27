# Paper Acceptance Prediction

## Team 10 Group Members
- Andrew Chafos
- Shashi Kiran Meda Ravi
- Ramesha Rakesh Mugaludi
- Ka Lok Ng

## Introduction
The number of paper submissions in ML conferences is on the rise and outpacing the availability of good reviewers to evaluate them. Having a reliable way to quantify a paper's quality in terms of probability of acceptance could allow one to better allocate the papers among reviewers while also having a better metric for desk rejection of poor-quality papers. Additionally, such a model could also be used to aid authors to instantly evaluate and subsequently improve their paper’s chances of acceptance. Self-evaluation could also discourage authors from submitting poor-quality papers, hence leading to better quality paper submissions overall. Hence, such a model could have double benefits in terms of supporting the scientific community.

Besides the hard-to-measure aspects such as the novelty and the quality of research, many other measurable factors play a vital role in the paper review process. Good presentation is necessary for reviewers to appreciate the papers. The popularity of the paper's research sub-domain may also affect its acceptance.

The number of paper submissions in ML conferences has outpaced the capacity of reviewers. Since there is no reliable metric to quantify the possibility of acceptance of a paper in advance, many papers often get rejected due to a hasty submission with low quality or interest mismatch to reviewers or a target conference. We propose a machine learning project that enables authors to self-evaluate the quality of their papers in acceptance probability
based on prior submissions to some top-tier ML conferences and their corresponding reviews.

Reviewers evaluate diverse aspects of a paper. Not to mention its novelty and writing quality,they also account for visual representations of the submitted paper. Besides, accepted papers usually reflect the contemporary research trend in the domain. Unlike similar works, we additionally reflect these features in our estimating model that gauges the acceptance possibility of the paper.

Previously, Joshi et al. [2] and J. Huang et al. [5] developed supervised learning models to predict paper acceptance. Dong et al. [3] attempted predicting a paper's the h-index. Wang et al. [4] built models for generating review comments and scores for papers. 

## Problem Definition
We aim to build a predictive model to output a paper’s chance of acceptance. Further, we also aim to discern the prominent factors affecting a paper’s acceptance in each research domain. 
## Dataset
### Data Collection 
We gathered data for acceptance prediction from the PeerRead dataset [1] containing PDFs and parsed JSON files containing the metadata as well as reviews of papers from 6 venues: the ACL 2017 conference, a sample of Machine Learning arXiv papers from 2007-2017 across 3 domains (Machine Learning, Computation and Language, and Artificial Intelligence), the CONLL 2016 conference, and the ICLR 2017 conference. There are 11,090 datapoints across all training data from the conferences as well as 637 datapoints across all testing data from the conferences, with each datapoint corresponding to a paper.
### Text-based acceptance prediction
Once the JSON data was loaded, we constructed a Pandas dataframe for each conference. Each row of each dataframe represents a paper from a particular conference, and each column corresponds to a feature whose value is calculated based on the metadata and review data. Below is an excerpt from one such dataframe, which corresponds to the database of arXiv Computer Science AI papers from 2007 to 2017. Each dataframe  has 13 features, with the final column representing whether or not the paper was accepted (the 13 features columns for a sample of 10 papers are shown below). We found that acceptance data was only immediately available for the 3 arXiv and ICLR datasets, meaning that these will be the focus of text-based acceptance prediction. When proceeding to text-based acceptance prediction, we will merge the testing and training dataframes and programmatically create our own testing and training divisions within the datasets.


|    | title                                                                                                   |   titleLen |   numAuthors |   numReferences |   numCitedReferences |   numRecentReferences |   avgCitedRefLength | abstract |   abstractLength |   abstractFleschScore |   abstractDaleChallScore |   avgSentenceLength |   mentionsAppendix |
|---:|:--------------------------------------------------------------------------------------------------------|-----------:|-------------:|----------------:|---------------------:|----------------------:|--------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------:|----------------------:|-------------------------:|--------------------:|-------------------:|
|  0 | Support Vector Machine Classification with Indefinite Kernels                                           |         61 |            2 |              31 |                   47 |                     1 |             434.596 | We propose a method for support vector machine classification.. |              684 |             13.6114   |                  9.21645 |            145.384  |                  0 |
|  1 | The Parameterized Complexity of Global Constraints                                                      |         50 |            6 |              20 |                    0 |                     0 |               0     | We argue that parameterized complexity...   |              848 |             22.0388   |                  6.62984 |             94.1111 |                  0 |
|  2 | Examples as Interaction: On Humans Teaching a Computer to Play a Game                                   |         69 |            2 |              25 |                    0 |                     1 |               0     | This paper reviews an experiment...   |              736 |             16.3282   |                  8.48901 |            186.216  |                  0 |
|  3 | Similarit\'e en intension vs en extension : \`a la crois\'ee de l'informatique et du th\'e\^atre        |         96 |            2 |               0 |                    0 |                     0 |               0     | La mise en scène...|              6.6775   |                 16.7851  |            195.187  |                  0 |
|  4 | On The Power of Tree Projections: Structural Tractability of Enumerating CSP Solutions                  |         86 |            2 |              20 |                   45 |                     0 |             230.689 | The problem of deciding whether CSP instances admit solutions...  |             1184 |             -0.807016 |                  8.76605 |            134.859  |                  0 |
|  5 | Two-Timescale Learning Using Idiotypic Behaviour Mediation For A Navigating Mobile Robot                |         88 |            3 |              20 |                   18 |                     0 |             435.556 | A combined Short-Term Learning (STL) and Long-Term Learning (LTL) approach... |             1010 |             16.729    |                  9.52101 |            153.148  |                  0 |
|  6 | sTeX+ - a System for Flexible Formalization of Linked Data                                              |         58 |            3 |              28 |                   34 |                     0 |             166     | We present the sTeX+ system... |              859 |             22.578    |                  9.0382  |            132.05   |                  0 |
|  7 | Application of Data Mining to Network Intrusion Detection: Classifier Selection Model                   |         85 |            2 |              11 |                   22 |                     1 |             145.818 | As network attacks have increased in number and severity over the past few years, intrusion detection system (IDS)... |             1030 |              6.48818  |                  9.19555 |            142.371  |                  0 |
|  8 | Survey on Various Gesture Recognition Techniques for Interfacing Machines Based on Ambient Intelligence |        103 |            5 |              19 |                   24 |                     0 |             124.917 | Gesture recognition is mainly apprehensive... |             1675 |             25.3356   |                  8.41612 |            117.804  |                  0 |
|  9 | On Elementary Loops of Logic Programs                                                                   |         37 |            3 |              41 |                  112 |                     1 |             198.661 | Using the notion of an elementary loop... |             1138 |             18.5446   |                  8.54851 |            103.247  |                  0 |


### Image-based acceptance prediction 

Like the data collection for text-based prediction, we used the PeerRead dataset as "meta-data" for the papers' titles and acceptance. Inside PeerRead, we particularly use the papers from ICLR and arXiv.org e-Print achieve since they have fewer technical barriers, e.g., access control and rate-limiting, for crawling. 

We downloaded the raw PDF files of the papers on these venues, where <2% of PDFs are missed, probably revoked by the authors. We have no choice but to remove these papers from our dataset.  

The resulting data is >10GB and beyond our capacity for model training. To subsample the PDFs, we convert them into PNG images via pdf2image and then extract and merge the first 4 pages of each paper into a 224x244 image since we deem the first few pages are the most influential ones to the reviewers' impression. Also, it is hard to uniformly capture all pages because the numbers of pages are different, depending on the format requirement of the venues and the length of the appendix.   

The following is an example of the extracted images. The resolution is low but should be enough as J. Huang [5] 's image-based approach works well with images of the same resolution.

<center><img src="blurry_image.png" alt="Timeline Picture"/></center>

### Sub-domain classification 

We are using unsupervised learning methods to discover the sub-domain of a paper and cluster similar domains together. We will be using techniques like Bag of words, TF-IDF, and BERT encoding to cluster the papers based on the sub-domain. Till now, we have created Bag-of-words and TF-IDF encoding using the words from "Title", and "Abstract" of papers from our PeerRead Dataset.

## Methods
Our main idea is to capture the wordings in papers as features, most likely using natural language processing (NLP) techniques to transform the paper contents into (word embedding) vectors. Furthermore, we will combine them with some "meta-data" of the papers, e.g., the citations, the number of figures/equations, etc.

### Unsupervised Learning
**Unsupervised learning** techniques would help us discover similar sub-domains and recent popular research trends by performing clustering based on inclusion of keywords related to specific sub-domains. We use k-means clustering techniques to identify sub-domains by using various feature representations as follows

1. Bag-of-words (BOW)
2. Term Frequency–Inverse Document Frequency (TF-IDF)
3. Bidirectional Encoder Representations from Transformers (BERT)

We have implemented Bag-of-words, TF-IDF and BERT encoding using words from ‘Title’, and ‘Abstract’ sections of the paper. We are using scikit’s feature_extraction libraries to construct our BOW and TF-IDF encodings. 

### Bag-of-words (BOW)

We chose this model because it is the simplest numerical  representation of text. For constructing this, we are using scikit’s  CountVectorizer to tokenize the sentences from ‘Title’ and ‘Abstract’  into a matrix of token counts. Then we are using scikit’s  fit_transform() api to learn the vocabulary dictionary and return the  document-term matrix. We are using pandas dataframes to store this  matrix.

During preprocessing the data, we are excluding English stop words  and numbers from tokenizing because they do not contribute for analyzing the sub-domain.

There are total 11727 papers, including both training and testing data, and the encoding resulted in 29445 unique words. A sample of our BOW table is as follows:

```py
# Getting bag of words data structure
CountVec = CountVectorizer(ngram_range=(1,1), stop_words='english')
Count_data = CountVec.fit_transform(merged_data)
cv_dataframe=pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names_out())
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
display(cv_dataframe.head(5).loc[:,cv_dataframe.head(5).any()])

# Dimension reduction on BOW using SVD
svd = TruncatedSVD(n_components=500, n_iter=10, random_state=42)
svd.fit(Count_data)
X_new = svd.transform(Count_data)
X_train = X_new[0:len(train_data)]
X_test = X_new[len(train_data):]
```

As we can see, there are total 11727 papers, including both training and testing data, and the encoding resulted in 29445 unique words. 

A sample of our BOW table is as follows:
    
|    |   annotation |   answering |   attention |   base |   based |   bridge |   cross |   detection |   effective |   embedding |
|---:|-------------:|------------:|------------:|-------:|--------:|---------:|--------:|------------:|------------:|------------:|
|  0 |            0 |           0 |           0 |      0 |       0 |        1 |       0 |           0 |           0 |           1 |
|  1 |            0 |           0 |           1 |      0 |       0 |        0 |       0 |           0 |           0 |           0 |
|  2 |            1 |           0 |           0 |      0 |       0 |        0 |       1 |           0 |           1 |           0 |
|  3 |            0 |           0 |           0 |      0 |       1 |        0 |       0 |           0 |           0 |           0 |
|  4 |            0 |           1 |           0 |      1 |       0 |        0 |       0 |           1 |           0 |           0 |


### Dimensionality Reduction

Initially we clustered similar papers together by considering both  Title and Abstract sections of the papers for encoding and clustering.  The total number of unique words resulted in 29445 which is the number  of features in this case. As you can already tell, this is a lot of  features and we have to reduce them better and faster clustering.

We have tried out the following ways to overcome this problem and  will be continuing with these approach for all clustering/encoding  methods: 

#### Just use Title of the paper for clustering

By doing this our features drastically reduced to 9172 unique words  which is very reasonable for clustering. In this case we don't even have to explicitly use dimension reduction techniques because this is  computationally reasonable and also lesser than the number of samples we have.

#### Singular Value Decomposition (SVD)

Alternatively, we also experimented using SVD for dimension reduction. In this case as well, we considered words from the Title of  the papers because after exploring with and without Abstract we could  see that Title alone is reasonably sufficient for clustering similar  papers together. One big advantage of excluding Abstract now is that we  can further reduce the number of features much lesser than 9172 unique  words we have from the Title.

### Clustering on BOW model

Now, we will be using the BOW encoding for clustering similar  sub-domain papers together. For this we are using K-means clustering  algorithm and we are determining the optimal number of clusters using  the popular elbow curve method.

#### Without SVD

From the below elbow curve, we are choosing *𝐾*=6 as the optimal number of clusters for our final clustering model.

![Elbow method](./elbow_bow.png)

Inertia which is sum of squared distances of samples to their closest cluster center is also obtained once the training is done. Inertia in this case is 71404.58780199478.

#### With SVD

From the below elbow curve, we are choosing *𝐾*=6 as the optimal number of clusters for our final clustering model.

![Elbow method](./elbow_bow_svd.png)

Inertia in this case is 44204.41325844073.

### Clustering Analysis
#### Without SVD
To understand what each cluster signifies in terms of sub-domain, we are obtaining top terms per cluster. To analyze in a better way, word clouds for these are generated.
<center><img src="./wordcloud_bow_cluster0.png" alt="wordcloud"/></center>
<center><img src="./wordcloud_bow_cluster1.png" alt="wordcloud"/></center>
<center><img src="./wordcloud_bow_cluster2.png" alt="wordcloud"/></center>
<center><img src="./wordcloud_bow_cluster3.png" alt="wordcloud"/></center>
<center><img src="./wordcloud_bow_cluster4.png" alt="wordcloud"/></center>
<center><img src="./wordcloud_bow_cluster5.png" alt="wordcloud"/></center>

Once the model is trained, we are clustering the papers in our testing set and the sample prediction is as shown below.

|      | Title                                                        | Cluster ID |
| ---: | :----------------------------------------------------------- | ---------: |
|    0 | Evaluation Metrics for Machine Reading Comprehension: Prerequisite Skills and Readability |          1 |
|    1 | A Neural Local Coherence Model                               |          2 |
|    2 | Neural Modeling of Multi-Predicate Interactions for Japanese Predicate Argument Structure Analysis |          2 |
|    3 | Neural Disambiguation of Causal Lexical Markers based on Context |          2 |
|    4 | Chunk-based Decoder for Neural Machine Translation           |          3 |
|    5 | What do Neural Machine Translation Models Learn about Morphology? |          3 |
|    6 | Detecting Lexical Entailment in Context                      |          1 |
|    7 | Support Vector Machine Classification with Indefinite Kernels |          1 |
|    8 | The Parameterized Complexity of Global Constraints           |          1 |
|    9 | Examples as Interaction: On Humans Teaching a Computer to Play a Game |          1 |

#### With SVD
Now, clustering on BOW by reducing the number of features to 500 using SVD, the following is the sample prediction:

|      | Title                                                        | Cluster ID |
| ---: | :----------------------------------------------------------- | ---------: |
|    0 | Evaluation Metrics for Machine Reading Comprehension: Prerequisite Skills and Readability |          1 |
|    1 | A Neural Local Coherence Model                               |          3 |
|    2 | Neural Modeling of Multi-Predicate Interactions for Japanese Predicate Argument Structure Analysis |          3 |
|    3 | Neural Disambiguation of Causal Lexical Markers based on Context |          2 |
|    4 | Chunk-based Decoder for Neural Machine Translation           |          2 |
|    5 | What do Neural Machine Translation Models Learn about Morphology? |          3 |
|    6 | Detecting Lexical Entailment in Context                      |          1 |
|    7 | Support Vector Machine Classification with Indefinite Kernels |          1 |
|    8 | The Parameterized Complexity of Global Constraints           |          1 |
|    9 | Examples as Interaction: On Humans Teaching a Computer to Play a Game |          1 |

### TF-IDF Encoding

This model is a numeric statistic that is intended to reflect how  important a word is to a document. Term frequency is a measure of how  frequently a term appears in a document and IDF is a measure of how  important a term is. In contrast to BOW, this model derives information  on the most and least important words, and hence is expected to perform  better. Implementing this is like the BOW approach, except that we will  be using a different tokenizer for our data. We are using scikit’s  TfidfVectorizer with English stop-words to avoid commonly used English  words.

A sample of our TF_IDF encoding is as follows:

|      | annotation | answering | attention |     base |    based |   bridge |    cross | detection | effective | embedding |
| ---: | ---------: | --------: | --------: | -------: | -------: | -------: | -------: | --------: | --------: | --------: |
|    0 |          0 |         0 |         0 |        0 |        0 | 0.467153 |        0 |         0 |         0 |  0.299487 |
|    1 |          0 |         0 |  0.318779 |        0 |        0 |        0 |        0 |         0 |         0 |         0 |
|    2 |   0.330864 |         0 |         0 |        0 |        0 |        0 | 0.286734 |         0 |  0.330864 |         0 |
|    3 |          0 |         0 |         0 |        0 | 0.215444 |        0 |        0 |         0 |         0 |         0 |
|    4 |          0 |   0.36217 |         0 | 0.429501 |        0 |        0 |        0 |  0.306265 |         0 |         0 |

To cluster papers belonging to similar sub-domains together, we have implemented K-Means algorithm on TF-TDF encoding. 

```py
# Training the Model
# Getting tf-idf data structure
# Vectorize the text
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(merged_data)
X_train = X[0:len(train_data)]
X_test = X[len(train_data):]
tf_idf_df=pd.DataFrame(X_train.toarray(),columns=vectorizer.get_feature_names_out())
display(tf_idf_df.head(5).loc[:,tf_idf_df.head(5).any()])
```

### Clustering on TF-IDF model

Now, we will be using the TF-IDF encoding for clustering similar sub-domain papers together. For this we are using K-means clustering algorithm and we are determining the optimal number of clusters using the popular elbow curve method.

#### Without SVD
From the below elbow curve, we are choosing K=6 as the optimal number of clusters for our final clustering model.
<center><img src="./elbow_tfidf.png" alt="Elbow method"/></center>

Inertia which is sum of squared distances of samples to their closest cluster center is also obtained once the training is done. Inertia in this case is 10789.440479184834.

#### With SVD
From the below elbow curve, we are choosing K=6 as the optimal number of clusters for our final clustering model.
<center><img src="./elbow_tfidf_svd.png" alt="Elbow method"/></center>

Inertia in this case is 5179.698138369373.

### Clustering Analysis
#### Without SVD
To understand what each cluster signifies in terms of sub-domain, we are obtaining top terms per cluster. To analyze in a better way, word clouds for these are generated.
<center><img src="./wordcloud_tfidf_cluster0.png" alt="wordcloud"/></center>
<center><img src="./wordcloud_tfidf_cluster1.png" alt="wordcloud"/></center>
<center><img src="./wordcloud_tfidf_cluster2.png" alt="wordcloud"/></center>
<center><img src="./wordcloud_tfidf_cluster3.png" alt="wordcloud"/></center>
<center><img src="./wordcloud_tfidf_cluster4.png" alt="wordcloud"/></center>
<center><img src="./wordcloud_tfidf_cluster5.png" alt="wordcloud"/></center>

Once the model is trained, we are clustering the papers in our testing set and the sample prediction is as shown below.

|      | Title                                                        | Cluster ID |
| ---: | :----------------------------------------------------------- | ---------: |
|    0 | Evaluation Metrics for Machine Reading Comprehension: Prerequisite Skills and Readability |          3 |
|    1 | A Neural Local Coherence Model                               |          2 |
|    2 | Neural Modeling of Multi-Predicate Interactions for Japanese Predicate Argument Structure Analysis |          0 |
|    3 | Neural Disambiguation of Causal Lexical Markers based on Context |          4 |
|    4 | Chunk-based Decoder for Neural Machine Translation           |          3 |
|    5 | What do Neural Machine Translation Models Learn about Morphology? |          3 |
|    6 | Detecting Lexical Entailment in Context                      |          0 |
|    7 | Support Vector Machine Classification with Indefinite Kernels |          3 |
|    8 | The Parameterized Complexity of Global Constraints           |          0 |
|    9 | Examples as Interaction: On Humans Teaching a Computer to Play a Game |          0 |

#### With SVD
Now, clustering on TF-IDF by reducing the number of features to 500 using SVD, the following is the sample prediction:

|      | Title                                                        | Cluster ID |
| ---: | :----------------------------------------------------------- | ---------: |
|    0 | Evaluation Metrics for Machine Reading Comprehension: Prerequisite Skills and Readability |          0 |
|    1 | A Neural Local Coherence Model                               |          4 |
|    2 | Neural Modeling of Multi-Predicate Interactions for Japanese Predicate Argument Structure Analysis |          3 |
|    3 | Neural Disambiguation of Causal Lexical Markers based on Context |          1 |
|    4 | Chunk-based Decoder for Neural Machine Translation           |          0 |
|    5 | What do Neural Machine Translation Models Learn about Morphology? |          0 |
|    6 | Detecting Lexical Entailment in Context                      |          1 |
|    7 | Support Vector Machine Classification with Indefinite Kernels |          0 |
|    8 | The Parameterized Complexity of Global Constraints           |          1 |
|    9 | Examples as Interaction: On Humans Teaching a Computer to Play a Game |          1 |

### K-Means Clustering

To decide the optimal number of clusters, we have used the elbow method.

```py
# cluster documents
distortions = []
K = range(1,15)

for k in K:
    model = KMeans(n_clusters=k, init='k-means++', max_iter=1000, n_init=10)
    model.fit(X_train)
    distortions.append(model.inertia_)

plt.plot(K, distortions)
```
<center><img src="elbow.png" alt="Elbow method"/></center>

From the above curve, we chose 6 as our optimal number of clusters and trained our model using the training dataset from TF-IDF encoding. 
```py
# Choosing the best k from elbow method.
true_k = 6
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=10)
model.fit(X_train)
```
These are the top terms in each cluster: 
<center><img src="cluster_terms.png" alt="Top Cluster Terms"/></center>

Then, we used the test dataset for predicting its sub-domain cluster.
```py
# Testing the Model
# Predict the cluster association of each paper
prediction = model.predict(X_test)
print(X_train.shape)
print(X_test.shape)
df = pd.DataFrame(list(zip(test_title, prediction)), columns =['Title', 'Cluster ID'])
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
display(df)
```
A sample output of our clustering algorithm is as follows:

| | Title | Cluster ID |
|---:|:---------------------------------------------------------------------------------------------------|-------------:|
| 0 | Evaluation Metrics for Machine Reading Comprehension: Prerequisite Skills and Readability | 0 |
| 1 | A Neural Local Coherence Model | 1 |
| 2 | Neural Modeling of Multi-Predicate Interactions for Japanese Predicate Argument Structure Analysis | 2 |
| 3 | Neural Disambiguation of Causal Lexical Markers based on Context | 0 |
| 4 | Chunk-based Decoder for Neural Machine Translation | 3 |
| 5 | What do Neural Machine Translation Models Learn about Morphology? | 3 |
| 6 | Detecting Lexical Entailment in Context | 2 |
| 7 | Support Vector Machine Classification with Indefinite Kernels | 5 |
| 8 | The Parameterized Complexity of Global Constraints | 4 |
| 9 | Examples as Interaction: On Humans Teaching a Computer to Play a Game | 0 |

### BERT Encoding
#### Without SVD
BERT stands for Bidirectional Encoder Representations from Transformers. BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

We are using sentence-transformers package which wraps the Huggingface Transformers library. For sentence, we are using a pre-trained model 'distilbert-base-nli-stsb-mean-tokens'.

We are then using this transformation for K-means clustering to group similar papers together, and as before we use elbow method to find the optimal number of clusters needed which happens to be 6 in this case.

<center><img src="./elbow_bert.png" alt="Elbow method"/></center>

The number of features after BERT encoding happens to be 768, and the total inertia for clustering is 1512914.25. <br>
The following is the sample clustering of 10 papers from our testing dataset:

|      | Title                                                        | Cluster ID |
| ---: | :----------------------------------------------------------- | ---------: |
|    0 | Evaluation Metrics for Machine Reading Comprehension: Prerequisite Skills and Readability |          5 |
|    1 | A Neural Local Coherence Model                               |          0 |
|    2 | Neural Modeling of Multi-Predicate Interactions for Japanese Predicate Argument Structure Analysis |          2 |
|    3 | Neural Disambiguation of Causal Lexical Markers based on Context |          0 |
|    4 | Chunk-based Decoder for Neural Machine Translation           |          0 |
|    5 | What do Neural Machine Translation Models Learn about Morphology? |          0 |
|    6 | Detecting Lexical Entailment in Context                      |          5 |
|    7 | Support Vector Machine Classification with Indefinite Kernels |          5 |
|    8 | The Parameterized Complexity of Global Constraints           |          5 |
|    9 | Examples as Interaction: On Humans Teaching a Computer to Play a Game |          5 |

#### With SVD
As the number of features are already low, there is no need for dimension reduction. However for consistency we are reducing the dimensions to 500 using SVD.
<center><img src="./elbow_bert_svd.png" alt="Elbow method"/></center>

The total inertia for clustering is 1490288.75. <br>
The following is the sample clustering of 10 papers from our testing dataset:

|      | Title                                                        | Cluster ID |
| ---: | :----------------------------------------------------------- | ---------: |
|    0 | Evaluation Metrics for Machine Reading Comprehension: Prerequisite Skills and Readability |          0 |
|    1 | A Neural Local Coherence Model                               |          4 |
|    2 | Neural Modeling of Multi-Predicate Interactions for Japanese Predicate Argument Structure Analysis |          3 |
|    3 | Neural Disambiguation of Causal Lexical Markers based on Context |          1 |
|    4 | Chunk-based Decoder for Neural Machine Translation           |          0 |
|    5 | What do Neural Machine Translation Models Learn about Morphology? |          0 |
|    6 | Detecting Lexical Entailment in Context                      |          1 |
|    7 | Support Vector Machine Classification with Indefinite Kernels |          0 |
|    8 | The Parameterized Complexity of Global Constraints           |          1 |
|    9 | Examples as Interaction: On Humans Teaching a Computer to Play a Game |          1 |

### Analysis of the results

We have experimented K-means clustering to group similar papers  together using 3 different encodings, namely BOW, TF-IDF, and BERT. We  have also tried to reduce the dimensions using SVD. 

The purpose of clustering similar papers together was to analyze which sub-domains have a better acceptance rate. To show this, we use the ground truth of paper acceptance and the clustering assignments. Below is one such observation we made with BOW encodings before SVD:

| Cluster # | Acceptance Rate |
|-----------|-----------------|
| 0         | 22 %            |
| 1         | 10 %            |
| 2         | 28 %            |
| 3         | 50 %            |
| 4         | 33 %            |
| 5         | 27 %            |


Conceptually the  encodings these models do for transforming strings to numbers is  completely different and hence we cannot definitely compare these as we  do not have the ground truth in this case.

Numerically, inertia can be used to compare the distances but we  cannot really say which clustering is better just based on the following numbers.

| Encoding | # features before SVD | # features after SVD | Inertia without SVD | Inertia after SVD |
| -------- | --------------------- | -------------------- | ------------------- | ----------------- |
| BOW      | 9172                  | 500                  | 71404               | 44204             |
| TF-IDF   | 9172                  | 500                  | 10789               | 5179              |
| BERT     | 768                   | 500                  | 1512914             | 1490288           |

But there are few observations we made: 

1. BOW and TF-IDF encodings result in very high dimensions compared  to BERT, hence BOW and TF-IDF models result in sparse matrix whereas BERT does not have this issue.
2. Ideally we expect BERT clustering to be better because BERT is  context-dependent encoding, but due to the lack of ground truth on sub-domains we do not have a definitive way to prove it.

#### K-means Clustering on Visual Embeddings from CNNs
We also cluster the papers based on their visual appearance, aiming to discover features that might be useful for other ML models. Soon to be introduced below, ResNet is a CNN that captures visual features and transforms them into vectors for a linear classifier, i.e., the final fully connected layer, for prediction. We regard the vectors feeding to the linear classifier as the visual embeddings of the input papers, which has 512 dimensions and use these embeddings to run a K-means clustering.
Below is the plot of the inertia of the K-mean clustering versus K, the number of clusters. By the elbow method, a good K is 7 as the inertia decreases relatively slowly since then. 

<center><img src="visual_kmean_elbow.png" alt="Accuracy Graph"/></center>

##### Results and Conclusion on Image-based Clustering

Below are the top 5 papers images in each cluster closest to their centroids. We present the 5 clusters with the least loss (the sum of L2-distance to all data points in this cluster. Each row is for a cluster. The papers in each cluster appear to have a similar format, e.g., on the titles, author lists, and column margins. Thus, the clustering is likely differentiating the format of the papers, which is specified by the conferences. 

<center><img src="visual_kmean_top.png" alt="Accuracy Graph"/></center>


Although differentiating the paper's format may not be useful for acceptance prediction, it may be helpful for other related tasks. One is predicting the intended venue of a paper that appeared on an online pre-print or archive, e.g., arVix, to automate their upload pipeline and provide a better user experience. It may also help a paper search engine classify the paper's venue when not provided.


### Supervised Learning
**Supervised learning** techniques help test our hypothesis about the factors and paper acceptance. Hopefully, we may discover some hidden factors that affect paper acceptance.

#### Image-based Classification via Convolutional Neural Network (CNN)

We trained a CNN for acceptance prediction, which inputs the sub-sampled image of a paper’s PDF and outputs a binary prediction of whether the paper is accepted. As a starter, we picked arXiv’s machine learning papers as our dataset, which has 3940 papers for training and 205 for testing.   

In addition to the sub-sampling, we also normalize the dataset to 0 mean and 1 standard variance. We used the pre-trained ResNet-18 by PyTorch and changed the last fully connected layer to a new one with only 2 output units. We fine-tuned all layers with our training dataset in 50 epochs via SGD with learning rate=0.001, momentum = 0.9, weight decay = 0.01, and we decayed the learning rate by a factor of 0.1 every 10 epochs.   

We twisted our loss function to combat the label-imbalance issue in our data because most papers presented on arXiv are likely to be rejected (by their indented conferences). To prevent the classifier from blindly skewing to one label, namely, rejection, we set weights of each label in our loss function equal to the inverse of their ratio in the training dataset.   

The following chart presents the accuracy trajectory during the training, where the highest test accuracy is 83.2%. Although the model overfits after a few epochs, the model demonstrates non-trivial performance, better than the baseline indicated by the dotted lines on 68%, which is the ratio of the rejected papers to the total paper. It means that, if we skim the paper layout without diving into the content, we can make an educated guess on the paper acceptance.

<center><img src="resnet_acc.png" alt="Accuracy Graph"/></center>

As the training accuracy is much higher than the test accuracy, our ResNet is overfitting. To mitigate overfitting, we added dropout layers and changed the frozen layers during the fine-tuning process. Yet, the test accuracy is at most different by a few percentage points.

The accuracy of ResNet-18 is decent, but it is still not perfect. We suspect two possible reasons: 1) ResNet is not good enough CNN architecture for the acceptance prediction, or 2) the visual features are not enough for our tasks.

To figure out the reasons, we also fine-tune a pre-trained VGG-11 with BatchNorm for our tasks. Yet, the resulting accuracy is similar. As shown below, the best attain accuracy is 81.2%, and the overfitting occurs. We are thus convinced that by just looking at the layout of the papers, the CNN classifiers can at best have decent but not impressive accuracy. 

<center><img src="vgg_acc.png" alt="Accuracy Graph"/></center>

##### Conclusion on the Performance of CNNs

Below are the normalized confusion matrices of our ResNet and our VGG, respectively. ResNet, on the top, appears to have a high recall (~90%) but a low true negative rate. VGG, on the other hand, has a slightly worse recall but marginally better true negative rate. Our ResNet is better at catching accepted papers, while our VGG is better at catching rejected papers. 

ResNet:
<center><img src="confusion_resnet.png" alt="Accuracy Graph"/></center>

VGG:
<center><img src="confusion_vgg.png" alt="Accuracy Graph"/></center>

##### Activation Heatmap: Where the NNs Look at and What We can Learn from it.

The CNNs can give us insight into what a good or bad paper looks like. We visualize the activation heatmap of our CNN to see where the NNs look when they predict the acceptance of a paper. In other words, we locate the areas in the input images that most impact the final CNN output.

To ensure we are not visualizing the heatmap where our CNNs make random guesses, we cherry-picked the test images with the least loss in our CNNs. It means our NNs can confidently give the correct prediction on them. 

Following are the activation heatmaps of the “bad” papers that our ResNet correctly predicts with the least loss. The brighter the area is, the more influence it has to the prediction. Our ResNet pays the most attention to the figures inside the papers, and these images occupy a large proportion of the first few pages. Thus, a rule of thumb for writing a good paper is perhaps not to put too many large images in the first few pages.

<center><img src="bad_heatmap_1.png" alt="Accuracy Graph"/></center>
<center><img src="bad_org_1.png" alt="Accuracy Graph"/></center>
<center><img src="bad_heatmap_2.png" alt="Accuracy Graph"/></center>
<center><img src="bad_org_2.png" alt="Accuracy Graph"/></center>

Below are the activation heatmaps of the “good” paper from our ResNet. Interestingly, the NN looks mostly in the margins of the paragraphs. It makes sense because accepted papers are usually well-engineered to comply with the conference format with good utilization of the text space to deliver more content. The lesson learned here for producing a good paper is to format the paper, especially the margin space, properly.

<center><img src="good_heatmap_1.png" alt="Accuracy Graph"/></center>
<center><img src="good_org_1.png" alt="Accuracy Graph"/></center>
<center><img src="good_heatmap_2.png" alt="Accuracy Graph"/></center>
<center><img src="good_org_2.png" alt="Accuracy Graph"/></center>

For VGG, we cannot provide such fine-grained activation heatmaps in the input as the architecture does not support them.

#### Text-based Classification
We will be using the labelled dataset to train supervised algorithms to predict the acceptance of a paper. There are a few prior works which propose the following algorithms:
1.	Naive Bayes
2.	K-Nearest Neighbor
3.	Logistic Regression
4.	Decision Tree
5.	Random Forest
6.	Support Vector Machine
7.	Neural-network (Multi-Layer Perceptron)

We have implemented each of these algorithms, fine-tuned their parameters to improve accuracy, and analyzed the relation between sub-domains and the classification algorithm that performs best for each one.

For each of these algorithms, we noticed that using the data available from the PeerRead database directly did not yield useful results because most of the papers were rejected; hence, the models were rewarded with high test accuracy if they simply guessed "reject". We compensated for this by combining all the prior training and testing data, and randomly sampling data corresponding to both rejected and accepted papers such that approximately 55% were rejected and 45% were accepted. The entire dataset from which paper data was sampled for text-based classification consisted of the 3 Arxiv datasets.

##### 1. Naive Bayes

Naive Bayes is a simple method for constructing classifiers i.e. assign class labels to problem instances, represented as feature vectors such as BOW, TF-IDF, BERT etc. Naive Bayes classifiers rely on the assumption that a particular feature is independent of all other features. Despite this naive design and simple approach, naive Bayes approach works quite well in most cases. An advantage of naive Bayes is that it requires only a small number of training data to estimate the parameters for classification, which is a especially useful in data-constrained settings. The probabilistic nature of the method makes it suitable for our use case.

Naive Bayes performed the worst, by far, of all the algorithms, implying that naive probabilistic relationships were not sufficient to model the relationship between the features we used and paper acceptance. The test accuracy was a staggeringly low 51.96%.

The accuracies, false vs. true positive graph, and confusion matrix for running Naive Bayes on our data are shown below.

<center><img src="naive_bayes_graphs.png" alt="Naive Bayes Graphs"/></center>

##### 2. K-Nearest Neighbor (KNN)

The K-Nearest Neighbor (KNN) algorithm is a type of supervised learning algorithm used for both regression and classification tasks. KNN tries to predict the  correct class for the test data by calculating the distance between the  test data and all the training points in feature space. It essentially relies on the assumption that similar things exist in close proximity to each other. Hence feature selection becomes crucial for good performance. 

KNN performed modestly, with its best test accuracy being 65.03% when ran with 41 neighbors.

The accuracies, false vs. true positive graph, and confusion matrix for running KNN on our data are shown below.

<center><img src="knn_graph.png" alt="KNN Graphs"/></center>

Additionally, the test accuracy for each of the number of neighbors for which we ran KNN is plotted below.

<center><img src="knn_hyperparam_graph.png" alt="KNN Hyperparameter Graphs"/></center>

##### 3. Logistic Regression

Binary Logistic regression is a supervised classification method used to predict the probability of a target variable. The nature of output is dichotomous, i.e. there would be two possible classes. In other words, the output is binarized i.e outputting either 1 or 0 only. Mathematically, a logistic regression model predicts P(Y=1) as a  function of X. It is one of the simplest ML algorithms that can be used  for various classification problems.

Logistic Regression performed modestly, with a test accuracy of 64.54%.

The accuracies, false vs. true positive graph, and confusion matrix for running Logistic Regression on our data are shown below.

<center><img src="logistic_graph.png" alt="Logistic Regression Graphs"/></center>

##### 4. Decision Tree

Decision Trees can be thought of as non-parametric supervised classification method. It aims to create a model that predicts an output by learning simple decision rules constructed from the data features. It is called a tree as these decision rules can be structurally be represented as a tree.

Using 1 Decission Tree with an unbounded depth performed modestly, with a test accuracy of 62.66%. As you can see from the training accuracy, there was total overfitting to the training data, suggesting that pruning would improve the test accuracy. This was done in the next method, random forest.

The accuracies, false vs. true positive graph, and confusion matrix for running one unruned Decision Tree on our data are shown below.

<center><img src="decision_tree_graph.png" alt="Decision Tree Graph"/></center>

##### 5. Random Forest

The Random forest is a classification algorithm consisting of many randomly constructed decisions trees. It uses bagging and feature randomness when building each individual tree to create an uncorrelated forest (ensemble) of trees whose prediction by committee is more accurate than that of any individual tree. It is effective in many scenarios and can be used for various classification problems.

Random Forest performed the best out of all supervised methods for text-based classification. We suspect this is because our features consisted of several metrics for which the relationship was not immediately clear via other means, so splitting along them as random forests do among many decision trees yielded the best way to partition our data according to acceptance.

The best test acceptance for random forest, and indeed of all the text-based models, was 70.42%, which was acheived by running Random Forest with 200 estimators and a maximum depth of 9.

The best accuracies, false vs. true positive graph, and confusion matrix for running Random Forest on our data are shown below.

<center><img src="random_forest_graph.png" alt="Random Forest Graph"/></center>

Additionally, the test accuracy for each of the maximum depths for which we ran Random Forest is plotted below.

<center><img src="random_forest_hyperparam_graph.png" alt="Random Forest Hyperparameter Graph"/></center>


##### 6. Support Vector Machine (SVM)

Support Vector Machine (SVM) are a class of supervised classification algorithms. In the SVM algorithm, we plot each data item as a point in n-dimensional feature space with the value of each feature being the value of a particular coordinate. The learning component aims to perform classification by finding the hyper-plane that differentiates the two classes in the best possible way.

SVM performed very well, almost as well as the top supervised learning method, with a test acceptance of 68.55%.

The accuracies, false vs. true positive graph, and confusion matrix for running SVM on our data are shown below.

<center><img src="svm_graph.png" alt="SVM Graph"/></center>


##### 7. Neural-network (Multi-Layer Perceptron)

Multi-layer Perceptron (MLP) is a supervised learning algorithm that learns a function by training on a dataset, it can be used to train and classification model. Given a set of features and a target classes (here only two) , it can learn a non-linear function approximator. It is different from logistic regression, as there can be one or more non-linear layers usually referred to as hidden layers. We explore several variations of the hyper-parameters available to find the best one.

The neural network performed very well, almost as well as the top supervised learning method, with a test acceptance of 69.11% for the one with the best parameters of an initial learning rate of 0.01 and an initial hidden layer size of 15.

The best accuracies, false vs. true positive graph, and confusion matrix for running SVM on our data are shown below.

<center><img src="neural_net_graph.png" alt="Neural Net Graph"/></center>

Additionally, the test accuracy for running the neural network on a variety of initial learning rates with an initial hidden layer of size 5, and subsequently running it on a variety of initial hidden layer sizes using the best initial learning rate, is plotted below.

<center><img src="neural_net_hyperparam_graph1.png" alt="Neural Net Hyperparameter Graph"/></center>
<center><img src="neural_net_hyperparam_graph2.png" alt="Neural Net Hyperparameter Graph"/></center>


#### Supervised Approaches Conclusion

Of all the supervised learning methods ran for text-based acceptance prediction, Naive Bayes performed the worst, and Random Forest performed the best for 200 estimators and a max. depth of 9. This suggests that, for the numerical features we selected from the papers, naive probabilistic relationships between the features yield little information, whereas splitting the data in a tree-like fashion, pruning, and averaging over many cases yields the optimal way of partitioning paper features for acceptance prediction.

## Future Results and Discussion

We shall evaluate the model on the curated test sets to determine the model's effectiveness in predicting paper quality. Apart from tracking the accuracy of our model we also aim to visualize our analysis with appropriate forms of charting such as bar plots, confusion matrices, and cluster plots. These visual tools would aid us in not only conveying our findings but would also allow us to make iterative improvements to our model by exposing interesting trends and features. It remains interesting to see how an ensemble of domain-specific models could improve the predictive capability of our approach.

## Work Division
We have planned individual members’ responsibility as follows. However, we are planning to make changes to it if needed as we are not exactly sure about the workload of sub-tasks. Additionally, we will be assigning weekly tasks to all the members and will be syncing up weekly to make sure all of us are progressing.

<center><img src="work_division.png" alt="Timeline Picture"/></center>

## Timeline
<center><img src="timeline.png" alt="Timeline Picture" /></center>

## References
[1] Dongyeop Kang, Waleed Ammar, Bhavana Dalvi, Madeleine van Zuylen, Sebastian Kohlmeier, Eduard Hovy, Roy Schwartz.
_A Dataset of Peer Reviews (PeerRead): Collection, Insights and NLP Applications_
North American Chapter of the Association for Computational Linguistics 2018.
[https://arxiv.org/abs/1804.09635](https://arxiv.org/abs/1804.09635)

[2] Deepali J. Joshi, Ajinkya Kulkarni, Riya PandeIshwari Kulkarni Siddharth Patil, Nikhil Saini. _Conference Paper Acceptance Prediction: Using Machine Learning_. Machine Learning and Information Processing 2021. [https://link.springer.com/chapter/10.1007/978-981-33-4859-2_14](https://link.springer.com/chapter/10.1007/978-981-33-4859-2_14)

[3] Yuxiao Dong, Reid A. Johnson, Nitesh V. Chawla. _Can Scientific Impact Be Predicted_. IEEE Transactions on Big Data 2016. [https://arxiv.org/abs/1606.05905](https://arxiv.org/abs/1606.05905)

[4] Qingyun Wang, Qi Zeng, Lifu Huang, Kevin Knight, Heng Ji, Nazneen Fatema Rajani. _ReviewRobot: Explainable Paper Review Generation based on Knowledge Synthesis_. International Conference on Natural Language Generation 2020. [https://arxiv.org/abs/2010.06119](https://arxiv.org/abs/2010.06119)

[5] Jia-Bin Huang. _Deep Paper Gestalt_. Computer Vision and Pattern Recognition 2018. [https://arxiv.org/pdf/1812.08775.pdf](https://arxiv.org/pdf/1812.08775.pdf)
