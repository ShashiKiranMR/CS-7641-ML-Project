# CS 7641 ML Project
## Introduction
The number of paper submissions in ML conferences is on the rise and outpacing the availability of good reviewers to evaluate them. Having a reliable way to quantify a paper's quality in terms of probability of acceptance could allow one to better allocate the papers among reviewers while also having a better metric for desk rejection of poor-quality papers. Additionally, such a model could also be used to aid authors to instantly evaluate and subsequently improve their paper’s chances of acceptance. Self-evaluation could also discourage authors from submitting poor-quality papers, hence leading to better quality paper submissions overall. Hence, such a model could have double benefits in terms of supporting the scientific community.

Besides the hard-to-measure aspects such as the novelty and the quality of research, many other measurable factors play a vital role in the paper review process. Good presentation is necessary for reviewers to appreciate the papers. The popularity of the paper's research sub-domain may also affect its acceptance.
## Problem Definition
We aim to build a predictive model to output a paper’s chance of acceptance. Further, we also aim to discern the prominent factors affecting a paper’s acceptance in each research domain. 
## Dataset
We use the PeerRead dataset [1] as ground truth, which consists of 14K papers, 10K actual reviews, and accept/reject decisions in ML top-tier conferences. We may collect additional data if needed.
## Methods
Our main idea is to capture the wordings in papers as features, most likely using natural language processing (NLP) techniques to transform the paper contents into (word embedding) vectors. Furthermore, we will combine them with some "meta-data" of the papers, e.g., the citations, the number of figures/equations, etc.

**Unsupervised learning** techniques would help us discover similar sub-domains and recent popular research trends by performing clustering based on inclusion of keywords related to specific sub-domains. Clustering techniques include:
1. k-means clustering
2. Gaussian mixture model clustering
3. Density-based clustering

While we aim to select sub-domains a priori, and perform clustering solely based on the papers’ contents, identifying keywords relevant to sub-domains. We will also reduce the number of relevant features for paper acceptance classification by using the following:
1. Principal Component Analysis
2. Independent Component Analysis

**Supervised learning** techniques help test our hypothesis about the factors and paper acceptance. Hopefully, we may discover some hidden factors that affect paper acceptance.
We will be using the above labelled dataset to train supervised algorithms to predict the acceptance of a paper. There are a few prior works which propose the following algorithms:
1.	Naive Bayes
2.	K-Nearest Neighbor
3.	Logistic Regression
4.	Decision Tree
5.	Random Forest
6.	Support Vector Machine

We will implement these algorithms and try to improve accuracy and analyze the relation between sub-domains and the classification algorithm that performs best for each one.
## Potential Results and Discussion

We shall evaluate the model on the curated test sets to determine the model's effectiveness in predicting paper quality. Apart from tracking the accuracy of our model we also aim to visualize our analysis with appropriate forms of charting such as bar plots, confusion matrices, and cluster plots. These visual tools would aid us in not only conveying our findings but would also allow us to make iterative improvements to our model by exposing interesting trends and features. It remains interesting to see how an ensemble of domain-specific models could improve the predictive capability of our approach.

## Work Division
We have planned individual members’ responsibility as follows. However, we are planning to make changes to it if needed as we are not exactly sure about the workload of sub-tasks. Additionally, we will be assigning weekly tasks to all the members and will be syncing up weekly to make sure all of us are progressing.
| Presentations & Reports | Data Collection & Preprocessing | Unsupervised Learning | Supervised Learning           |
|-------------------------|---------------------------------|-----------------------|-------------------------------|
| All | All | Andrew, Ka Lok Ng |Shashi Kiran, Rakesh, Changseok|

## Timeline
<img src="timeline.png" alt="Timeline Picture" style="float: left; margin-right: 10px;" />
## References
[1] https://github.com/allenai/PeerRead

[2] https://link.springer.com/chapter/10.1007/978-981-33-4859-2_14

[3] https://arxiv.org/abs/1606.05905

[4] https://aclanthology.org/2020.inlg-1.44.pdf

[5] https://arxiv.org/pdf/1812.08775