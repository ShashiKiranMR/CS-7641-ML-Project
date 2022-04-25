import sys
import os

import numpy as np
import pandas as pd
import json, os
from pathlib import Path
import matplotlib.pyplot as plt
import re
import syllables
from nltk import word_tokenize, sent_tokenize, download
download('punkt')

import re
dale_chall_words = open('dale_chall_words.txt', 'r').read()
dale_chall_words = set(dale_chall_words.lower().split(','))

def get_all_json(path):
  dev_parsed = {}
  dev_reviews = {}
  test_parsed = {}
  test_reviews = {}
  train_parsed = {}
  train_reviews = {}
  for subdir, dirs, files in os.walk(path):
    for f in files:
      path_name = os.path.join(subdir, f)
      id = '.'.join([part for part in path_name.split('/')[-1].split('.') if part.isnumeric()])
      if path_name.endswith('.pdf.json'):
        if 'dev' in path_name:
          dev_parsed[id] = json.load(open(path_name, 'r'))
        elif 'test' in path_name:
          test_parsed[id] = json.load(open(path_name, 'r'))
        elif 'train' in path_name:
          train_parsed[id] = json.load(open(path_name, 'r'))
      elif path_name.endswith('.json'):
        if 'dev' in path_name:
          dev_reviews[id] = json.load(open(path_name, 'r'))
        elif 'test' in path_name:
          test_reviews[id] = json.load(open(path_name, 'r'))
        elif 'train' in path_name:
          train_reviews[id] = json.load(open(path_name, 'r'))
  return dev_parsed, dev_reviews, test_parsed, test_reviews, train_parsed, train_reviews

def complexity_scores(text):
  words = [w for w in word_tokenize(text) if re.search("^.*(\\d|[A-z]).*$", w)]
  syllable_counts = np.array([syllables.estimate(w) for w in words])
  sentences = sent_tokenize(text)
  difficult_words = [w for w in words if re.sub("[^a-z']", '', w.lower()) not in dale_chall_words]
  flesch_score = None 
  dale_chall_score = None
  if len(words) > 0:
    flesch_score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (np.sum(syllable_counts) / len(words))
    flesch_score = max(flesch_score, -40)
    dale_chall_score = 15.79 * (len(difficult_words) / len(words)) + 0.0496 * (len(words) / len(sentences))
  return flesch_score, dale_chall_score, sentences

## create Dataframes
def create_dataframe(json_data, json_reviews, dataset_name=''):
  modified_arr = []
  for id in json_data.keys():
    fields = {'id': id}
    # fields of metadata are 'source', 'title', 'authors', 'emails', 'sections', 'references', 'referenceMentions', 'year', 'abstractText', 'creator'
    # fields of review_data are 'reviews', 'abstract', 'histories', 'id', 'title'
    metadata = json_data[id]['metadata']
    review_data = json_reviews[id]
    fields['title'] = review_data['title']
    fields['titleLen'] = len(review_data['title'])
    authors = []
    if 'authors' in review_data.keys() and len(review_data['authors']) > 0:
      authors = review_data['authors']
    elif 'authors' in metadata.keys() and len(metadata['authors']) > 0:
      authors = metadata['authors']
    fields['numAuthors'] = len(authors)
    fields['numReferences'] = len(metadata['references'])
    fields['numCitedReferences'] = len(metadata['referenceMentions'])
    fields['numRecentReferences'] = len([reference for reference in metadata['references'] if reference['year'] == metadata['year']])
    fields['avgCitedRefLength'] = 0 if len(metadata['referenceMentions']) == 0 else np.mean([len(citedReference['context']) for citedReference in metadata['referenceMentions']])
    # to add a field to the dataframe, simply add an entry to the fields dictionary in this for loop. All data can be found in metadata and review_data, whose respective fields
    # for the acl dataset are listed above. When using a new field, ensure it works across all . If you need to access a field specific to a particular dataset, use the
    # dataset_name parameter, which will be supplied accordingly.
    abstract = review_data['abstract']
    if metadata['abstractText'] and len(metadata['abstractText']) > len(abstract):
      abstract = metadata['abstractText']
    fields['abstract'] = abstract
    fields['abstractLength'] = len(abstract)

    abstract_flesch, abstract_dale_chall, _ = complexity_scores(abstract)
    fields['abstractFleschScore'], fields['abstractDaleChallScore'] = abstract_flesch, abstract_dale_chall

    allSections = metadata['sections'] if metadata['sections'] else []
    if len(allSections) != 0:
      all_sections_text = '. '.join([section['text'] for section in allSections])
      # sections_flesch, sections_dale_chall, sections_sentences = complexity_scores(all_sections_text)
      sections_sentences = sent_tokenize(all_sections_text)
      fields['avgSentenceLength'] = sum([len(sentence) for sentence in sections_sentences])/len(sections_sentences)
      fields['mentionsAppendix'] = int(type(all_sections_text) == ''.__class__ and 'appendix' in all_sections_text.lower())
      # fields['allSectionsFleschScore'], fields['allSectionsDaleChallScore'] = sections_flesch, sections_dale_chall
    else:
      fields['avgSentenceLength'] = 0
      fields['mentionsAppendix'] = 0
    
    ### GT and subjective GT section
    ## average reviewer score (weighted by confidence)
    if(len(review_data['reviews'])>0):
        wtSumRecomm = 0
        wtSumOrig = 0
        wtSumAppro = 0
        wtSumImpact = 0
        wtSumClarity = 0
        totalWt = 0
        totalRevs = len(review_data['reviews'])
        for rev in range(totalRevs):
          currRev = review_data['reviews'][rev]
          if 'REVIEWER_CONFIDENCE' in currRev.keys():
            wt = float(currRev['REVIEWER_CONFIDENCE'])
            totalWt += wt
            if 'RECOMMENDATION' in currRev.keys():
              wtSumRecomm += wt*float(currRev['RECOMMENDATION'])
            if 'ORIGINALITY' in currRev.keys():
              wtSumOrig += wt*float(currRev['ORIGINALITY'])
            if 'APPROPRIATENESS' in currRev.keys():
              wtSumAppro += wt*float(currRev['APPROPRIATENESS'])
            if 'IMPACT' in currRev:
              wtSumImpact += wt*float(currRev['IMPACT'])
            if 'SUBSTANCE' in currRev:
              wtSumImpact += wt*float(currRev['SUBSTANCE'])
            if 'CLARITY' in currRev:      
              wtSumClarity += wt*float(currRev['CLARITY'])
        if totalWt > 0:
          fields['avgReviewerConf'] = wtSumRecomm / totalWt   ## avg reviewer score
          fields['avgOrig'] = wtSumOrig / totalWt     ## avg novelty (unique)
          fields['avgAppro'] = wtSumAppro / totalWt   ## avg domain closeness conf of reviewers
          fields['avgImpact'] = wtSumImpact / totalWt    ## avg path breaking-ness/novelty (quality)
          fields['avgClarity'] = wtSumClarity / totalWt    ## avg clarity(subjective quality of idea in a way)
    else:
        fields['avgReviewerConf'] = None
        fields['avgOrig'] = None
        fields['avgAppro'] = None
        fields['avgImpact'] = None
        fields['avgClarity'] = None        
    
    ## labels    
    if 'accepted' in review_data.keys():
        fields['accepted'] = int(review_data['accepted'])
    elif fields['avgReviewerConf'] is not None:
        avg_score = fields['avgReviewerConf']    
        fields['accepted'] = int( avg_score >= 3 ) 
    else:
        fields['accepted'] = None

    modified_arr.append(fields)

  return modified_arr #pd.DataFrame(modified_arr)

def create_label_df(review_path, testing_df=None):
  review_path = review_path
  all_filename = os.listdir(review_path)
  review_json = list(filter(lambda x: Path(x).suffix == '.json', all_filename)) 
  all_id = [x[:-5] for x in review_json]
  review_path = [os.path.join(review_path, x) for x in review_json]
  accepted_lst = []
  for i, this_path in enumerate(review_path):
    with open(this_path, 'r') as file:
      this_json = json.load(file)
      paper_id = all_id[i]
      accepted = None
      if 'accepted' in this_json.keys():
        accepted = this_json['accepted']
      elif testing_df is not None:
        avg_score = testing_df.loc[testing_df['id'] == paper_id]['avgReviewerConf'].values
        if avg_score is not None:
          accepted = len(avg_score) > 0 and avg_score[0] >= 3
      accepted_lst.append(int(accepted))
  d = {'paper_id': all_id, 'accepted': accepted_lst}
  #df = pd.DataFrame.from_dict(d)

  return d #df

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    return text

'''
paths = {
  'acl': '../PeerRead/data/acl_2017',
  'cs_ai': '../PeerRead/data/arxiv.cs.ai_2007-2017',
  'cs_cl': '../PeerRead/data/arxiv.cs.cl_2007-2017',
  'cs_lg': '../PeerRead/data/arxiv.cs.lg_2007-2017',
  'conll': '../PeerRead/data/conll_2016',
  'iclr': '../PeerRead/data/iclr_2017',
  'nips': '../PeerRead/data/nips_2013-2017'
}

dataset_names = ['acl', 'cs_ai', 'cs_cl', 'cs_lg', 'conll', 'iclr']
dataset_data = [get_all_json(paths[s]) for s in dataset_names]

### dataframe 
# 637 test
# 11,090 train

parent_paths = [paths[s] for s in dataset_names]
review_paths = sum([[path + '/dev/reviews', path + '/test/reviews', path + '/train/reviews'] for path in parent_paths], [])

testing_dfs = [create_dataframe(data[4], data[5], dataset_name=i) for i,data in enumerate(dataset_data)]
training_dfs = [create_dataframe(data[2], data[3]) for data in dataset_data]
label_dfs = [create_label_df(review_path,testing_df=testing_dfs[i//3]) for i, review_path in enumerate(review_paths)]

#testing_dfs[1]
'''