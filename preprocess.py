#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:54:54 2020

@author: anooppanyam
"""

from __future__ import division
from os import path
from collections import Counter
import re
import string

try:
   import cPickle as pickle
except:
   import pickle

import numpy as np
   
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from verbalexpressions import VerEx

__file__ = 'preprocess.py'

PROJECT_DIR = (path.dirname(path.abspath(__file__)))
STOPWORDS = stopwords.words('english')

class TextProcessor(object):
    """
    Process raw yelp review data 
        Added features:
            - (avg) review sentiment
            - (avg) word count
            - (avg) punctuation count
    
    @Params:
        - df: DataFrame containing a reviews column
    """
    
    def __init__(self, df):
        self.df = df
    
    def pos_neg_words(self, value, file):
        words = {}
        with open(file, encoding="ISO-8859-1") as f:
            for line in f:
                if (len(line) == 0) or (line == '\n') or (line[0] == ';'): 
                    continue
                words[line.replace('\n', '')] = value
        
        return words
    
    def sentiment_words(self, filename):
        """
        Parameters
        ----------
        filename : str
            file path for sentiment scores. Represented with a pos, neg, or both score. .

        Returns
        -------
        Dictionary of sentiment scores for words.
        
        """
        
        df = pd.read_table(filename, skiprows=26)
        df['score'] = df['PosScore'] - df['NegScore']
        df = df[['SynsetTerms', 'score']]
        df.columns = ['words', 'score']

        # remove neutral words
        mask = df['score'] != 0
        df = df[mask]

        # Regex to find number
        rx1 = re.compile('#([0-9])')
        
        
        # Regex to find words
        verEx = VerEx()
        exp = verEx.range('a', 'z', 'A', 'Z')
        rx2 = re.compile(exp.source())
        
        sent_dict = {}
        for i, row in df.iterrows():
            w = row['words']
            s = row['score']
            nums = re.findall(rx1, w)
            
            w = w.split(' ')
            words = []
            if len(w) == 1:
                words = ''.join(re.findall(rx2, str(w)))
            else:
                words = [''.join(re.findall(rx2, str(string))) for string in w]
                
                
            for nn, ww in zip(nums, words):
                # only sentiment for the most common meaning of the word
                if nn == '1':
                    sent_dict[ww] = s

        return sent_dict
    
    def create_sent_dicts(self):
        
        """
        Returns
        -------
        None
        
        Create sentiment dictionaries from three sources :
        - https://www.quora.com/Is-there-a-downloadable-database-of-positive-and-negative-words
        - http://sentiwordnet.isti.cnr.it/
        - https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html

        """
        
        # Positive words
        self.pos_neg_dict = self.pos_neg_words(1, PROJECT_DIR + '/data/positive-words.txt')
        
        #Combine with negative words
        self.pos_neg_dict.update(self.pos_neg_words(-1, PROJECT_DIR + '/data/negative-words.txt'))
        
        
        self.sent_dict = self.sentiment_words(PROJECT_DIR + '/data/SentiWordNet.txt')
    
    def update_sentiment_score(self, val, sent_dict):
        """
        Parameters
        ----------
        val : float
            sentiment score
        sent_dict : dict
            {'score': #, 'pos_cnt': #, 'neg_cnt': #}

        Returns
        -------
        Dict with values updated on sign(val)

        """
        
        sent_dict['score'] += val 
        
        if val > 0:
            sent_dict['pos_cnt']+=1
        elif val < 0:
            sent_dict['neg_cnt'] += 1
        
        return sent_dict
    
    def text_features(self, row):
        text = row['text'].lower()
        
        reg = re.compile('[%s]' % re.escape(string.punctuation))
        punc_ct = Counter(re.findall(reg, text))
        row['e'] = punc_ct['!']
        row['q'] = punc_ct['?']
        row['punc'] = np.sum(punc_ct.values())
        row['chars'] = len(re.sub('\s+', '', text))
        
        # Strip punctuation
        text = re.sub("'", '', text)
        text = re.sub(reg, ' ', text).strip()
        
        words = re.split(r'\s+', text)
        row['words'] = len(words)
        
        # Add sentiment scores 
        sent_dict_1 = {'score':0.0, 'pos_cnt':0.0, 'neg_cnt':0.0}
        sent_dict_2 = sent_dict_1.copy()
        
        for w in words:
            val_1 = self.sent_dict.get(w, 0)
            val_2 = self.pos_neg_dict.get(w, 0)
            
            sent_dict_1 = self.update_sentiment_score(val_1, sent_dict_1)
            sent_dict_2 = self.update_sentiment_score(val_2, sent_dict_2)
            
            
            # Add sentiment features
            row['sent_1_score'] = sent_dict_1['score']
            row['sent_1_rate'] = sent_dict_1['score']/len(words)
            row['sent_1_pct'] = sent_dict_1['pos_cnt']
            row['sent_1_nct'] = sent_dict_1['neg_cnt']
            
            row['sent_2_score'] = sent_dict_2['score']
            row['sent_2_rate'] = sent_dict_2['score']/len(words)
            row['sent_2_pct'] = sent_dict_2['pos_cnt']
            row['sent_2_nct'] = sent_dict_2['neg_cnt']
            
        
        return row
    
    
    def update_feats(self):
        """
        MAIN -> update df features

        Returns
        -------
        None.

        """
        
        self.create_sent_dicts()
        self.df = self.df.apply(self.text_features, axis=1)
    
    
    def save_df(self, path=PROJECT_DIR + '/data/saved_df.pkl'):
        """Save the DataFrame to disk as pickle object."""

        self.df.to_pickle(path)

    def load_df(self, path=PROJECT_DIR + '/data/saved_df.pkl'):
        """Return a saved DataFrame."""

        with open(path, 'r') as f:
            return pickle.load(f)
        
    
def tokenize(text):
        """
        Parameters
        ----------
        text : str

        Returns
        -------
        Return tokenized words with stopwords removed

        """
        
        tokenizer = RegexpTokenizer(r'\w+')
        text = text.lower().replace("'", '')
        
        tokens = tokenizer.tokenize(text)
        
        return np.array(tokens), np.array([word for word in tokens if not word in STOPWORDS])
        
    
        
        
    
    
            
            
        
    
    
    
        
    
        
    
    
    
            
            
            
        
                
                