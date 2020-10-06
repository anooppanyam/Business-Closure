#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:32:21 2020

@author: anooppanyam
"""


from __future__ import division
import re
import string
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

sns.set_style('whitegrid')

from preprocess import tokenize 

from wordcloud import WordCloud, STOPWORDS


def plot_sentiment(df):
    """ Plot bar chart of sentiment rates from various sources grouped by open status """
    
    cols = ['is_open', 'sent_1_score', 'sent_2_score'] #'WeightedSentiment']
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
    df[cols].groupby('is_open').mean()['sent_1_score'].plot(kind='bar', ax=ax[0])
    df[cols].groupby('is_open').mean()['sent_2_score'].plot(kind='bar', ax=ax[1])
    
    titles = ['Mean Hu & Liu Lexicon Score', 'Mean WordNet Score']
    for i in range(len(ax)):
        ax[i].set(title=titles[i])
    
    
    
def plot_checkins(df):
    """
    Plot bar chart of total checkins grouped by open status

    """
    cols = ['is_open', 'checkins']
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
    df[cols].groupby('is_open')['checkins'].mean().plot(kind='bar', ax=ax)
    ax.set(title='Total Checkins', ylabel='Mean Number of Checkins')

def unique_word_ct_util(text):
    all_words, without_stop = tokenize(text)
    print('Number of unique words in all reviews: {0}'.format(len(set(all_words))))
    print('Number of unique words (excluding stop words): {0}'.format(len(set(without_stop))))

def plot_punc(df):
    """ Plot bar chart of mean number of ? and ! grouped by open status """
    
    cols = ['is_open', 'q', 'e']
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
    
    df[cols].groupby('is_open').mean()['q'].plot(kind='bar', ax=ax[0])
    df[cols].groupby('is_open').mean()['e'].plot(kind='bar', ax=ax[1])
    
    titles = ['Mean Number of Question Marks', 'Mean Number of Exclamaiton Marks']
    for i in range(len(ax)):
        ax[i].set(title=titles[i], ylabel='Mean Count')
   
    

def preprocess_rev(df):
    text = df['text'].str.lower().str.replace("'", '')
    for punc in string.punctuation:
        text = text.str.replace(punc, ' ')
    
    text = text.str.replace(r'\s+', ' ')
    return ' '.join(text.values)

def unique_ct(df):
    return unique_word_ct_util(preprocess_rev(df))

def open_distr(df):
    cols = ['is_open']
    plt.figure(figsize=(6,6))
    sns.countplot(x='is_open', data=df[cols]).set_title('Distribution of Open Status')
    plt.show()

def star_distr(df):
    cols = ['stars']
    plt.figure(figsize=(6,6))
    sns.countplot(x='stars', data=df[cols]).set_title('Distribution of Yelp Star Ratings')

def cat_distr(df):
    df['categories'] = df['categories'].apply(lambda x : x if type(x) == str else "")   
    business_cats=','.join(df['categories'])
    cats=pd.DataFrame(business_cats.split(','),columns=['category'])
    cats_ser = cats.category.value_counts()

    cats_df = pd.DataFrame(cats_ser)
    cats_df.reset_index(inplace=True)
    plt.figure(figsize=(12,10))
    f = sns.barplot( y= 'index',x = 'category' , data = cats_df.iloc[0:20])
    f.set_title('Top 20 Business Categories')
    f.set_ylabel('Category')
    f.set_xlabel('Number of businesses')

def name_cloud(df):
    cols = ['name']
    plt.figure(figsize=(12,10))
    wordcloud = WordCloud(background_color='white',
                          width=1200,
                      stopwords = STOPWORDS,
                          height=1000
                         ).generate(str(df[cols]))
    plt.imshow(wordcloud)
    plt.axis('off')
    
    

def show_all(df):
     open_distr(df)
     print('\n')
     star_distr(df)
     print('\n')
     cat_distr(df)
     print('\n')
     name_cloud(df)
     print('\n')
     plot_sentiment(df)
     print('\n')
     plot_checkins(df)
     print('\n')
     plot_punc(df)
     #unique_ct(df)
     plt.show()

    
    
    