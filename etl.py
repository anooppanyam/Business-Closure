#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 23:37:36 2020

@author: anooppanyam
"""

import numpy as np
import pandas as pd
from os import path
from datetime import datetime
from textblob import TextBlob


__data__ = 'etl.py'
DATA_DIR = path.dirname(path.abspath(__data__))


def filter(path, critera, size=1000):
    chunk_list = []
    reader = pd.read_json(path, lines=True, chunksize=size)
    
    for chunk in reader:
        chunk_filter = chunk[chunk.business_id.isin(critera)]
        chunk_list.append(chunk_filter)
    
    reader.close()
    return pd.concat(chunk_list)


class ETL(object):
    def __init__(self, business_path, checkins_path, reviews_path, criteria):
        self.criteria = criteria
        try:
            self.business = pd.read_json(business_path, lines=True)
            self.relevant_businesses = set(self.business[self.business['state'].isin(self.criteria)].business_id.unique())
        except:
            print("Invalid business path...")
        
        
        try:
            self.checkins = filter(checkins_path, self.relevant_businesses, 1000)
            self.checkins['date'] = self.checkins['date'].apply(lambda x : list(map(lambda y : datetime.strptime(y, "%Y-%m-%d %H:%M:%S"), x.split(", "))))
            self.checkins['checkins'] = self.checkins['date'].apply(len)
            self.checkins['interval'] = self.checkins['date'].apply(lambda x : (x[-1] - x[0]).days if (x[-1] != x[0]) else 0)
            
        except:
            print("Invalid checkin path...")
            
            
        try:
            self.reviews = filter(reviews_path, self.relevant_businesses, 1000)
            
        except:
            print("Invalid reviews path...")
        
        
        self.merge()
    
    
    def getWeightedSentiment(self):
        bus_rev_dict = {}
        for x in self.relevant_businesses:
            bus_rev_dict[x] = []
        for x,c in self.reviews.iterrows():
            try:
                bus_rev_dict[c['business_id']].append((c['text'], c['useful']))
            except:
                pass
        
        def findavgsentiment_useful(textarr):
            totalSent = 0
            for text, useful in textarr:
                blob = TextBlob(text)
                totalSent += blob.sentiment.polarity*(useful+1)
            return totalSent/len(textarr)
        
        bus_score_dict = {}
        for x in self.relevant_businesses:
            bus_score_dict[x] = []
        for item in bus_rev_dict.items():
            bus_score_dict[item[0]] = findavgsentiment_useful(item[1])
        
        self.bus_score_dict = bus_score_dict
    
    
    
    
    def merge(self, include_weighted_sent=False):
        self.df = None
        # Select businesses in 
        self.business = self.business[self.business['state'].isin(self.criteria)]
        
        try:
            # inner join checkins with main df on business id
            self.df = self.business.merge(self.checkins, how='inner', on='business_id').drop(['date'], axis=1)
        except:
            print('Error with merge of checkins and business')
        
        # Add weighted TextBlob sentiment based on usefulness
        if include_weighted_sent:
            self.getWeightedSentiment() 
            self.df['WeightedSentiment'] = self.df['business_id'].map(self.bus_score_dict)
            
            # Process reviews for business id as large corpus, include mean and median star data
        reviews = self.reviews.groupby(by='business_id')
        means = reviews['stars'].mean().reset_index(name = 'MeanStars')
        medians = reviews['stars'].median().reset_index(name='MedianStars')
            
        means = dict(zip(means['business_id'], means['MeanStars']))
        medians = dict(zip(medians['business_id'], medians['MedianStars']))
            
        self.df['MedianStars'] = self.df['business_id'].map(medians)
        self.df['MeanStars'] = self.df['business_id'].map(means)
        self.df = self.df.merge(self.reviews.groupby(by='business_id').text.apply(list).reset_index(name='text'), how='inner', on='business_id')
        self.df['text'] = self.df['text'].apply(lambda x : " ".join([n.strip() for n in x]))






