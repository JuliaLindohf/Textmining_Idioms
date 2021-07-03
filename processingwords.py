# to mount the drive into the system 
from google.colab import drive
drive.mount('/content/drive')

import os
import json
import time
import requests
import datetime
import dateutil
import unittest
import pandas as pd
import configparser
from dateutil.relativedelta import relativedelta
import urllib3
import nltk
from os import getcwd
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp 
import seaborn as sns
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer
import spacy
import bz2 
import os
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm") 

nlp = spacy.load("en_core_web_sm")

class processingwords: 
  # to process the entities, words and collocations upon the word level
  def __init__(self, wholepassages, keysentences): 
    # two input variables are included 
    # the whole passages, the cleaned text data
    self.passages =  wholepassages
    # the keysentences, the cleaned data 
    self.sentences = keysentences 
    self.passage_entitylabels = []
    self.passage_entitytext = []
    # to create an empty dictionary, to store all interesting collocations
    self.collocationsdict = dict()
    self.bigrams = {}
    self.trigrams= {}

  def entity_determinations(self, passage):
    # to operation one passage at the time   
    doc = nlp(passage) 

    # to determine the entities: 
    for ent in doc.ents:
      self.passage_entitylabels.append(ent.label_)
      self.passage_entitytext.append(ent.text) 

  def passageentities(self):
    # to find out which entities are often used in the passages, which contain the keywords 
    wholepassage = self.passages 
    # to collect
    for passages in wholepassage: 
      self.entity_determinations( passages[0] )

    # to create a dictionary, which could store all the detected entities:
    entityDictionary = dict( ) 

    # to create a new varianle for this work  
    entities =  self.passage_entitytext 

    for ent in entities: 
      if ent in entityDictionary:
        entityDictionary[ent] += 1
      else:
        entityDictionary[ent] = 1

    return    entityDictionary

  def entity_labels_keysentences(self): 
    # the aim of constructing this function, is to count, how often a specific entity appear in a key sentence
    # to look at the key sentences
    keysentences = self.sentences 

    # to count the labels and entities
    entity_count = {}

    for sent in keysentences: 
      # to process the passage 
      doc = nlp(sent) 
      for ent in doc.ents: 
        tempent = (ent.text, ent.label_) 

        if tempent in entity_count: 
          entity_count[tempent] += 1
        else: 
          entity_count[tempent] = 1

    return entity_count


  def bigram_in_keysentence(self, apassage): 
    # to create an empty list, to store all the words 
    # the input variable is one row from the dataframe, not the entire list of sentences  
    n = 2

    wordlist = apassage.split( )
    l = len(wordlist)
    if l == 2: 
      tempbi = (wordlist[0], wordlist[1])
      if tempbi in self.bigrams:
        self.bigrams[tempbi] += 1
      else:
        self.bigrams[tempbi] = 1

    if l > 2 :
      for i in range(l-1): 
        tempbi = (wordlist[i], wordlist[i+1]) 

        if tempbi in self.bigrams:
          self.bigrams[tempbi] += 1
        else:
          self.bigrams[tempbi] = 1

    return self.bigrams

  def trigram_in_keysentence(self, apassage): 
    # to create an empty list, to store all the words 
    # the input variable is one row from the dataframe, not the entire list of sentences  
      wordslist = []

      n = 3
      wordlist = apassage.split( )
       
      l = len(wordlist) 
      # print(l)
      if l == 3: 
        tempbi = (wordlist[0], wordlist[1], wordlist[2])
        if tempbi in self.trigrams:
          self.trigrams[tempbi] += 1
        else:
          self.trigrams[tempbi] = 1

      if l > 3 :
        for i in range(l-2): 
          tempbi = (wordlist[i], wordlist[i+1], wordlist[i+2]) 
          if tempbi in self.trigrams:
            self.trigrams[tempbi] += 1
          else:
            self.trigrams[tempbi] = 1

      return self.trigrams

  def collectbigrams(self): 
     allkeysentences = self.sentences 

     for sentence in allkeysentences: 
          temp = self.bigram_in_keysentence(sentence)

     return self.bigrams

  def collecttrigrams(self): 
     allkeysentences = self.sentences 

     for sentence in allkeysentences: 
          temp = self.trigram_in_keysentence(sentence)

     return self.trigrams

  
