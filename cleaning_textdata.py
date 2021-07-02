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

class cleantextdata:
  # The procedures is designed to clean the input text data, from a data frame 
  # To store both the 
  def __init__(self, inputdataframe, keywords): 
    # the scraped data is the input parametre
    # three columns should be included: urls, passages, newspaper title
    # this dataframe will be used through the entire object
    self.df = inputdataframe
    # to store the intended keyword list
    self.keywords = keywords
    # to create an empty diction
    self.sentencebank = dict()

  def preprocess(self, text): 
    # to process one sentence token at the time
    # it is a service function for further process 
    # the input variable sentencetoken is a complete sentence 
    doc = nlp(text, disable=["tagger", "parser", "ner"]) 
    # to create an empty list, which we will be storing the words late 
    wordlist = []


    # tokenisation: Segmenting text into words, punctuations marks etc.
    for words in doc:
      # lower case
      if str(words).isalpha() == True: 
         # stop word removal 
        if words.is_stop == False: 
          # to get rid of all non-alphabetic signs from the sentence token
          if words.lemma_.isalpha() == True:
          # lemmaisation: Assigning the base forms of words.
            wordlist.append(words.lemma_)
    return ' '.join(wordlist)


  def sentencetokens(self, apassage): 
    # to tokenise the passages into sentences and store the sentences into a list
    temptext = sent_tokenize(apassage)  

    # to create an empty list, to store all tokenised sentences
    sentencelist = []
 
    for sentence in temptext: 
      newsentence = self.preprocess(sentence)
      # to store all clearned textdata into a list 
      sentencelist.append(newsentence)

    return sentencelist

  def lookforexpressions(self, keywords, tokenisedpassage): 
    # to place the keywords in a container
    tempkeywordlist = self.keywords 
    refwords = set(tempkeywordlist) 

    # to count the number of sentence tokens in the current passage
    L = len(tokenisedpassage) 

    # to create an empty list, to store the relevent sentences
    storedsentence = [] 

    for i in range(L): 

      # to fetch one sentence at the time
      currenttoken = tokenisedpassage[i] 
      # to split the sentence into individual string variables
      currentwords = currenttoken.split( )
      # To store three sentences around the expressions, which contain the keywords 
      # the results could be used in the next step of processing
      for word in currentwords:
        if word in refwords:
          storedsentence.append(tokenisedpassage[i-1])
          storedsentence.append(tokenisedpassage[i])
          break

    # the result is a string of words, which contain the key phrase 
    result_sentence = ' '.join(storedsentence) 
    return     result_sentence 



  def buildaframe(self): 
    # to create a new dataframe, which contains the passages, which could be used
    # in the next round of processing 
    tempdataframe = self.df 
    temppassages = tempdataframe['passage'] 

    # to store the entire cleaned passage
    list_of_passage = [] 

    # to store the relevent sentences 
    list_of_studied_sentences = [ ]
    print('wee')


    # to use this keyword as the reference keywords 
    currentkeywords = self.keywords

    for passage in temppassages: 
        sentencetokens = self.sentencetokens(passage) 
        list_of_passage.append(sentencetokens) 
        keysentence = self.lookforexpressions(currentkeywords, sentencetokens)
        list_of_studied_sentences.append(keysentence) 

    new_df = pd.DataFrame(list(zip(list_of_passage, list_of_studied_sentences)), 
                         columns =['passage', 'keysentences'])

    return new_df 
