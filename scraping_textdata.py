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


import urllib.request
from bs4 import BeautifulSoup
import requests
import random
import re

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm") 

nlp = spacy.load("en_core_web_sm") 


# This function was used to fetch the weburl, based on the result from a google search

class scraping_data: 
  def __init__(self, google_url, number_result):
    A = ("Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36",
         "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.1 Safari/537.36",
         "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36",
           )
 
    Agent = A[random.randrange(len(A))]
 
    headers = {'user-agent': Agent}
    response = requests.get(google_url + "&num=" + str(number_result), headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    result = soup.find_all('div', attrs = {'class': 'ZINbbc'})
    results=[re.search('\/url\?q\=(.*)\&sa',str(i.find('a', href = True)['href'])) for i in result if "url" in str(i)]
    links=[i.group(1) for i in results if i != None]
    self.weburl = links 
    self.NumberofResults = number_result

  def Collecting_text(self): 
    # to fetch the number of urls which will be processed 
    antal = self.NumberofResults 
    session = requests.Session() 
    # to fetch all scraped urls 
    webadress = self.weburl
    # to create an empty list, to store the fetched text data 
    passage_list = []  
    for item in range(antal-1): 
      tempurl = webadress[item]
      req = session.get(tempurl)
      soup = BeautifulSoup(req.text, 'html.parser')
      paragraphs = soup.find_all('p')
      temp = ""
      for text in paragraphs:
        temptext = str(text.get_text())
        temp = temp + temptext
      L = len(temp)
      if L > 10:
        passage_list.append(temp)

    return passage_list 
  
  
  
  
