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


import urllib.request
from bs4 import BeautifulSoup
import requests
import random
import re


def google_results(google_url, number_result):
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
    return (links)
