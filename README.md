# Final Project
Using a version of VADER package specialized in financial lexicon and based on previous financial news, we assign sentiment scores to stock tickers on a daily, weekly and monhtly basis.
This is followed by a visual presentation of the sentiment analysis.

## Software and Data Requirements 

Data needs to be complete text data without NaN values and empty strings. 
FinVADER requires Python 3.8 - 3.11, and NLTK 3.6.  
Other essential packages to be imported are specified in the code and are as follows:
```python
import requests
import pandas as pd
import datetime
import time
import random
import nltk
from tqdm import tqdm
from finvader import finvader
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
```
