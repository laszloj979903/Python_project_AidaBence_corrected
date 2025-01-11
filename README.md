# Final Project
Using a version of VADER package specialized in financial lexicon and based on previous financial news, we assign sentiment scores to stock tickers on a daily, weekly and monhtly basis.
This is followed by some visual presentation of the results from the sentiment analysis.

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
## How to use

- In user input part enter the stock ticker for which you want to see a brief overview of sentiment analysis of related news. (e.g. BA *for Boeing*)  
- The time range will be determined also based on user's input, whether it will fetch news data from the past day, week or month. *Due to the availability of data within the chosen approach, the current maximum of the time span is one month.*  
- Running the rest of the code will trigger assigning sentiment scores to the chosen asset and a brief visual analysis with wordcloud and a plot to show sentiment score evolution over time.  
- The code also offers a deeper dive into the negative scores by filtering for data with poor sentiment scores to get more context.



