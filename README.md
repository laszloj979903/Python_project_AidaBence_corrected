# Final Project
Using a version of the VADER package specialized in finance lexicon and using data from a financial news website, we assigned sentiment scores to stock tickers on a daily, weekly and monhtly horizon.    
  
We present those results using streamlit and its interactive interface and provide the user with wordclouds, trend analysis and similar visualizations based on the chosen stock ticker and date range.

## How to use
Running ```python streamlit run streamlit_stock_analysis.py ``` should trigger the Streamlit and allow you to choose prefered stock ticker and time range on the left side.  
The code assigns sentiment scores (floats ranging from -1 to 1) to the chosen asset and provides a brief visual analysis with wordcloud and a plot of sentiment score evolution over time.  
The code also offers a deeper dive into the negative sentiments by giving a tabular view and filtering for data with poor sentiment scores (below 0) and fetching more context, as well as generating a new wordcloud with only the bad news.

The code will return the following message if no news articles were published for a certain time range, e.g. last 24h.
![image](https://github.com/user-attachments/assets/344da849-4e21-4ff2-98a8-d4a47f13c1f9)


## Software and Data Requirements 
requirements.txt file is provided in the GitHub repository and includes the following packages:
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
FinVADER package requires Python 3.8 - 3.11, and NLTK 3.6.  
The package required complete text data without NaN values and empty strings.  
  
We set up FinVADER as explained by Korab (2023):  

```python
#finVADER implementation
def sentiment_analysis(text):
    try:
        score = finvader(
            text,
            use_sentibignomics=True,  # Use SentiBignomics lexicon
            use_henry=True,          # Use Henry's lexicon
            indicator="compound"     # Compound sentiment score
        )
        return score
    except Exception as e:
        print(f"Error in FinVADER analysis: {e}")
        return None
```


