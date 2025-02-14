# Final Project
Using a version of the VADER package specialized in financial lexicon and using data from a financial news website, we assigned sentiment scores to stock tickers on a daily, weekly and monhtly horizon.    
  
We present those results using streamlit and its interactive interface and provide the user with wordclouds, trend analysis and similar visualizations based on the chosen stock ticker and date range.

## How to use
This script uses the vader_lexicon resource from NLTK for sentiment analysis that is downloaded automatically. If it fails to download or you get an error showing 'Resource vader_lexicon not found.' you need to ensure to download the resource and have it in the folder where you run the script from, e.g., '/Users/xy.../nltk_data'. To download run the following command in your Python environment: 
```import nltk```
```nltk.download('vader_lexicon') ```
After completing the above steps: running ```streamlit run streamlit_stock_analysis.py ``` should trigger the Streamlit and allow the user to choose prefered asset and time range.  
Sentiment scores (floats ranging from -1 to 1) are assigned to the assets based on a combination of news articles' titles and descriptions.
The code also offers a deeper dive into the negative sentiments by giving a tabular view and filtering for data with poor sentiment scores (below 0) and fetching more context, as well as generating a new wordcloud with only the bad news.

Using financial news data obtained from YahooFinance, we also provide plots of the adjusted close prices against the computed sentiment scores. However, it is important to keep in mind that the past week and month are defined differently for stock trading and news articles. We have yet to come up with a solution for this.  
  
The code will return the following message if no news articles were published for a given asset and time range, e.g. last 24h.  
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
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from finvader import finvader
import yfinance as yf
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


