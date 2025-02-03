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

<<<<<<< main
## Example:

In the following user input, we choose to analyze Boeing Co
  
![image](https://github.com/user-attachments/assets/a1cb13dd-737a-4f33-9880-9da90e7e49e4)

by using financial news from the past 30 days, counting from today:  
  
![image](https://github.com/user-attachments/assets/659c9615-23e2-4b63-b911-374bbb2e2a55)

and a FinVADER set up as encouraged by Korab (2023):  

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
and get the following output:  
  
  **1. Assigned sentiment scores (floats from -1 to 1)**  
```python news_df['combined_score'].describe()``` tells us the average sentiment score assigned to Boeing Co. is postive, but there is a wide range of values (from -0.807 to 0.929), particularily for daily and weekly sentiment scores.

<img src="https://github.com/user-attachments/assets/767e7832-f581-4634-9337-47bdf491d416" width="400"/>

   
  **2. WordCloud: Cluster of words presented by their frequency in financial news for BA stock**
  
<img src="https://github.com/user-attachments/assets/0bbe5eeb-4697-4ac1-a669-7fc960587d80" width="800"/>

  **3. Evolution of sentiment scores for BA over the selected time horizon**
  
<img src="https://github.com/user-attachments/assets/f5ec333e-14a3-4a0d-ab09-8a7997d7f432" alt="Evolution of sentiment scores for BA" width="800"/>

  **4. Filtering for news with poor semantic scores, we take a look at new word clusters in filtered data**  

  ![image](https://github.com/user-attachments/assets/157e10b7-1b2e-49c6-8de7-bcc6f903ba4f)


