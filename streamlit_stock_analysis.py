import streamlit as st
import requests
import pandas as pd
import datetime
import time
import random
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
from finvader import finvader

#%%Creating streamlit title and short description
st.title("Stock News Sentiment Analysis")
st.write("Using a version of VADER package specialized in financial lexicon and based on previous financial news, sentiment scores are assigned to stock tickers on a daily, weekly and monthly basis. The available input data is presented in the table below, that is followed by visual presentations (WordCloud, historical chart) based on the results of the sentiment analysis. In the end, a WordCloud based only for Articles with Poor Sentiment Scores is presented (if there are negative sentiment scores in the analysed timeframe).")

#%%Required inputs for sentiment analysis, API key is specific to the news site
api_key = 'OTl2SM9_8xGEkqop_pj57cYyS4gjsurl'
news_url = "https://api.polygon.io/v2/reference/news"

#%%Creating the input bars for streamlit
ticker = st.sidebar.text_input("Enter the stock ticker symbol (e.g., BA):", "AAPL").strip().upper()
date_choice = st.sidebar.selectbox("Choose the date range:", ["Last day", "Last week", "Last month"])

#%%Creating the sidebar for color maps that work for both wordcloud and the trend analysis chart
color_map = {
    "Blues": "blue",
    "Reds": "red",
    "Greens": "green",
    "coolwarm": "cyan",
    "viridis": "yellowgreen",
    "plasma": "orangered",
    "inferno": "darkred",
    "magma": "purple",
    "cividis": "gold"
}
color_scheme = st.sidebar.selectbox("Choose Color Scheme:", list(color_map.keys()))
chart_color = color_map[color_scheme]

#%%Determining the date range based on possible inputs 
today = datetime.date.today()
if date_choice == "Last day":
    start_date = (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
elif date_choice == "Last week":
    start_date = (today - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
elif date_choice == "Last month":
    start_date = (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
end_date = today.strftime("%Y-%m-%d")

#%%News macro that fetches news articles from the site based on the ticker input (issues when data is requested too quickly -- sleeper added, also issues when requesting more than 5 tickers per minute -- cannot be solved as higher limits are locked behind pay wall but added error message that shows issue)
def get_news(ticker, start_date, end_date, api_key, news_url, limit=100):
    params = {
        "ticker": ticker,
        "published_utc.gte": start_date,
        "published_utc.lte": end_date,
        "limit": limit,
        "apiKey": api_key
    }
    time.sleep(random.uniform(1, 3))  
    
    response = requests.get(news_url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get("results", [])
    else:
        st.error(f"Failed to get news: {response.status_code} - {response.text}")
        return []

#%%Acquiring news articles for the selected ticker (as ticker input is not dropdown -- added warning when no news articles could be found ie wrong ticker)
articles = get_news(ticker, start_date, end_date, api_key, news_url)

if not articles:
    st.warning(f"No news articles found for the selected period for {ticker}.")
else:
    #Save articles in a dataframe --- there is limit for the number of news articles that we can download from the site = 100
    df = pd.DataFrame(articles)[["title", "description", "published_utc"]]
    
#%%FinVADER sentiment analysis on news articles macro
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

    
df['combined_text'] = df['title'] + " " + df['description'].fillna("")
df['combined_score'] = df['combined_text'].apply(sentiment_analysis)
    
#%%Aggregate the sentiment analysis data based on date
df['published_utc'] = pd.to_datetime(df['published_utc'])
sentiment_data = df.groupby(df['published_utc'].dt.date)['combined_score'].mean().reset_index()
    
#%%Including the dataframe table in streamlit with filters
st.subheader(f"News Articles for {ticker}")
df_filtered = st.data_editor(df, num_rows="dynamic")

#%%Including historical sentiment trend chart for tickewr
if not df.empty and 'combined_score' in df.columns:
    st.subheader(f"Historical Sentiment Trend for {ticker}")

    df['published_utc'] = pd.to_datetime(df['published_utc'])

    daily_avg_sentiment = df.groupby(df['published_utc'].dt.date)['combined_score'].mean().reset_index()

    fig = px.scatter(df, x='published_utc', y='combined_score', title='Historical Sentiment Scores',
                     labels={'combined_score': 'Sentiment Score', 'published_utc': 'Date'},
                     color_discrete_sequence=[chart_color])

    best_articles = df.nlargest(5, 'combined_score')
    worst_articles = df.nsmallest(5, 'combined_score')

    fig.add_scatter(x=best_articles['published_utc'], y=best_articles['combined_score'], 
                    mode='markers', marker=dict(color='green', size=10, symbol='star'), 
                    name='Best Articles')

    fig.add_scatter(x=worst_articles['published_utc'], y=worst_articles['combined_score'], 
                    mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), 
                    name='Worst Articles')

    fig.add_scatter(x=daily_avg_sentiment['published_utc'], y=daily_avg_sentiment['combined_score'], 
                    mode='lines', line=dict(color='blue', width=2), 
                    name='Daily Average Sentiment')
    st.plotly_chart(fig)
    
#%%Including descriptive statistics based on the dataframe table
if not df.empty and 'combined_score' in df.columns:
    st.subheader(f"Descriptive Statistics for {ticker}")

    descriptive_stats = df['combined_score'].describe().to_frame()
    descriptive_stats.rename(columns={'combined_score': 'Sentiment Score Statistics'}, inplace=True)

    st.dataframe(descriptive_stats)
else:
    st.warning("No sentiment analysis data available for descriptive statistics.")
    
#%%Including the best and worst rated articles separately
if not df.empty and 'combined_score' in df.columns:
    best_articles = df.nlargest(5, 'combined_score')[["title", "description", "combined_score", "published_utc"]]
    worst_articles = df.nsmallest(5, 'combined_score')[["title", "description", "combined_score", "published_utc"]]

    st.subheader(f"Top 5 Articles with the Best Sentiment Scores for {ticker}")
    st.data_editor(best_articles, num_rows="dynamic")

    st.subheader(f"Top 5 Articles with the Worst Sentiment Scores for {ticker}")
    st.data_editor(worst_articles, num_rows="dynamic")
    
#%%Generating WordCloud based on ticker input
text_data = " ".join(df["combined_text"].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=color_scheme).generate(text_data)
    
st.subheader(f"Word Cloud for {ticker}")
fig, ax = plt.subplots()
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)
    
#%%Generating waterfall chart based on sentiment trend analysis
st.subheader(f"Sentiment Trend for {ticker}")
fig = px.bar(sentiment_data, x='published_utc', y='combined_score', title='Sentiment Analysis Over Time', 
                 text_auto=True, labels={'combined_score': 'Sentiment Score', 'published_utc': 'Date'},
                 color_discrete_sequence=[chart_color])
st.plotly_chart(fig)
    
#%%Generating WordCloud based only on poor sentiment news articles (score below 0) based on ticker input
poor_sentiment_data = df[df['combined_score'] < 0]
if not poor_sentiment_data.empty:
        st.subheader(f"Word Cloud for Articles with Poor Sentiment Scores for {ticker}")
        wordcloud = WordCloud(width=800, height=400, background_color="white", colormap=color_scheme).generate(" ".join(poor_sentiment_data['combined_text'].dropna()))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)