import requests 
import pandas as pd
import datetime
import time
import random
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from finvader import finvader


def get_news(ticker, start_date, end_date, api_key, news_url, limit=100):
    """
    Fetch news for a ticker within a given date range.
    """
    params = {
        "ticker": ticker,
        "published_utc.gte": start_date,
        "published_utc.lte": end_date,
        "limit": limit,
        "apiKey": api_key
    }
    time.sleep(random.uniform(1, 2))  # Shorter sleep for testing/debugging
    
    response = requests.get(news_url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get("results", [])
    else:
        print(f"Failed to get news: {response.status_code} - {response.text}")
        return []


def generate_weekly_intervals(start_date, end_date):
    """
    Generate weekly date intervals within the specified date range.
    """
    intervals = []
    current_end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    current_start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    
    while current_end_date > current_start_date:
        week_start_date = max(current_start_date, current_end_date - datetime.timedelta(days=6))
        intervals.append((week_start_date.strftime("%Y-%m-%d"), current_end_date.strftime("%Y-%m-%d")))
        current_end_date -= datetime.timedelta(days=7)
    
    return intervals


def aggregate_sentiment(news_df):
    """
    Aggregate sentiment scores into daily, weekly, and monthly averages.
    """
    if news_df.empty:
        print("No data to aggregate.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Aggregation for 1-day intervals
    daily_sentiment = news_df.groupby(['published_utc', 'ticker'])['finvader_compound'].mean().reset_index()
    daily_sentiment.rename(columns={'finvader_compound': 'daily_sentiment'}, inplace=True)

    # Aggregation for 1-week intervals
    news_df['week'] = news_df['published_utc'] - pd.to_timedelta(news_df['published_utc'].dt.dayofweek, unit='d')
    weekly_sentiment = news_df.groupby(['week', 'ticker'])['finvader_compound'].mean().reset_index()
    weekly_sentiment.rename(columns={'finvader_compound': 'weekly_sentiment'}, inplace=True)

    # Aggregation for 1-month intervals
    news_df['month'] = news_df['published_utc'].dt.to_period('M')
    monthly_sentiment = news_df.groupby(['month', 'ticker'])['finvader_compound'].mean().reset_index()
    monthly_sentiment.rename(columns={'finvader_compound': 'monthly_sentiment'}, inplace=True)

    return daily_sentiment, weekly_sentiment, monthly_sentiment


def generate_wordcloud(news_df, ticker):
    """
    Generate a word cloud for a specific ticker.
    """
    month_data = news_df[news_df['ticker'] == ticker]
    if month_data.empty:
        print(f"No data available for ticker {ticker} to generate word cloud.")
        return

    text = ' '.join(month_data['title'].dropna())
    if not text:
        print(f"No valid text found for ticker {ticker} to generate word cloud.")
        return

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for {ticker}")
    plt.show()


def visualize_sentiment_trends(daily_sentiment, weekly_sentiment, monthly_sentiment, ticker):
    """
    Visualize sentiment trends.
    """
    plt.figure(figsize=(12, 6))
    
    if not daily_sentiment.empty:
        plt.plot(daily_sentiment['published_utc'], daily_sentiment['daily_sentiment'], label='Daily Sentiment', marker='o')

    if not weekly_sentiment.empty:
        plt.plot(weekly_sentiment['week'], weekly_sentiment['weekly_sentiment'], label='Weekly Sentiment', marker='s')

    if not monthly_sentiment.empty:
        plt.plot(monthly_sentiment['month'], monthly_sentiment['monthly_sentiment'], label='Monthly Sentiment', marker='^')

    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.title(f"Sentiment Trends for {ticker}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def analyze_sentiment_finvader(text):
    """
    Analyze sentiment using FinVADER.
    """
    try:
        if not text or not isinstance(text, str):
            return None

        score = finvader(
            text,
            use_sentibignomics=True,
            use_henry=True,
            indicator="compound"
        )
        return score
    except Exception as e:
        print(f"Error in FinVADER analysis: {e}")
        return None
