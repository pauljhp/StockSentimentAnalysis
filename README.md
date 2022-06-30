# StockSentimentAnalysis
SentimentAnalysis model for stock news

## Using this package

### Set up

Clone this repository

```$ git clong https://github.com/pauljhp/StockSentimentAnalysis/```

Make sure you are using conda. The following script uses codna to set up the environment

Then run the setup script:

```$ sudo bash setup.sh ```

Using the package:

```>>> from StockSentimentAnalysis import news_sentiment
>>> import datetime as dt
>>> summary = news_sentiment.get_daily_sentiment_series("FB", 
    start_date=dt.date(2022, 1, 1), lim=1000)```
