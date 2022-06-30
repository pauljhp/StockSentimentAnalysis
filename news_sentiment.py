from FinancialModelingPrep.tickers import Ticker
from FinancialModelingPrep.indices import Index
import transformers
from transformers import (AutoModel, BertTokenizer,
    BertForSequenceClassification,)
import torch
import torch.nn as nn
import torch.nn.functional as F
from finBERT.finbert import utils
import pandas as pd
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datetime as dt
import pandas as pd
import numpy as np
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, List, Optional, Tuple

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
TODAY = dt.date.today()

def get_prediction(text: str, 
    model: AutoModelForSequenceClassification=model, 
    tokenizer=tokenizer) -> np.array:
    """
    Get one prediction.

    Parameters
    ----------
    text: str
        The text to be analyzed.
    model: BertModel
        The model to be used.
    tokenizer: BertTokenizer
        The tokenizer to be used.

    Returns
    -------
    predition: np.array
        An array that includes probabilities for each class.
    """

    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    token_type_ids = [0] * len(tokens)
    attention_mask = [1] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    padding = [0] * (64 - len(input_ids))
    input_ids += padding
    attention_mask += padding
    token_type_ids += padding

    features = []
    features.append(
        utils.InputFeatures(input_ids=input_ids,
                      token_type_ids=token_type_ids,
                      attention_mask=attention_mask,
                      label_id=None))

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    model.eval()
    logits = model(input_ids=all_input_ids, attention_mask=all_attention_mask, 
        token_type_ids=all_token_type_ids).get('logits')
    prediction = F.softmax(logits,
        dim=logits.shape[0]).detach().numpy()
    return prediction

def get_daily_sentiment_series(ticker: str, 
    start_date: Union[dt.date, str]=dt.date(2017, 1, 1)):
    """get historical news for a ticker and run inference"""
    if isinstance(start_date, dt.date):
        start_date = start_date.strftime("%Y-%m-%d")
    elif isinstance(start_date, str):
        pass
    else:
        raise TypeError("start_date must be a datetime.date or a string")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(Ticker.get_stock_news, ticker, start_date=start_date),
            executor.submit(Ticker.get_historical_price, ticker, start_date=start_date, end_date=TODAY)]
        news, prices = (future.result() for future in as_completed(futures))

    if news.shape[-1] > prices.shape[-1]:
        news, prices = prices, news
    prices.columns = pd.MultiIndex.from_tuples([("price", i) for i in prices.columns])
    news = Ticker.get_stock_news(ticker, start_date="2022-01-01")
    news.loc[:, "day"] = news.index.get_level_values(
        "publishedDate").to_series().dt.to_period("D").values
    title_sent = news.set_index(["day"]).title.apply(
        get_prediction).to_frame().reset_index()
    content_sent = news.set_index(["day"]).text.apply(
            get_prediction).to_frame().reset_index()
    columns = pd.MultiIndex.from_tuples(
        list(itertools.product(["title_sentiment", "content_sentiment"],
            ["positive", "negative", "neutral"])) + [("news_count", "count")])
    summary = pd.DataFrame(columns=columns, 
        index=title_sent.groupby("day").mean().index)
    summary.loc[:, ('title_sentiment', 'positive')] = \
        title_sent.set_index("day").title.apply(lambda x: x[0][0]).groupby("day").mean()
    summary.loc[:, ('title_sentiment', 'negative')] = \
        title_sent.set_index("day").title.apply(lambda x: x[0][1]).groupby("day").mean()
    summary.loc[:, ('title_sentiment', 'neutral')] = \
        title_sent.set_index("day").title.apply(lambda x: x[0][2]).groupby("day").mean()
    summary.loc[:, ('content_sentiment', 'positive')] = \
        content_sent.set_index("day").text.apply(lambda x: x[0][0]).groupby("day").mean()
    summary.loc[:, ('content_sentiment', 'negative')] = \
        content_sent.set_index("day").text.apply(lambda x: x[0][1]).groupby("day").mean()
    summary.loc[:, ('content_sentiment', 'neutral')] = \
        content_sent.set_index("day").text.apply(lambda x: x[0][2]).groupby("day").mean()

    summary.loc[:, ("news_count", "count")] = news.groupby("day").count().negative.values
    summary.merge(prices, left_index=True, right_index=True, 
        left_on="day", right_on="date")
    return summary