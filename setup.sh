#!/bin/bash
git clone https://github.com/ProsusAI/finBERT
git clone https://github.com/pauljhp/FinancialModelingPrep
cd ..
conda env create -f ./StockSentimentAnalysis/environment.yml
conda activate sent_env
mkdir -p ./finBERT/models/sentiment/pretrained/finbert-sentiment/
# curl https://prosus-public.s3-eu-west-1.amazonaws.com/finbert/finbert-sentiment/pytorch_model.bin > ./finBERT/models/sentiment/pretrained/finbert-sentiment/pytorch_model.bin
cp ./finBERT/config.json ./finBERT/models/sentiment/pretrained/finbert-sentiment/config.json
