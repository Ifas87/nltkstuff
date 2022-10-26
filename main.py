import pandas as pd
import matplotlib as plt
import nltk
import numpy
import string
import tensorflow
import tensorflow_text as tf

def main():
    data_sample = pd.read_csv('Tweets.csv')
    #print(data_sample.columns)
    #print(data_sample['text'])
    #print(data_sample['airline_sentiment'])

    relevant_sample = data_sample[ ['text', 'airline_sentiment'] ]
    #print(relevant_sample.head())
    # print(relevant_sample["airline_sentiment"].value_counts())
    # print(relevant_sample.shape)
    relevant_sample = relevant_sample[relevant_sample['airline_sentiment'] != "neutral"]
    # print(relevant_sample["airline_sentiment"].value_counts())
    # print(relevant_sample["airline_sentiment"].factorize())
    nptext = relevant_sample['text'].values
    tokenisers = tensorflow.keras.preprocessing.text.Tokenizer(num_words=5000)
    tokenisers.fit_on_texts(nptext)


if __name__ == '__main__':
    main()