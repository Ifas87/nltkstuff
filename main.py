import pandas as pd
import matplotlib as plt
import nltk
import numpy
import string


def main():
    data_sample = pd.read_csv('Tweets.csv')
    #print(data_sample.columns)
    #print(data_sample['text'])
    #print(data_sample['airline_sentiment'])

    relevant_sample = data_sample[ ['text', 'airline_sentiment'] ]
    #print(relevant_sample.head())
    #print(relevant_sample["airline_sentiment"].value_counts())




if __name__ == '__main__':
    main()