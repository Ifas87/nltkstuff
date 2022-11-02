from nrclex import NRCLex
import pandas as pd


def main():
    sample_data = pd.read_csv('Tweets.csv')["text"]
    bigString = ",".join(sample_data)
    
    text_stuff = NRCLex(bigString)
    results = text_stuff.raw_emotion_scores
    print(results)


if __name__ == '__main__':
    main()