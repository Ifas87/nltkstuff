from nrclex import NRCLex
import pandas as pd


def main():
    sample_data = pd.read_csv('Tweets.csv')["text"]
    bigString = ",".join(sample_data)
    # print(bigString)
    text_stuff = NRCLex(bigString)


if __name__ == '__main__':
    main()