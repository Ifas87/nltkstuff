from nrclex import NRCLex
import pandas as pd
import matplotlib.pyplot as plt


def main():
    sample_data = pd.read_csv('Tweets.csv')["text"]
    bigString = ",".join(sample_data)
    
    text_stuff = NRCLex(bigString)
    results = text_stuff.raw_emotion_scores
    
    labels = list(results.keys())
    data = list(results.values())

    # fig,ax=plt.subplots(1,1,dpi=135)
    plt.figure(figsize=(10, 3))
    plt.bar(range(len(results)), data, tick_label=labels)
    plt.show()


if __name__ == '__main__':
    main()