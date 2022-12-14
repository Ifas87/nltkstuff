import pandas as pd
import tensorflow
import matplotlib.pyplot as plt


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.python.keras.layers import Embedding


def sentiment_analysis(text, tokenisers, model, sentiment_label):
    tw = tokenisers.texts_to_sequences([text])
    tw = tensorflow.keras.preprocessing.sequence.pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    print("Predicted label: ", sentiment_label[1][prediction])


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
    sentiment_label = relevant_sample['airline_sentiment'].factorize()
    nptext = relevant_sample['text'].values
    tokenisers = tensorflow.keras.preprocessing.text.Tokenizer(num_words=5000)
    tokenisers.fit_on_texts(nptext)

    encoded_result = tokenisers.texts_to_sequences(nptext)
    padded_result = tensorflow.keras.preprocessing.sequence.pad_sequences(encoded_result, maxlen=200)

    len_vocab = len(tokenisers.word_index) + 1

    embedding_vector_length = 32
    model = Sequential()
    model.add(Embedding(len_vocab, embedding_vector_length, input_length=200))
    model.add(SpatialDropout1D(0.25))
    model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

    history = model.fit(padded_result, sentiment_label[0], validation_split=0.2, epochs=5, batch_size=32)

    
    # plt.plot(history.history['accuracy'], label='acc')
    # plt.plot(history.history['val_accuracy'], label='val_acc')
    # plt.legend()
    # plt.savefig("Accuracy plot.jpg")
    # plt.show()

    # plt.plot(history.history['loss'], label='loss')
    # plt.plot(history.history['val_loss'], label='val_loss')
    # plt.legend()
    # plt.savefig("Loss plt.jpg")
    # plt.show()
    
    example_texts_1 = "I had a good and bad time there."
    sentiment_analysis(example_texts_1, tokenisers, model, sentiment_label)


if __name__ == '__main__':
    main()