# Sentiment Analysis for Twitter US Airline Review Project
This is a GitHub repository for Twitter US Airline Review Sentiment Analysis Project.

Understanding language such *big textual data* and *spoken language or speech* can be very *tricky* since it contains ambiguity and can be differ from one analyst to another. This makes **Natural Language Processing (NLP)** important because it helps computers to understand text and spoken language and then obtain insights such as sentiment or opinion value (positive or negative emotions).

## Dataset
Text dataset is obtained from [Kaggle](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) contains 14000 tweets of US airline services reviews and manually annotated sentiment value (positive, neutral or negative) done by dataset contributors.

| text| airline_sentiment|
| ------------- |:-------------:|
| @VirginAmerica What @dhepburn said| neutral |
| @VirginAmerica plus you've added commercials t...| positive|
| @VirginAmerica I didn't today...Must mean I n...| neutral|

## Text Preprocessing
Perform text preprocessing with `RegEx` module and tokenization
* Remove mentions (@VirginAmerica)
* Remove single characters and white spaces
* Convert all text characters into lower-case characters
* Convert texts into words and sequences using Tokenization

## Building the Sentiment Analysis Model using Recurrent Neural Network (RNN)
Text data have *certain patterns* and *dependencies* between words. In order to learn text data, **Long Short Term Memory (LSTM)**, *an artificial recurrent neural network*, is used to learn and understand about those text data and its dependencies between words and also its context. This sentiment analysis model is built using `tensorflow` module.
```python
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout

model = Sequential()
model.add(Embedding(500, 256, input_length=X.shape[1]))
model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
model.add(LSTM(256, dropout=0.3, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))
```
The text dataset (tweets) is split into training data (80%) and validation data (20%). Resulting trained recurrent neural network model achieved **97.86% accuracy on validation dataset.**

## Results
| Confusion Matrix 3 x 3| Real Positive| Real Neutral| Real Negative|
| ------------- |:-------------:|:-------------:| :-------------:|
| Positive Predictions| 440| 0| 19|
| Neutral Predictions| 0| 579| 1|
| Negative Predictions| 0| 0| 1889|
