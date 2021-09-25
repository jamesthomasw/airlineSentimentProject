#LOAD LIBRARIES
import tensorflow as tf
import numpy as np
import pandas as pd

#INITIALIZE GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e:
        print(e)

#LOAD DATA
df = pd.read_csv('Tweets.csv')
df.head()

df = df[['text', 'airline_sentiment']]
df.head()

#DATA VISUALIZATION
df['airline_sentiment'].value_counts().sort_index().plot.bar()
df['text'].str.len().plot.hist()
df['text'].str.len().max()

#TEXT PREPROCESSING
import re

df['text'] = df['text'].str.replace('@VirginAmerica', '')
df['text'] = df['text'].apply(lambda x: x.lower())
df['text'] = df['text'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]', '', x))
df.head()

#CONVERT WORD TO INDEX
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=500, split=" ")
tokenizer.fit_on_texts(df['text'].values)

X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X)
X[:5]

#CREATE MODEL
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout

model = Sequential()
model.add(Embedding(500, 256, input_length=X.shape[1]))
model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
model.add(LSTM(256, dropout=0.3, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#PREPARE TRAINING AND TESTING DATA
y = pd.get_dummies(df['airline_sentiment']).values
for i in range(5):
    print(df['airline_sentiment'][i], y[i])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#TRAINING MODEL
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='accuracy',
    restore_best_weights=True,
    min_delta=0,
    patience=5,
    verbose=1,
    mode='auto'
)

history = model.fit(
    X_train, 
    y_train, 
    epochs=50, 
    batch_size=64,
    callbacks=[early_stop]
)

#SAVING TRAINED MODEL
model.save_weights('sentiment_model_weights.h5')
model.save('sentiment_model.h5')

#EVALUATING MODEL ON TEST SET
predictions = model.predict(X_test)

for i in range(5):
    print(df['text'][i], predictions[i], y_test[i])

pos_count, neu_count, neg_count = 0, 0, 0
real_pos, real_neu, real_neg = 0, 0, 0

for i, prediction in enumerate(predictions):
    if np.argmax(prediction)==2:
        pos_count+=1
    elif np.argmax(prediction)==1:
        neu_count+=1
    else:
        neg_count+=1
        
    if np.argmax(y_test[i])==2:
        real_pos+=1
    elif np.argmax(y_test[i])==1:
        real_neu+=1
    else:
        real_neg+=1
        
print('Positive Predictions:', pos_count)
print('Neutral Predictions:', neu_count)
print('Negative Predictions:', neg_count)
print('Real Positive:', real_pos)
print('Real Neutral:', real_neu)
print('Real Negative:', real_neg)