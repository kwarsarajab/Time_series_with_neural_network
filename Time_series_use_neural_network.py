#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import file
import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt
import tensorflow as tf


# In[ ]:


df = pd.read_csv('../content/drive/MyDrive/Colab Notebooks/testset.csv', parse_dates=['datetime_utc'])
df


# In[ ]:


#Exploratory Data Analysis (EDA) dan Feature Engineering
New_df = df.loc[:,['datetime_utc', ' _tempm']]
New_df = New_df.rename(index=str, columns={'datetime_utc': 'Tanggal', ' _tempm': 'temperatur'})
New_df.head()


# In[ ]:


New_df.isnull().sum()


# In[ ]:


# will fill with previous valid value
New_df.ffill(inplace=True)
New_df[New_df.isnull()].count()


# In[ ]:


New_df.describe()


# In[ ]:


New_df = New_df[New_df.temperatur < 50]


# In[ ]:


temp_train = list(New_df)[1:2]


# In[ ]:


df_for_training = New_df[temp_train].astype(float)


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)


# In[ ]:


trainX =[]
trainY =[]

n_future = 1
n_past = 12

for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])
    
trainX, trainY = np.array(trainX), np.array(trainY)


# In[ ]:


print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))


# In[ ]:


#pemodelan
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

optimizer = tf.keras.optimizers.SGD(lr=1.0000e-04, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

model.summary()


# In[ ]:


#callback
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('mae') < good_mae):
      print("\nmae kurang dari <7%!")
      self.model.stop_training = True
callbacks = myCallback()


# In[ ]:


history = model.fit(trainX, trainY, epochs=30, validation_split=0.2, callbacks=[callbacks])

