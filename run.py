import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# 데이터를 pandas dataframe으로 불러온다
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# print(train.head())

# 크게 관련 없는 columns를 삭제한다
# Ticket, Fare, Cabin, PasengerId, Name
train.drop(["Ticket", "Fare", "Cabin", "PassengerId", "Name"], axis=1, inplace=True)

# NaN값이 들어있는 columns를 확인한다
train.isnull().sum(axis=0)

# 각 column별로 값을 확인한다
for index in train.columns:
    train[index].value_counts()

# 훈련용 데이터를 슬라이싱을 통해 train / validation set으로 분할한다
train_num = 650
val_num = train.shape[0] - train_num

x_train = train.drop("Survived", axis=1)[:train_num]
y_train = train["Survived"][:train_num]
x_val = train.drop("Survived", axis=1)[train_num:]
y_val = train["Survived"][train_num:]

# 넘파이배열로 변환
x_train = x_train.values
y_train = y_train.values
x_val = x_val.values
y_val = y_val.values


# keras모델 설계
model = models.Sequential()
model.add(layers.Dense(16, activation="relu", input_shape=(6,)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    x_train, y_train, epochs=20, batch_size=1, validation_data=(x_val, y_val)
)

history_dict = history.history
history_dict.keys()

