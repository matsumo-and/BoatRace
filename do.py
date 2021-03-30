import pandas as pd

xfile = "/Users/osd/Desktop/machinelearning/BoatRace/BoatSche2015-.csv"
yfile = "/Users/osd/Desktop/machinelearning/BoatRace/OBJBoatSche2015-.csv"

#Xデータとして番組表をインポート,RANKをダミーに
df = pd.read_csv(xfile)
df_dummies = pd.get_dummies(df["rank"])

df = df.drop('rank', axis=1)
df = df.drop("RaceID", axis=1)
df = pd.merge(df, df_dummies, how="left", left_index=True, right_index=True)

#Yデータとしてランキング表をインポート
df2 = pd.read_csv(yfile).fillna(0)
df2 = df2.drop("bnum", axis = 1)
#３連複を予想する。3位以上なら1、より下位なら0に置き換える
df2.loc[(df2['Ranking'] > 0) & (df2['Ranking'] < 4), 'triple'] = "1"
df2.loc[~((df2['Ranking'] > 0) & (df2['Ranking'] < 4)), 'triple'] = "0"

df2 = df2.drop("Ranking", axis = 1)
df = pd.merge(df, df2, how="left", left_index=True, right_index=True)
df.astype("float32")

#機械学習
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataX = df.drop(["triple"], axis=1)
dataY = df["triple"]

# この段階で説明変数を標準化しておく
sc = StandardScaler()
dataX_std = pd.DataFrame(sc.fit_transform(dataX), columns=dataX.columns, index=dataX.index)
X_train, X_test, y_train, y_test = train_test_split(dataX_std, dataY, test_size=0.2, stratify=dataY)

from sklearn.linear_model import LogisticRegression

# 分類器を作成（ロジスティック回帰）
clf = LogisticRegression(max_iter=131071)

# 学習
clf.fit(X_train, y_train)

# 予測
y_pred = clf.predict(X_test)

# 正解率を表示
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# 適合率を表示
from sklearn.metrics import precision_score
print(precision_score(y_test, y_pred, average="macro"))

# F値を表示
from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred, average="macro"))

#deeplearning
import keras
from keras import models, layers, regularizers

X_train, y_train, X_test, y_test = train_test_split(dataX_std, dataY, test_size=0.2, random_state=0)

model = models.Sequential()
model.add(layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001), input_shape=(X_train.shape[1], )))
model.add(layers.Dropout(0.49))
model.add(layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.49))
model.add(layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.49))
model.add(layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.49))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

EPOCHS=1000
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(X_train,
                    X_test,
                    epochs=EPOCHS,
                    batch_size=512,
                    validation_split=0.2,
                    callbacks=[early_stop])

score = model.evaluate(y_train, y_test)
print("loss : {}".format(score[0]))
print("Test score : {}".format(score[1]))

history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

#ドットは訓練データを表しており、折れ線は検証データを表しています

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

#”bo”は”blue dot”（青のドット）を意味する
plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
#”b”は"solid blue line"(青の実線）を意味する
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()