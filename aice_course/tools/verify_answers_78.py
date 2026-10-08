"""7·8회차 실습 정답 코드 검증 (경고를 오류로 취급). 노트북과 같은 환경 설정을 쓴다."""
import os
import platform
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.simplefilter("error", UserWarning)      # 수강생 화면에 보이는 경고만 오류로 취급
warnings.simplefilter("error", FutureWarning)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, recall_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

if platform.system() == "Darwin":
  tf.config.set_visible_devices([], "GPU")
tf.get_logger().setLevel("ERROR")
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

COLS = {"PassengerId": "승객ID", "Survived": "생존", "Pclass": "객실등급", "Name": "이름", "Sex": "성별", "Age": "나이",
        "SibSp": "동반형제배우자", "Parch": "동반부모자녀", "Ticket": "티켓번호", "Fare": "운임", "Cabin": "객실번호", "Embarked": "탑승항구"}
HC = {"longitude": "경도", "latitude": "위도", "housing_median_age": "주택연식", "total_rooms": "총방수", "total_bedrooms": "총침실수",
      "population": "인구", "households": "가구수", "median_income": "소득중앙값", "median_house_value": "주택가격"}


def mock_exam() -> None:
  df = pd.read_csv("data/titanic_train.csv").rename(columns=COLS)
  print("Q1", df.shape)
  high_missing = df.columns[df.isnull().mean() > 0.5].tolist()
  print("Q2", high_missing)
  plt.figure(); sns.countplot(data=df, x="성별", hue="생존"); plt.close()
  print("Q3", df.groupby("객실등급")["생존"].mean().round(3).to_dict())
  corr = df.corr(numeric_only=True)
  plt.figure(); sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm"); plt.close()
  print("Q4", corr["생존"].drop("생존").abs().idxmax())
  df = df.drop(columns=["승객ID", "이름", "티켓번호"] + high_missing)
  df["나이"] = df["나이"].fillna(df.groupby(["성별", "객실등급"])["나이"].transform("median"))
  df["탑승항구"] = df["탑승항구"].fillna(df["탑승항구"].mode()[0])
  print("Q5", df.isnull().sum().sum())
  df["가족수"] = df["동반형제배우자"] + df["동반부모자녀"] + 1
  df["성별"] = df["성별"].map({"male": 0, "female": 1})
  df = pd.get_dummies(df, columns=["탑승항구"], drop_first=True, dtype=int)
  print("Q6", df.shape)
  X, y = df.drop(columns=["생존"]), df["생존"]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
  scaler = StandardScaler()
  X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
  print("Q7", X_train_scaled.shape)
  rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42).fit(X_train, y_train)
  print("Q8", round(accuracy_score(y_test, rf.predict(X_test)), 4))
  grid = GridSearchCV(RandomForestClassifier(random_state=42), {"max_depth": [4, 6, 8], "n_estimators": [100, 300]},
                      cv=5, scoring="accuracy", n_jobs=-1).fit(X_train, y_train)
  print("Q9", grid.best_params_, round(grid.best_score_, 4), round(accuracy_score(y_test, grid.best_estimator_.predict(X_test)), 4))
  keras.utils.set_random_seed(42)
  dnn = keras.Sequential([keras.Input(shape=(X_train_scaled.shape[1],)), layers.Dense(64, activation="relu"), layers.Dropout(0.2),
                          layers.Dense(32, activation="relu"), layers.Dropout(0.2), layers.Dense(1, activation="sigmoid")])
  dnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
  es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
  history = dnn.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es], verbose=0)
  print("Q10 epochs", len(history.history["loss"]))
  print("Q11", int(np.argmin(history.history["val_loss"])) + 1)
  rows = []
  for name, proba in [("RF", grid.best_estimator_.predict_proba(X_test)[:, 1]), ("DNN", dnn.predict(X_test_scaled, verbose=0).ravel())]:
    pred = (proba >= 0.5).astype(int)
    rows.append({"모델": name, "acc": accuracy_score(y_test, pred), "rec": recall_score(y_test, pred),
                 "f1": f1_score(y_test, pred), "auc": roc_auc_score(y_test, proba)})
  print("Q12", pd.DataFrame(rows).round(4).to_dict("records"))


def session7() -> None:
  t = pd.read_csv("data/titanic_train.csv").rename(columns=COLS).drop(columns=["승객ID", "이름", "티켓번호", "객실번호"])
  t["나이"] = t["나이"].fillna(t["나이"].median())
  t["탑승항구"] = t["탑승항구"].fillna(t["탑승항구"].mode()[0])
  t["가족수"] = t["동반형제배우자"] + t["동반부모자녀"] + 1
  t["혼자탑승"] = (t["가족수"] == 1).astype(int)
  t["성별"] = t["성별"].map({"male": 0, "female": 1})
  t = pd.get_dummies(t, columns=["탑승항구"], drop_first=True, dtype=int)
  a, b, ya, yb = train_test_split(t.drop(columns=["생존"]), t["생존"], test_size=0.2, random_state=42, stratify=t["생존"])
  sc = StandardScaler()
  a, b = sc.fit_transform(a).astype("float32"), sc.transform(b).astype("float32")

  keras.utils.set_random_seed(0)
  dnn1 = keras.Sequential([keras.Input(shape=(a.shape[1],)), layers.Dense(64, activation="relu"),
                           layers.Dense(32, activation="relu"), layers.Dense(1, activation="sigmoid")])
  print("7-Q1 params", dnn1.count_params())
  dnn1.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
  dnn1.fit(a, ya, epochs=40, batch_size=32, validation_split=0.2, verbose=0)
  _, acc = dnn1.evaluate(b, yb, verbose=0)
  p1 = (dnn1.predict(b, verbose=0).ravel() >= 0.5).astype(int)
  print("7-Q3", round(acc, 4), round(f1_score(yb, p1), 4))

  keras.utils.set_random_seed(0)
  dnn2 = keras.Sequential([keras.Input(shape=(a.shape[1],)), layers.Dense(64, activation="relu"), layers.Dropout(0.2),
                           layers.Dense(32, activation="relu"), layers.Dropout(0.2), layers.Dense(1, activation="sigmoid")])
  dnn2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
  cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("data/best_dnn2.keras", save_best_only=True)]
  h2 = dnn2.fit(a, ya, epochs=200, batch_size=32, validation_split=0.2, callbacks=cb, verbose=0)
  print("7-Q4", len(h2.history["loss"]), round(dnn2.evaluate(b, yb, verbose=0)[1], 4))
  os.remove("data/best_dnn2.keras")

  h = pd.read_csv("data/california_housing_train.csv").rename(columns=HC)
  h["가구당방수"] = h["총방수"] / h["가구수"]
  h["침실비율"] = h["총침실수"] / h["총방수"]
  h["가구당인구"] = h["인구"] / h["가구수"]
  ra, rb, rya, ryb = train_test_split(h.drop(columns=["주택가격"]), h["주택가격"], test_size=0.2, random_state=42)
  s2 = StandardScaler()
  ra, rb = s2.fit_transform(ra).astype("float32"), s2.transform(rb).astype("float32")
  keras.utils.set_random_seed(0)
  dr = keras.Sequential([keras.Input(shape=(ra.shape[1],)), layers.Dense(128, activation="relu"),
                         layers.Dense(64, activation="relu"), layers.Dense(1)])
  dr.compile(optimizer="adam", loss="mse", metrics=["mae"])
  dr.fit(ra, rya / 100_000, epochs=50, batch_size=256, validation_split=0.2, verbose=0,
         callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
  p = dr.predict(rb, verbose=0).ravel() * 100_000
  print("7-Q5", int(np.sqrt(mean_squared_error(ryb, p))), round(r2_score(ryb, p), 4))


if __name__ == "__main__":
  mock_exam()
  session7()
  print("all ok")
