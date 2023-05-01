---
layout: post
title:  "UFO 웹앱 구현하기"
published: true
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


# ML모델을 이용한 웹 앱 빌드하기

## 데이터 정리하기
기본 설정 및 데이터 가져오기, 해당 포스터에서는 UFO를 목격했던 데이터들을 불러옵니다.
```python
import pandas as pd
import numpy as np

ufos = pd.read_csv('https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/3-Web-App/1-Web-App/data/ufos.csv')
ufos.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>city</th>
      <th>state</th>
      <th>country</th>
      <th>shape</th>
      <th>duration (seconds)</th>
      <th>duration (hours/min)</th>
      <th>comments</th>
      <th>date posted</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10/10/1949 20:30</td>
      <td>san marcos</td>
      <td>tx</td>
      <td>us</td>
      <td>cylinder</td>
      <td>2700.0</td>
      <td>45 minutes</td>
      <td>This event took place in early fall around 194...</td>
      <td>4/27/2004</td>
      <td>29.883056</td>
      <td>-97.941111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10/10/1949 21:00</td>
      <td>lackland afb</td>
      <td>tx</td>
      <td>NaN</td>
      <td>light</td>
      <td>7200.0</td>
      <td>1-2 hrs</td>
      <td>1949 Lackland AFB&amp;#44 TX.  Lights racing acros...</td>
      <td>12/16/2005</td>
      <td>29.384210</td>
      <td>-98.581082</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10/10/1955 17:00</td>
      <td>chester (uk/england)</td>
      <td>NaN</td>
      <td>gb</td>
      <td>circle</td>
      <td>20.0</td>
      <td>20 seconds</td>
      <td>Green/Orange circular disc over Chester&amp;#44 En...</td>
      <td>1/21/2008</td>
      <td>53.200000</td>
      <td>-2.916667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10/10/1956 21:00</td>
      <td>edna</td>
      <td>tx</td>
      <td>us</td>
      <td>circle</td>
      <td>20.0</td>
      <td>1/2 hour</td>
      <td>My older brother and twin sister were leaving ...</td>
      <td>1/17/2004</td>
      <td>28.978333</td>
      <td>-96.645833</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10/10/1960 20:00</td>
      <td>kaneohe</td>
      <td>hi</td>
      <td>us</td>
      <td>light</td>
      <td>900.0</td>
      <td>15 minutes</td>
      <td>AS a Marine 1st Lt. flying an FJ4B fighter/att...</td>
      <td>1/22/2004</td>
      <td>21.418056</td>
      <td>-157.803611</td>
    </tr>
  </tbody>
</table>
</div>


ufo 데이터를 작은 데이터 프레임으로 변환합니다. 
```python
ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})

ufos.Country.unique()
```

<pre>
array(['us', nan, 'gb', 'ca', 'au', 'de'], dtype=object)
</pre>


null값의 삭제와 관찰된 시간이 1초 이상부터 60초 이내인 데이터만 남기고 전부 삭제합니다. 
```python
ufos.dropna(inplace=True)

ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]

ufos.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
Index: 25863 entries, 2 to 80330
Data columns (total 4 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   Seconds    25863 non-null  float64
 1   Country    25863 non-null  object 
 2   Latitude   25863 non-null  float64
 3   Longitude  25863 non-null  float64
dtypes: float64(3), object(1)
memory usage: 1010.3+ KB
</pre>


국가의 텍스트 값을 숫자로 변환합니다. 
```python
from sklearn.preprocessing import LabelEncoder

ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])

ufos.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Seconds</th>
      <th>Country</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>20.0</td>
      <td>3</td>
      <td>53.200000</td>
      <td>-2.916667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20.0</td>
      <td>4</td>
      <td>28.978333</td>
      <td>-96.645833</td>
    </tr>
    <tr>
      <th>14</th>
      <td>30.0</td>
      <td>4</td>
      <td>35.823889</td>
      <td>-80.253611</td>
    </tr>
    <tr>
      <th>23</th>
      <td>60.0</td>
      <td>4</td>
      <td>45.582778</td>
      <td>-122.352222</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3.0</td>
      <td>3</td>
      <td>51.783333</td>
      <td>-0.783333</td>
    </tr>
  </tbody>
</table>
</div>

## 모델 구축 

```python
from sklearn.model_selection import train_test_split

Selected_features = ['Seconds','Latitude','Longitude']

X = ufos[Selected_features]
y = ufos['Country']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

로지스틱 회귀를 통한 모델 교육합니다. 
```python
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print('Predicted labels: ', predictions)
print('Accuracy: ', accuracy_score(y_test, predictions))
```

<pre>
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        41
           1       0.82      0.24      0.37       250
           2       1.00      1.00      1.00         8
           3       1.00      1.00      1.00       131
           4       0.96      1.00      0.98      4743

    accuracy                           0.96      5173
   macro avg       0.96      0.85      0.87      5173
weighted avg       0.96      0.96      0.95      5173

Predicted labels:  [4 4 4 ... 3 4 4]
Accuracy:  0.9605644693601392
</pre>
<pre>
c:\Users\박건희\contoso\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
</pre>


## 모델 '피클'

피클은 Python 개체 구조를 직렬화 및 역직렬화하는 Python 모듈입니다.
해당 모듈을 통해 위에서 교육시킨 모델을 '피클'화하여 웹에서 사용할 수 있도록 구조를 직렬화하거나 평면화합니다. 
```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

<pre>
[1]
</pre>
<pre>
c:\Users\박건희\contoso\venv\Lib\site-packages\sklearn\base.py:439: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names
  warnings.warn(
</pre>


##Flask 앱 빌드
![경로 지정](https://user-images.githubusercontent.com/117708673/235430221-500a5d08-6f27-44b0-a908-9750a509c6d7.png)

사진에서 보이는 것처럼
1. ufo-model.pkl 파일이 있는 notebook.ipynb 파일 옆에 web-app 이라는 폴더를 생성하여 시작합니다 .
2. 해당 폴더에 css 폴더가 있는 static 폴더 와 templates 폴더를 세 개 더 만듭니다.

```
web-app/
  static/
    css/
  templates/
notebook.ipynb
ufo-model.pkl
```

3. web-app 폴더 에 가장 먼저 생성할 파일은 requirements.txt 파일입니다. JavaScript 앱의 package.json 과 마찬가지로 이 파일은 앱에 필요한 종속성을 나열합니다. requirements.txt 에 다음 줄을 추가합니다.

```
scikit-learn
pandas
numpy
flask
```
