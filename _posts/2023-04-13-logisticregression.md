---
layout: post
title:  "파이썬 로지스틱 회귀 "
published: true
---
파이썬을 이용한 로지스틱 회귀 분류기 튜토리얼

이 커널에서는 파이썬과 사이킷런을 사용하여 로지스틱 회귀를 구현합니다. 
오늘 오후에 오스트레일리아에서 비가 올지 여부를 예측하기 위해 로지스틱 회귀 분류기를 구축합니다. 
로지스틱 회귀를 사용하여 이진 분류 모델을 학습합니다.

## 1. 로지스틱 회귀 소개

데이터 과학자들이 새로운 분류 문제를 마주하게 되면, 가장 먼저 떠오르는 알고리즘은 로지스틱 회귀일 수 있습니다. 로지스틱 회귀는 관측값을 이산적인 클래스 집합으로 예측하는 데 사용되는 지도 학습 분류 알고리즘입니다. 실제로, 관측값을 다른 카테고리로 분류하는 데 사용됩니다. 따라서, 로지스틱 회귀의 출력은 이산적인 특성을 가집니다. 로지스틱 회귀는 로짓 회귀(Logit Regression)로도 불립니다. 분류 문제를 해결하기 위해 사용되는 가장 간단하고 직관적이며 다목적 분류 알고리즘 중 하나입니다.

## 2. 로지스틱 회귀 이해

통계학에서 로지스틱 회귀 모델은 주로 분류 목적으로 사용되는 널리 사용되는 통계 모델입니다. 즉, 주어진 관측값 집합에 대해 로지스틱 회귀 알고리즘은 이러한 관측값을 두 개 이상의 이산적인 클래스로 분류하는 데 도움을 줍니다. 따라서 대상 변수는 이산적인 특성을 가집니다.

로지스틱 회귀 알고리즘은 다음과 같이 작동합니다.

선형 방정식 구현하기

로지스틱 회귀 알고리즘은 독립 변수 또는 설명 변수로 선형 방정식을 구현하여 반응 값을 예측합니다. 예를 들어, 우리는 공부한 시간과 시험 통과 확률의 예를 고려할 수 있습니다. 여기서, 공부한 시간은 설명 변수이며 x1로 표시됩니다. 시험 통과 확률은 반응 또는 대상 변수이며 z로 표시됩니다.설명 변수(x1)가 하나이고 반응 변수(z)가 하나인 경우, 선형 방정식은 다음 수식으로 수학적으로 표시됩니다.

z = β0 + β1x1

여기서, 계수 β0와 β1은 모델의 매개 변수입니다.
만약 설명 변수가 여러 개이면, 위의 방정식은 다음과 같이 확장될 수 있습니다.

z = β0 + β1x1+ β2x2+……..+ βnxn

여기서, 계수 β0, β1, β2 및 βn은 모델의 매개 변수입니다. 따라서 예측된 반응 값은 위의 방정식으로 주어지며 z로 표시됩니다.

시그모이드 함수

예측된 반응 값을 z로 표시하고, 이 값을 0과 1 사이의 확률 값으로 변환합니다. 예측 값에서 확률 값으로 매핑하기 위해 시그모이드 함수를 사용합니다. 이 시그모이드 함수는 어떤 실수 값을 0과 1 사이의 확률 값으로 매핑합니다.머신 러닝에서는 시그모이드 함수를 사용하여 예측 값을 확률 값으로 매핑합니다. 시그모이드 함수는 S 모양의 곡선을 가지며, sigmoid curve로도 불립니다.시그모이드 함수는 로지스틱 함수의 특수한 경우입니다. 이는 다음 수학적 공식으로 나타낼 수 있습니다.시각적으로, 시그모이드 함수는 다음 그래프로 나타낼 수 있습니다.

결정 경계

시그모이드 함수는 0과 1 사이의 확률 값을 반환합니다. 이 확률 값은 "0" 또는 "1"인 이산적인 클래스에 매핑됩니다. 이 이산적인 클래스로 확률 값을 매핑하기 위해 결정 경계라는 임계값을 선택합니다. 결정 경계 값 이상이면 확률 값을 클래스 1로 매핑하고, 결정 경계 값 미만이면 확률 값을 클래스 0으로 매핑합니다.

수학적으로 다음과 같이 표현할 수 있습니다:

p ≥ 0.5 => 클래스 = 1
p < 0.5 => 클래스 = 0

일반적으로, 결정 경계는 0.5로 설정됩니다. 따라서 확률 값이 0.8 (> 0.5)인 경우, 이 관측치를 클래스 1로 매핑합니다. 마찬가지로, 확률 값이 0.2 (< 0.5)인 경우, 이 관측치를 클래스 0으로 매핑합니다. 이것은 아래 그래프에서 나타낼 수 있습니다.

이제, 로지스틱 회귀에서 시그모이드 함수와 결정 경계에 대해 알게 되었습니다. 시그모이드 함수와 결정 경계에 대한 지식을 사용하여 예측 함수를 작성할 수 있습니다. 로지스틱 회귀의 예측 함수는 관측치가 긍정적인 (Yes 또는 True)일 확률을 반환합니다. 이를 class 1로 표시하며, P(class = 1)로 표기합니다. 확률이 1에 가까워질수록 관측치가 class 1에 속할 가능성이 높아지며, 그렇지 않으면 class 0에 속합니다.

## 3. 로지스틱 회귀 분석의 가정

이 로지스틱 회귀 모델은 몇 가지 주요 가정이 필요합니다. 다음과 같습니다.

1. 로지스틱 회귀 모델은 종속 변수가 이항, 다항 또는 순서형이어야합니다.

2. 관측치는 서로 독립적이어야합니다. 따라서 관측치는 반복 측정에서 비롯되어서는 안됩니다.

3. 로지스틱 회귀 알고리즘은 독립 변수 간 다중 공선성이 적거나 전혀 없어야합니다. 즉, 독립 변수는 서로 너무 높게 상관되어서는 안됩니다.

4. 로지스틱 회귀 모델은 독립 변수와 로그 오즈의 선형성을 가정합니다.

5. 로지스틱 회귀 모델의 성공은 표본 크기에 따라 다릅니다. 일반적으로 높은 정확도를 달성하기 위해서는 큰 표본 크기가 필요합니다.


## 4. 로지스틱 회귀 분석의 유형

로지스틱 회귀 모델은 대상 변수의 범주에 따라 세 가지 그룹으로 분류될 수 있습니다. 이 세 가지 그룹은 다음과 같이 설명됩니다.

1. 이항 로지스틱 회귀(Binary Logistic Regression)
이항 로지스틱 회귀에서 대상 변수는 두 개의 가능한 범주를 갖습니다. 일반적인 예는 yes 또는 no, good 또는 bad, true 또는 false, spam 또는 no spam, pass 또는 fail 등이 있습니다.

2. 다항 로지스틱 회귀(Multinomial Logistic Regression)
다항 로지스틱 회귀에서 대상 변수는 세 개 이상의 범주를 가집니다. 이 범주들은 특정한 순서가 없으므로 셋 이상의 명목 범주(nominal categories)가 있습니다. 예를 들면 과일의 종류인 사과, 망고, 오렌지, 바나나 등이 있습니다.

3. 순서 로지스틱 회귀(Ordinal Logistic Regression)
순서 로지스틱 회귀에서 대상 변수는 세 개 이상의 순서형 범주(ordinal categories)를 가집니다. 따라서 범주에 내재된 순서가 있습니다. 예를 들어, 학생의 성적은 poor, average, good, excellent와 같이 분류될 수 있습니다.

## 5. 라이브러리 가져오기
```python
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import warnings
warnings.filterwarnings('ignore')
```

## 6. 데이터셋 가져오기
```python
data = '/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv'

df = pd.read_csv(data)
```

## 7. 탐색적 데이터 분석

이제 데이터를 탐색하여 데이터에 대한 통찰력을 얻겠습니다.

```python
df.shape
```

우리는 데이터 세트에 142193개의 인스턴스와 24개의 변수가 있음을 알 수 있습니다.

```python
df.head()
col_names = df.columns
col_names
```

RISK_MM 변수 삭제

데이터 세트 설명에 따르면, RISK_MM 피처 변수를 제거해야합니다. 따라서 다음과 같이 제거해야합니다.

```python
df.drop(['RISK_MM'], axis=1, inplace=True)
df.info()
```

변수 유형

이 섹션에서는 데이터 집합을 범주형 변수와 수치형 변수로 구분합니다. 데이터 집합에는 범주형 변수와 수치형 변수가 혼합되어 있습니다. 범주형 변수는 데이터 유형이 객체입니다. 수치형 변수는 데이터 유형이 float64입니다.

우선, 범주형 변수를 찾겠습니다.

```python
categorical = [var for var in df.columns if df[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :', categorical)
df[categorical].head()
```

범주형 변수 요약
- 날짜 변수가 있습니다. 이것은 Date 열로 표시됩니다.
- 위치(Location), WindGustDir, WindDir9am, WindDir3pm, RainToday, RainTomorrow 6개의 범주형 변수가 있습니다.
- RainToday와 RainTomorrow는 두 개의 이진 범주형 변수입니다.
- RainTomorrow은 목표 변수입니다.

범주형 변수 내의 문제 탐색

먼저 범주형 변수를 탐색해보겠습니다.

범주형 변수 내의 결측값(Missing values)

```python
df[categorical].isnull().sum()
cat1 = [var for var in categorical if df[var].isnull().sum()!=0]
print(df[cat1].isnull().sum())
```

범주형 변수 내 결측값 탐색 결과, WindGustDir, WindDir9am, WindDir3pm 및 RainToday 변수에만 결측값이 있는 것으로 확인됩니다.

범주형 변수의 빈도수

이제 범주형 변수의 빈도수를 확인하겠습니다.

```python
for var in categorical: 
    print(df[var].value_counts())
for var in categorical: 
    print(df[var].value_counts()/np.float(len(df)))
```

레이블 수: 기수(cardinality)

범주형 변수 내 레이블 수를 기수(cardinality)라고 합니다. 변수 내 레이블 수가 높으면 고기수(high cardinality)라고합니다. 고기수는 머신러닝 모델에서 일부 심각한 문제를 일으킬 수 있습니다. 따라서, 고기수를 확인하겠습니다.

```python
for var in categorical:
    print(var, ' contains ', len(df[var].unique()), ' labels')
```

우리는 Date 변수가 전처리 되어야 함을 알 수 있습니다. 다음 섹션에서 전처리를 수행하겠습니다.

다른 모든 변수는 상대적으로 적은 수의 변수를 포함합니다.

Date 변수의 Feature Engineering

```python
df['Date'].dtypes
```

Date 변수의 데이터 유형이 객체(object)임을 확인할 수 있습니다. object로 인코딩된 날짜를 datetime 형식으로 파싱하겠습니다.

```python
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Year'].head()
df['Month'] = df['Date'].dt.month
df['Month'].head()
df['Day'] = df['Date'].dt.day
df['Day'].head()
df.info()
```

Date 변수에서 생성된 세 개의 추가 열이 있음을 확인할 수 있습니다. 이제 데이터 집합에서 원래 Date 변수를 삭제하겠습니다.

```python
df.drop('Date', axis=1, inplace = True)
df.head()
```

이제 데이터 집합에서 Date 변수가 삭제된 것을 확인할 수 있습니다.

범주형 변수 탐색

이제 하나씩 범주형 변수를 탐색하겠습니다

```python
categorical = [var for var in df.columns if df[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :', categorical)
```

데이터 집합에는 6 개의 범주형 변수가 있습니다. Date 변수가 제거되었습니다. 먼저 범주형 변수에서 결측값을 확인하겠습니다.

```python
df[categorical].isnull().sum()
```

WindGustDir, WindDir9am, WindDir3pm, RainToday 변수에는 결측값이 포함되어 있습니다. 하나씩 탐색해보겠습니다.

Location 변수 탐색

```python
print('Location contains', len(df.Location.unique()), 'labels')
df.Location.unique()
df.Location.value_counts()
pd.get_dummies(df.Location, drop_first=True).head()
```

WindGustDir 변수 탐색

```python
print('WindGustDir contains', len(df['WindGustDir'].unique()), 'labels')
df['WindGustDir'].unique()
df.WindGustDir.value_counts()
pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()
pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).sum(axis=0)
```

WindDir9am 변수 탐색

```python
print('WindDir9am contains', len(df['WindDir9am'].unique()), 'labels')
df['WindDir9am'].unique()
df['WindDir9am'].value_counts()
pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()
pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).sum(axis=0)
```

WindDir9am 변수에 결측값이 10013개 있음을 알 수 있습니다.

WindDir3pm 변수 탐색

```python
print('WindDir3pm contains', len(df['WindDir3pm'].unique()), 'labels')
df['WindDir3pm'].unique()
df['WindDir3pm'].value_counts()
pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()
pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).sum(axis=0)
```

WindDir3pm 변수에는 3778개의 결측값이 있습니다.

RainToday 변수 탐색

```python
print('RainToday contains', len(df['RainToday'].unique()), 'labels')
df['RainToday'].unique()
df.RainToday.value_counts()
pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()
pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).sum(axis=0)
```

RainToday 변수에는 1406개의 결측값이 있습니다.

수치 변수 탐색

```python
numerical = [var for var in df.columns if df[var].dtype!='O']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)
df[numerical].head()
```

수치형 변수 요약

총 16 개의 수치형 변수가 있습니다.

MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustSpeed, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am 및 Temp3pm입니다.

모든 수치형 변수는 연속형입니다.

수치형 변수 내 문제점 탐색

이제 수치형 변수를 탐색하겠습니다.

수치형 변수에서 결측값 찾기

```python
df[numerical].isnull().sum()
```

16개의 수치 변수에 결측값이 모두 포함되어 있음을 알 수 있습니다.

수치형 변수의 이상치 탐색

```python
print(round(df[numerical].describe()),2)
```

자세히 살펴보면, Rainfall, Evaporation, WindSpeed9am 및 WindSpeed3pm 열에는 이상치가 있을 수 있습니다.

위 변수들에서 이상치를 시각화하기 위해 상자그림을 그려보겠습니다.

```python
plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')
plt.subplot(2, 2, 2)
fig = df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')
plt.subplot(2, 2, 3)
fig = df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')
plt.subplot(2, 2, 4)
fig = df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')
```

위의 상자 그림은 이러한 변수에 특이치가 많다는 것을 확인합니다.

변수의 분포 확인

이제 분포를 확인하기 위해 히스토그램을 그려볼 것입니다. 정규 분포를 따르면 극단값 분석을 하겠습니다. 그렇지 않은 경우에는 IQR (Interquantile range)을 찾아볼 것입니다.

```python
plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')
plt.subplot(2, 2, 2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')
plt.subplot(2, 2, 3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')
plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')
```

모든 네 개의 변수가 치우쳐져(skewed) 있습니다. 그래서 IQR(Interquantile range)을 사용하여 이상치를 찾겠습니다.

```python
IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

강수량(Rainfall) 변수에서 최소값과 최대값은 각각 0.0과 371.0입니다. 따라서, 이상치는 3.2보다 큰 값들입니다.

```python
IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)
print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

Evaporation의 최소값과 최대값은 각각 0.0과 145.0입니다. 이에 따라, 이상치는 21.8보다 큰 값입니다.

```python
IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)
print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

WindSpeed9am의 최솟값과 최댓값은 각각 0.0과 130.0입니다. 이에 따라 이상치는 값이 55.0보다 큰 값입니다.

```python
IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)
print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

WindSpeed3pm의 최소값은 0.0이고 최대값은 87.0입니다. 따라서, 이상치는 57.0보다 큰 값입니다.

## 8. 특성 벡터와 목표 변수 선언

```python
X = df.drop(['RainTomorrow'], axis=1)
y = df['RainTomorrow']
```

## 9. 데이터를 훈련 세트와 테스트 세트로 분할

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train.shape, X_test.shape
```

## 10. 특성 엔지니어링

특성 엔지니어링은 원시 데이터를 유용한 피처로 변환하여 모델을 더 잘 이해하고 예측 능력을 향상시키는 과정입니다. 서로 다른 유형의 변수에 대해 특성 엔지니어링을 수행할 것입니다.

먼저 범주형 변수와 수치형 변수를 다시 따로 표시하겠습니다.

```python
X_train.dtypes
categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
categorical
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']
numerical
```

숫자 변수의 결측값 엔지니어링

```python
X_train[numerical].isnull().sum()
X_test[numerical].isnull().sum()
for col in numerical:
    if X_train[col].isnull().mean()>0:
        print(col, round(X_train[col].isnull().mean(),4))
```

가정:
데이터가 완전히 랜덤으로 누락되었다고 가정합니다(MCAR). 결측값을 대치하는 데 사용될 수 있는 두 가지 방법이 있습니다. 하나는 평균이나 중간값 대치이고, 다른 하나는 무작위 표본 대치입니다. 데이터셋에 이상치가 있는 경우 중간값 대치를 사용해야 합니다. 따라서 중간값 대치를 사용할 것입니다. 중간값 대치는 이상치에 강건하기 때문입니다.

저는 적절한 통계적 측정치, 즉 중간값으로 결측값을 대체할 것입니다. 결측값 대치는 학습 세트에서 수행한 후, 테스트 세트로 전파되어야 합니다. 즉, 결측값을 채우기 위해 사용되는 통계적 측정치는 학습 세트에서만 추출되어 학습 세트와 테스트 세트 모두에 적용되어야 합니다. 이는 과적합을 피하기 위한 것입니다.

```python
for df1 in [X_train, X_test]:
    for col in numerical:
        col_median=X_train[col].median()
        df1[col].fillna(col_median, inplace=True)           
X_train[numerical].isnull().sum()
X_test[numerical].isnull().sum()
```

지금은 학습 세트와 테스트 세트의 숫자 열에서 결측값이 없는 것으로 보입니다.

범주형 변수에서 결측값 처리 방법에 대해 알아봅시다.

```python
X_train[categorical].isnull().mean()
for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))
for df2 in [X_train, X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)
X_train[categorical].isnull().sum()
X_test[categorical].isnull().sum()
```

마지막으로 X_train 및 X_test에서 결측값이 있는지 확인하겠습니다.

```python
X_train.isnull().sum()
X_test.isnull().sum()
```

X_train 및 X_test에서 결측값이 없는 것을 확인할 수 있습니다.

숫자 변수의 이상치 처리

Rainfall, Evaporation, WindSpeed9am 및 WindSpeed3pm 열에 이상치가 있다는 것을 알았습니다. 이러한 변수에서 최대 값을 제한하고 이상치를 제거하기 위해 상위 코딩(top-coding) 접근법을 사용할 것입니다.

```python
def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])
for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)
X_train.Rainfall.max(), X_test.Rainfall.max()
X_train.Evaporation.max(), X_test.Evaporation.max()
X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()
X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()
X_train[numerical].describe()
```

이제 Rainfall, Evaporation, WindSpeed9am 및 WindSpeed3pm 열의 이상치가 최대 값으로 제한된 것을 볼 수 있습니다.

범주형 변수 인코딩

```python
categorical
X_train[categorical].head()
import category_encoders as ce
encoder = ce.BinaryEncoder(cols=['RainToday'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
X_train.head()
```

RainToday 변수에서 RainToday_0과 RainToday_1이라는 두 가지 추가 변수가 생성된 것을 볼 수 있습니다.

이제 X_train 학습 세트를 만들겠습니다.

```python
X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location), 
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)
X_train.head()
```

마찬가지로 X_test 테스트 세트를 만들겠습니다.

```python
X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_test.Location), 
                     pd.get_dummies(X_test.WindGustDir),
                     pd.get_dummies(X_test.WindDir9am),
                     pd.get_dummies(X_test.WindDir3pm)], axis=1)
X_test.head()
```

이제 모델 구축을 위한 학습 및 테스트 세트가 준비되었습니다. 그 전에, 모든 특성 변수를 동일한 척도로 매핑해야 합니다. 이를 feature scaling이라고 합니다. 다음과 같이 수행하겠습니다.

## 11. 특성 스케일링 (Feature Scaling)

```python
X_train.describe()
cols = X_train.columns
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
X_train.describe()
```
이제 Logistic Regression 분류기에 입력할 준비가 된 X_train 데이터셋이 있습니다. 다음과 같이 수행하겠습니다.

## 12. 모델 학습

```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='liblinear', random_state=0)
logreg.fit(X_train, y_train)
```

## 13. 결과 예측

```python
y_pred_test = logreg.predict(X_test)
y_pred_test
```

predict_proba 메서드는 배열 형태로 대상 변수(이 경우 0과 1)의 확률을 제공합니다.

0은 비가 오지 않을 확률, 1은 비가 올 확률입니다.

```python
logreg.predict_proba(X_test)[:,0]
logreg.predict_proba(X_test)[:,1]
```

## 14. 정확도 점수 확인

```python
from sklearn.metrics import accuracy_score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
```

여기서, y_test는 테스트 세트의 실제 클래스 레이블이고 y_pred_test는 예측된 클래스 레이블입니다.

학습 세트와 테스트 세트의 정확도 비교

이제 학습 세트와 테스트 세트의 정확도를 비교하여 과적합 여부를 확인하겠습니다.

```python
y_pred_train = logreg.predict(X_train)
y_pred_train
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
```

과적합과 과소적합 확인하기

```python
print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))
```

학습 세트의 정확도 점수는 0.8476이고, 테스트 세트의 정확도 점수는 0.8501입니다. 이 두 값은 매우 비슷합니다. 따라서, 과적합의 문제가 없습니다.

로지스틱 회귀에서는 C = 1의 기본값을 사용합니다. 이는 학습 세트와 테스트 세트 모두 약 85%의 정확도를 제공합니다. 그러나, 학습 세트와 테스트 세트의 모델 성능이 매우 비슷합니다. 이는 과소적합의 가능성이 있습니다.

따라서, C를 증가시켜 보다 유연한 모델을 적합해 보겠습니다.

```python
logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)
logreg100.fit(X_train, y_train)
print('Training set score: {:.4f}'.format(logreg100.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(logreg100.score(X_test, y_test)))
```

C=100으로 설정했을 때 테스트 세트 정확도가 더 높으며 약간의 향상된 훈련 세트 정확도를 보입니다. 따라서 더 복잡한 모델이 더 잘 수행될 것으로 결론을 내릴 수 있습니다.

이제 C=1의 기본값보다 더 정규화된 모델을 사용하여 조사해 보겠습니다. C=0.01로 설정합니다.

```python
logreg001 = LogisticRegression(C=0.01, solver='liblinear', random_state=0)
logreg001.fit(X_train, y_train)
print('Training set score: {:.4f}'.format(logreg001.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(logreg001.score(X_test, y_test)))
```

만약 C=0.01과 같이 보다 규제화된 모델을 사용한다면, 기본 매개변수 대비 학습 및 테스트 세트의 정확도가 감소합니다.

모델 정확도를 null 정확도와 비교합니다. null 정확도는 항상 가장 빈번한 클래스를 예측하는 것으로 얻을 수 있는 정확도입니다.

따라서 먼저 테스트 세트에서 클래스 분포를 확인해야합니다.

```python
y_test.value_counts()
```

가장 빈번한 클래스의 발생 횟수는 22067입니다. 따라서, 전체 발생 횟수로 나누어 22067을 계산하여 Null Accuracy를 계산할 수 있습니다.

```python
null_accuracy = (22067/(22067+6372))
print('Null accuracy score: {0:0.4f}'. format(null_accuracy))
```

모델 정확도 점수가 0.8501이고 널 정확도 점수가 0.7759임을 확인할 수 있습니다. 따라서 로지스틱 회귀 모델이 클래스 레이블을 예측하는 데 매우 잘 작동하고 있다고 결론 짓을 수 있습니다.

이제 위의 분석을 바탕으로 우리의 분류 모델 정확도가 매우 우수하다는 결론을 내릴 수 있습니다. 우리 모델은 클래스 레이블을 예측하는 데 매우 잘 작동하고 있습니다.

하지만 이는 값의 분포에 대한 정보를 제공하지 않습니다. 또한 분류기가 만드는 오류 유형에 대해서도 알려주지 않습니다.

이를 보완하기 위한 Confusion matrix라는 또 다른 도구가 있습니다.

## 15. Confusion matrix

혼동 행렬은 분류 알고리즘의 성능을 요약하는 도구입니다. 혼동 행렬은 분류 모델의 성능과 모델이 생성하는 오류 유형에 대한 명확한 그림을 제공합니다. 이는 각 범주별로 올바른 및 부정확한 예측을 요약한 것입니다. 이 요약은 표 형태로 나타납니다.

분류 모델 성능을 평가하는 동안 네 가지 결과가 가능합니다. 이 네 가지 결과는 아래와 같이 설명됩니다.

True Positives (TP) – True Positives는 관측치가 특정 클래스에 속한다고 예측하고, 관측치가 실제로 그 클래스에 속하는 경우 발생합니다.

True Negatives (TN) – True Negatives는 관측치가 특정 클래스에 속하지 않는다고 예측하고, 관측치가 실제로 그 클래스에 속하지 않는 경우 발생합니다.

False Positives (FP) – False Positives는 관측치가 특정 클래스에 속한다고 예측하지만, 관측치가 실제로 그 클래스에 속하지 않는 경우 발생합니다. 이러한 유형의 오류는 제 1종 오류라고 합니다.

False Negatives (FN) – False Negatives는 관측치가 특정 클래스에 속하지 않는다고 예측하고, 관측치가 실제로 그 클래스에 속하는 경우 발생합니다. 이는 매우 심각한 오류이며 제 2종 오류라고 합니다.

이 네 가지 결과는 아래에 제공된 혼동 행렬에서 요약됩니다.

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])
```

혼동 행렬은 20892 + 3285 = 24177개의 올바른 예측과 3087 + 1175 = 4262개의 잘못된 예측을 보여줍니다.

이 경우, 다음과 같습니다.

- True Positives (실제 Positive:1 및 예측 Positive:1) - 20892
- True Negatives (실제 Negative:0 및 예측 Negative:0) - 3285
- False Positives (실제 Negative:0 but 예측 Positive:1) - 1175 (1형 오류)
- False Negatives (실제 Positive:1 but 예측 Negative:0) - 3087 (2형 오류)

```python
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
```

## 16. 분류 지표

분류 보고서

분류 보고서는 분류 모델의 성능을 평가하는 또 다른 방법입니다. 모델의 정밀도(precision), 재현율(recall), f1 점수 및 지원(support) 점수를 표시합니다. 나중에 이 용어들에 대해 설명하겠습니다.

분류 보고서는 다음과 같이 출력할 수 있습니다.

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_test))
```

분류 정확도 (Classification accuracy)

```python
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
```

Classification error

```python
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('Classification error : {0:0.4f}'.format(classification_error))
```

정밀도

정밀도는 예측한 양성 중 올바르게 예측한 비율입니다. 이는 true positives(TP)를 true positives와 false positives(TP+FP)의 합으로 나눈 비율로 표시할 수 있습니다.

따라서, 정밀도는 올바르게 예측한 양성 결과의 비율을 나타냅니다. 이는 음성 클래스보다는 양성 클래스에 더 관심이 있습니다.

수학적으로, 정밀도는 TP/(TP+FP)로 정의될 수 있습니다.

```python
precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))
```

Recall

Recall는 모든 실제 양성 결과 중에서 올바르게 예측된 양성 결과의 백분율로 정의될 수 있습니다. 진양성(True positives, TP)을 실제 양성(True positives, TP)과 거짓 음성(False negatives, FN)의 합(TP + FN)으로 나눈 것으로 나타낼 수 있습니다. 민감도(Sensitivity)라고도 불립니다.

Recall은 실제 양성의 비율을 올바르게 예측합니다.

수학적으로, Recall은 TP를 (TP + FN)으로 나눈 비율로 나타낼 수 있습니다.

```python
recall = TP / float(TP + FN)
print('Recall or Sensitivity : {0:0.4f}'.format(recall))
```

True Positive Rate

True Positive Rate은 Recall과 동의어입니다.

```python
true_positive_rate = TP / float(TP + FN)
print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
```

False Positive Rate

```python
false_positive_rate = FP / float(FP + TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
```

Specificity

```python
specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))
```

f1-점수는 정밀도와 재현율의 가중조화평균입니다. 가장 좋은 f1-점수는 1.0이며, 가장 나쁜 점수는 0.0입니다. f1-점수는 정밀도와 재현율의 조화평균입니다. 따라서 f1-점수는 정확도 측정에 비해 항상 낮습니다. 가중 평균 f1-점수는 분류기 모델을 비교하는 데 사용해야하며, 전체 정확도보다 더 중요합니다.

Support는 데이터 세트에서 클래스의 실제 발생 횟수입니다.

## 17. 임계값 조정

```python
y_pred_prob = logreg.predict_proba(X_test)[0:10]
y_pred_prob
```

Observations
- 각 행에서 숫자는 1의 합계를 이룹니다.
- 2개의 열은 0과 1의 2개의 클래스에 해당합니다.

  - 클래스 0 - 내일 비가 오지 않을 확률의 예측 확률.

  - 클래스 1 - 내일 비가 올 확률의 예측 확률.

- 예측 확률의 중요성

  - 우리는 강수 또는 비가 오지 않을 확률에 따라 관측치를 순위별로 나눌 수 있습니다.

- predict_proba 과정

  - 확률을 예측합니다.

  - 가장 높은 확률의 클래스를 선택합니다.

- 분류 임계값

  - 분류 임계값은 0.5입니다.

  - 확률 > 0.5이면 클래스 1 - 비가 올 확률이 예측됩니다.

  - 확률 < 0.5이면 클래스 0 - 비가 오지 않을 확률이 예측됩니다.

