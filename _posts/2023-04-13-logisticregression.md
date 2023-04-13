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
