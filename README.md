# AI-X-DeepLearning
## 목차

1. [Proposal](#i-proposal)
2. [Datasets](#ii-datasets)
3. [Methodology](#iii-methodology)
4. [Evaluation & Analysis](#iv-evaluation--analysis)
5. [Related Work](#v-related-work)
6. [Conclusion](#vi-conclusion-discussion)

## Title: 교통체증 구간 예측
## Members
- 신지성 | 한양대 에리카 ICT융합학부 js0429@hanyang.ac.kr
- 권영진 | 한양대 에리카 ICT융합학부 zjekfks@hanyang.ac.kr
- 종예기 | 한양대 에리카 정보사회미디어

## I. Proposal
### Motivation
누구나 자동차를 타다가 종종 밀리는 구간을 만나본 경험이 있을 것이다. 이를 경험적으로 출퇴근 시간이나 사람이 많이 사는 지역 때문에 밀리는 것이라고 추론하곤 하는데 인공지능도 이러한 추론이 가능할 것 같았고, 또한 인공지능 학습에 필요한 특징들이 비교적 뚜렷하고 많을 것으로 예상되어 교통체증 구간 예측을 주제로 정하였다.     
### Goal
우리의 골은 특정한 상황이 주어젔을 때 어느 정도의 교통체증이 일어날지를 예측하는 모델을 만드는 것이고 우리가 일반적으로 생각하지 못했던 교통체증 원인도 발견하길 기대한다. 
## II. Datasets
국내의 교통과 관련된 데이터셋을 만들기 위해 경기도 교통DB, 국가교통 데이터에서 가져온 데이터를 사용한다.  
파이썬의 판다스 모듈을 사용하여 행과 열의 속성을 가지는 자료형인 데이터프레임을 처리할 것이고 가져온 파일 형식은 csv이다  
먼저 행의 항목에 올 것은 읍면동이기 때문에 읍면동을 기준으로 여기저기서 갖고온 데이터들을 병합하는 것이 목표이며 타겟 데이터는 시간대별 교통량이다  

각기 다른 데이터셋마다 읍면동을 표현하는 것이 조금씩 차이가 있기 때문에 읍면동의 데이터들의 형식을 통일해야 한다.  
(읍면동의 데이터 형식을 통일하는 코드 넣을 것)  

import pandas as pd  
df1=pd.read_execl('./일반국도 1-6.xlsx')  
df2=pd.read_execl('./일반국도 17-38.xlsx')  
df3=pd.read_execl('./일반국도 39-43.xlsx')  
df4=pd.read_execl('./일반국도 44-46.xlsx')  
df5=pd.read_execl('./일반국도 47-75.xlsx')  
df6=pd.read_execl('./일반국도 77-87.xlsx')  

df=pd.concat([df1,df2,df3,df4,df,df6])  
df.to_csv('./result.csv')  

df_time=pd.read_csv('./result')  
df_ground=pd.read_csv('./대도시 미개발 토지 정보.csv')  
df_time=df_time[['지점번호','시간대','행정구역','지점번호','전차종합계']]  
df=df.merge(df_ground, how='inner', on='행정구역')  

(결과로 나온 파일은 깃헙에 올릴 것)  
## III. Methodology
### Algorithms

다음은 데이터셋을 분석하고 예측하는 모델을 만들기 위해 사용된 알고리즘이다. <br>

 * <b>나이브 베이즈(Naive Bayes) :</b>  

    &nbsp; 각 변수가 독립적이라는 가정하에 확률에 기반하여 데이터를 분류하는 방법  

    &nbsp; 많은 상황에서 쓰였을 때 일반적으로 좋은 성능을 보이는 알고리즘이기에 하나의 방법으로 채택하였다.  
  

* <b>랜덤 포레스트(Random Forest) :</b>    
      
    &nbsp; <span>의사결정 나무(Decision Tree)의 단점을 보완한 모델로 훈련으로 구성해놓은 다수의 의사결정 나무로부터 분류결과를 취합해 결론을 얻는 방법</span>

    &nbsp; 데이터가 잘 문서화되어있어야 효과적인 모델이므로 이번 프로젝트에 사용되기 부적합 할 수 있지만 다른 모델들과의 비교를 위하여 채택하였다.  <br>
    
  
### Features

  <b>목표값(Target Feature)</b> : 시간별 교통량에 따른 교통체증의 정도

<b>특성(Features)</b>   
  
  * 미개발 토지 : 지역에 대한 미개발 토지의 수 로 지역의 면적대비 활성도로 활용한다

  * 추정 교통량 : 지역의 전체적인 평균 교통량으로 목표값과의 비교목적으로 활용한다

  * 역세권 실거래 정보 : 지역에대한 가치를 간접적으로 평가할 목적으로 활용한다

  * 평균 속도 : 지역에서 차량들의 평균 속도로 도로 상태를 평가할 목적으로 활용한다

## IV. Evaluation & Analysis
### Graphs, tables, any statistics


먼저 판다스로 csv형식의 데이터셋을 읽어온다.
```python
import pandas as pd

df=pd.read_csv("./dataset.csv") # 데이터셋 읽어들이기
```

그런 다음, 이번 경우에는 불완전한 행을 Evaluation 하기에는 데이터의 분포가 불균형적이기때문에 <b>dropna()</b> 함수를 이용해 결측값이 있는 행은 삭제하기로 했다.
```python
df = df.dropna() #결측값이 있는 행 삭제
```

지역 이름은 인덱스와 같은 역할을 하고있기때문에 <b>drop()</b> 함수로 삭제해주고, 목표값과 특성을 정의해준다
```python
df = df.drop('area', axis=1)

X=df.drop('total_vehicle_amount',axis=1)
y=df['total_vehicle_amount']
```

예측 정확도 향상을 위해 목표값을 각각 전체 목표값의 평균으로 나누어준다
```python

df['total_vehicle_amount'] = pd.to_numeric(df['total_vehicle_amount'], errors='coerce') # 문자열로 인식되었기에 숫자로 바꿔주었다
# errors='coerce' 는 숫자가 아닌 항목을 Nan으로 처리해준다

df['total_vehicle_amount'] = df['total_vehicle_amount'] // df['total_vehicle_amount'].mean()
```

학습을 위한 훈련용 데이터와 데스트에 쓰일 데이터를 분리한다
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=50)
```


랜덤포레스트 모델을 불러오고 훈련시킨다
```python
from sklearn.ensemble import RandomForestClassifier

rand_clf = RandomForestClassifier(criterion='entropy', random_state=42)

rand_clf.fit(X_train, y_train)  
rand_predicted = rand_clf.predict(X_test)
```

모델의 정확도를 평가한다
```python
from sklearn.metrics import accuracy_score

rand_accuracy = accuracy_score(y_test, rand_predicted)

print('랜덤 포레스트 정확도: {:.3f}' .format(rand_accuracy))
```

나이브베이스 모델을 불러오고 훈련시킨다
```python
from sklearn.naive_bayes import GaussianNB

NB_clf = GaussianNB()

NB_clf.fit(X_train, y_train)
nb_predicted = NB_clf.predict(X_test)
```

모델의 정확도를 평가한다
```python
from sklearn.metrics import accuracy_score

nb_accuracy = accuracy_score(y_test, nb_predicted)

print('나이브 베이스 정확도: {:.3f}' .format(nb_accuracy))
```

## V. Related Work
### Tools, libraries, blogs, or any documentation that we have used to do this project

## VI. Conclusion: Discussion
### Feel free to use any format we like
