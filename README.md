# AI-X-DeepLearning
> 동영상 링크: https://youtu.be/TZccazEeNqw
## 목차

1. [Proposal](#i-proposal)
2. [Datasets](#ii-datasets)
3. [Methodology](#iii-methodology)
4. [Evaluation & Analysis](#iv-evaluation--analysis)
5. [Related Work](#v-related-work)
   
## Title: 교통체증 구간 예측
## Members
- 신지성 | 한양대 에리카 ICT융합학부 js0429@hanyang.ac.kr
- 권영진 | 한양대 에리카 ICT융합학부 zjekfks@hanyang.ac.kr
- 종예기 | 한양대 에리카 정보사회미디어 jongyeki@gmail.com

## I. Proposal
### Motivation
누구나 자동차를 타다가 종종 밀리는 구간을 만나본 경험이 있을 것이다. 이를 경험적으로 출퇴근 시간이나 사람이 많이 사는 지역 때문에 밀리는 것이라고 추론하곤 하였다. 인공지능 또한 이러한 추론이 가능할 것 같았고 인공지능 학습에 필요한 특징들이 비교적 뚜렷하고 많을 것으로 예상되어 교통체증 구간 예측을 주제로 정하였다.     
### Goal
우리의 골은 특정한 상황이 주어젔을 때 어느 정도의 교통체증이 일어날지를 예측하는 모델을 만드는 것이고 우리가 일반적으로 생각하지 못했던 교통체증 원인도 발견하길 기대한다. 
## II. Datasets
우리가 원하는 데이터셋은 웹상에서 없기 때문에 여러 국내 DB사이트를 통해 데이터셋을 가져와 원하는 데이터셋을 만들 것이다.  
우리는 경기도 교통DB, 국가교통 데이터에서 가져온 데이터셋을 사용할 것이다.
```
데이터셋 링크  
추정교통량: 주간단위로 특정 행정구역에 나타난 차의 수를 담고 있다
- https://www.bigdata-transportation.kr/frn/prdt/detail?prdtId=PRDTNUM_000000020420  

대도시 미개발 토지 정보: 행정구역별로 미개발 토지 정보를 담고 있다  
- https://www.bigdata-transportation.kr/frn/prdt/detail?prdtId=PRDTNUM_000000020277  

평균속도: 행정구역별로 평균속도 정보를 담고 있다
- https://www.bigdata-transportation.kr/frn/prdt/detail?prdtId=PRDTNUM_000000020421  

역세권 공동주택 실거래 정보: 행정구역별로 역세권 공동주택 실거래 정보를 담고 있다(거래방식, 거래금액 등)
- https://www.bigdata-transportation.kr/frn/prdt/detail?prdtId=PRDTNUM_000000020052  

시간대별 일반국도 교통량 참고 페이지(실시간 일반국도의 교통량 정보를 담고 있다)(파일은 위에 별도로 첨부하겠다)  
- https://gits.gg.go.kr/gtdb/web/trafficDb/trafficVolume/alwaysTrafficVolumeByTime.do  
```

데이터셋들을 정제할 때 파이썬을 이용할 것이고 판다스 모듈을 사용할 것이다.  

### 데이터셋 통합 및 처리
위의 가져온 데이터셋들은 모두 경기도 내의 읍/면/동에 대한 정보를 가지고 있다 따라서 경기도내의 읍/면/동을 기준으로 그룹핑 하여 데이터셋들을 통합할 것이다. 
이때 알고가야 할 점은 데이터셋마다 읍/면/동에 대한 정보는 가지고 있지만 저장 형식에 차이가 있기 때문에 형식을 통일하고 그룹핑 할 것이다.  

먼저 시간대별 일반국도 교통량 데이터셋들을 df_integra.csv 파일로 통합하자  
```python
import pandas as pd  
df1=pd.read_execl('./일반국도 1-6.xlsx')  #경로를 참고하여 파일을 읽어 데이터프레임으로 저장한다
df2=pd.read_execl('./일반국도 17-38.xlsx')  
df3=pd.read_execl('./일반국도 39-43.xlsx')  
df4=pd.read_execl('./일반국도 44-46.xlsx')  
df5=pd.read_execl('./일반국도 47-75.xlsx')  
df6=pd.read_execl('./일반국도 77-87.xlsx')  

df=pd.concat([df1,df2,df3,df4,df,df6])  
df.to_csv('./df_integra.csv')  
```
다음은 대도시 미개발 토지 정보 데이터셋을 정리해보자  
이 데이터셋은 경기도 외의 시/도가 포함되어 있고 읍/면/동에서 끝에 불필요한 코드가 있으니 이것들을 제거해주어야 한다. 또 각 읍/면/동 별로 몇개의 미개발 토지가 있는지의 정보를 추출하면 된다.  
경기도 행은 메모장으로 열어 직접 따로 추출하였고.  
읍/면/동의 불필요한 코드 제거와 토지 개수 추출은 파이썬을 이용해준다  
```python
import pandas as pd
df_ground=pd.read_csv('./대도시 미개발 토지 정보.csv')
for i in range(len(df_ground)):
    tmp=df_ground.loc[i,'행정구역'].split()
    del tmp[-1]
    df_ground.loc[i,'행정구역'] = ' '.join(tmp)
df_ground=df_ground.groupby('행정구역').count()
df_ground[['pnu']].to_csv('./df_ground.csv')
```
다음은 추정교통량 데이터셋을 처리하자 
이 데이터는 읍/면/동에 대한 정보가 한글이 아닌 숫자 코드로 적혀 있기에 코드에 맞게 행정구역을 붙여준다(읍면동 코드2는 따로 첨부했다)  
```python
import pandas as pd
df_might = pd.read_csv('./2019년_추정교통량_행정구역_읍면동단위.csv')
df_code = pd.read_csv('./읍면동 코드2.csv')
df_vel=df_vel[['emd_code','FGCR_AADT']]
df_vel['행정구역']=''
for i in range(len(df_vel)):
    for j in range(len(df_code)):
        if df_vel.loc[i,'emd_code'] == df_code.loc[j,'emd_code']:
            df_vel.loc[i,'행정구역'] = df_code.loc[j,'행정구역']
df_vel.dropna(axis=0, how='any', inplace=True)
df_vel.index=list(range(len(df_vel)))
df_vel.to_csv('./df_might.csv')
```
다음은 평균속도 데이터셋을 처리해보자  
이 데이터 또한 추정교통량 데이터셋과 형식은 동일하기에 같은 방법으로 정리해준다  
```python
import pandas as pd
df_vel = pd.read_csv('./2019년_평균속도_행정구역_읍면동단위.csv')
df_code = pd.read_csv('./읍면동 코드2.csv')
df_vel=df_vel[['emd_code','velocity_AVRG']]
df_vel['행정구역']=''
for i in range(len(df_vel)):
    for j in range(len(df_code)):
        if df_vel.loc[i,'emd_code'] == df_code.loc[j,'emd_code']:
            df_vel.loc[i,'행정구역'] = df_code.loc[j,'행정구역']
df_vel.dropna(axis=0, how='any', inplace=True)
df_vel.index=list(range(len(df_vel)))
df_vel.to_csv('./df_vel.csv')
```
다음은 역세권 공동주택 실거래 정보 데이터셋을 정리해보자  
역세권 공동주택은 거래된 가격을 추출하려 하지만 전세, 월세, 매매 세가지의 방식에 따라 가격 데이터가 남아있는 형식이라 데이터셋 안에서 가장 많이 있던 방식인 전세 방식만 사용한다.  
```python
import pandas as pd
df_trans=pd.read_csv('./2023_역세권 공동주택 실거래정보.csv')
df_trans=df_trans[['adres_nm','trns_clsf','assrnc_amt']] #각각 행정구역, 거래방식, 거래가격이다
df_trans=df_trans[df_trans['trns_clsf']=='전세']
df_trans.index=list(range(len(df_trans)))

df_trans=df_trans[['adres_nm','assrnc_amt']]
for i in range(len(df_trans)): #행정구역(읍/면/동)의 불필요한 정보는 제거해준다
    tmp=df_trans.loc[i,'adres_nm'].split()
    del tmp[-1]
    df_trans.loc[i,'adres_nm']=' '.join(tmp) 
df_trans=df_trans.groupby('adres_nm').sum()
df_trans.to_csv('./df_trans.csv')
```
이제 모든 데이터셋을 정리하였다  
각 데이터셋은 행정구역 중 읍/면/동이 가장 뒤쪽에 포함된 문자열을 데이터로 가지고 있으므로 이것을 토대로 통합한다

```python
import pandas as pd

df_ground=pd.read_csv('./df_ground.csv',index_col=0)
df_integra=pd.read_csv('./df_integra.csv',index_col=0)
df_might=pd.read_csv('./df_might3.csv',index_col=0)
df_trans=pd.read_csv('./df_trans.csv',ubdex_col=0)
df_vel=pd.read_csv('./df_vel.csv',index_col=0)
def integration(a,b,txt):
    a[txt]=''
    for i in range(len(a)):
        tmpa = a.loc[i,'행정구역'].split()
        for j in range(len(b)):
            tmpb = b.loc[j,'행정구역'].split()
            if tmpa[-1] == tmpb[-1]:
                a.loc[i,txt]=b.loc[j,txt]
    return a
test=integration(df_integra,df_ground,'count')
test=integration(df_integra,df_might,'FGCR_AADT')
test=integration(df_integra,df_trans,'assrnc_amt')
test=integration(df_integra,df_vel,'velocity_AVRG')
test.to_csv('./dataset.csv')
```
## III. Methodology
<p align = 'center'>
  <img src = 'https://github.com/js0429/AI-X-DeepLearning/blob/main/methology.png'>
</p>
&nbsp; 분류를 실행할때의 작업 흐름이다.<br>위 사진과 같이 데이터셋을 훈련과 테스트에 쓰일 두 개로 분리하고, 훈련용 데이터셋으로 모델을 학습하고 데스트용 데이터셋으로 모델을 평가하고, 정확도를 알아볼것이다 

### Algorithms




다음은 데이터셋을 분석하고 예측하는 모델을 만들기 위해 사용된 알고리즘이다. <br>

 * <b>나이브 베이즈(Naive Bayes) :</b>

&nbsp; 각 변수가 독립적이라는 가정하에 확률에 기반하여 데이터를 분류하는 방법  

<p align = 'center'>
    <img src = 'https://github.com/js0429/AI-X-DeepLearning/blob/main/nb02.png'>
</p>

> 위 식이 나이브 베이즈 에 쓰이는 베이즈 공식이다. 식을 해석해보자면<br>
>* P(h): 데이터에 관계없이 가설 h의 가능성이 참일 확률 (h의 사전 확률)
>* P(D): 가설에 관게없이 데이터의 가능성이 참일 확률 (사전 확률)
>* P(h|D): 데이터가 참일때 가설 h가 참일 확률 (h의 사후 확률)
>* P(D|h): 가설 h가 참일때 데이터가 참일 확률 (사후 확률)

>사전 확률은 특정 사건이 일어나기 전의 확률을 말하고, 사후 확률은 한 사건이 일어난 후의 확률을 말한다


&nbsp; 많은 상황에서 쓰였을 때 일반적으로 좋은 성능을 보이는 알고리즘이기에 하나의 방법으로 채택하였다.  
  

* <b>랜덤 포레스트(Random Forest) :</b>

<p align = 'center'>
  <img src = 'https://github.com/js0429/AI-X-DeepLearning/blob/main/rf02.png'>
</p> 

&nbsp; <span>의사결정 나무(Decision Tree)의 단점을 보완한 모델로 훈련으로 구성해놓은 다수의 의사결정 나무로부터 분류결과를 취합해 결론을 얻는 방법</span>

>랜덤포레스트의 설명을 위해 예를 보자면

<p align = 'center'>
  <img src = 'https://github.com/js0429/AI-X-DeepLearning/blob/main/rf01.png'>
</p>

>위의 사진은 "이번 시험에서 A를 받을 수 있는가?"를 랜덤포레스트를 통해 예측한 것인데<br>특성이 수면시간 (8시간 이상), 컷닝 할 계획 (없음), 공부 한 시간 (3시간 이하), 평균 학점 (B) 일때 11개의 의사결정 나무가 YES를 6개, NO를 5개 내보냈기에 랜덤포레스트 모델의 예측값은 YES 로 나오게 된다는 것을 설명해준다
      

&nbsp; 데이터가 잘 문서화되어있어야 효과적인 모델이므로 이번 프로젝트에 사용되기 부적합 할 수 있지만 다른 모델과의 비교를 위하여 채택하였다.  <br>
    
  
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
<p align = 'center'>
  <img src = 'https://github.com/js0429/AI-X-DeepLearning/blob/main/accuracy.png'>
</p>

>결과는 다음과 같이 나왔다


## V. Related Work
### Tools, libraries, blogs, or any documentation that we have used to do this project
* 알고리즘 선택
    * https://tensorflow.blog/머신-러닝의-모델-평가와-모델-선택-알고리즘-선택-1/

- 나이브 베이즈
  - https://www.datacamp.com/tutorial/naive-bayes-scikit-learn
  - https://ineed-coffee.github.io/posts/Naive-Bayesian/

+ 랜덤포래스트
  + https://www.datacamp.com/tutorial/random-forests-classifier-python
  + https://ineed-coffee.github.io/posts/RandomForest/
  + https://hleecaster.com/ml-random-forest-concept/

<br>

>신지성: 데이터셋 찾고 통합 및 처리, 동영상 녹화 및 업로드 <br>
    권영진: 예측 모델 코드, 블로그 작성<br> 
    종예기: 데이터셋 찾기
