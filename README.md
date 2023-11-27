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

## III. Methodology
### Algorithms

다음은 데이터셋을 분석하고 예측하는 모델을 만들기 위해 사용된 알고리즘이다. <br>

 * <b>나이브 베이즈(Naive Bayes) :</b>  

    &nbsp; 각 변수가 독립적이라는 가정하에 확률에 기반하여 데이터를 분류하는 방법  

    &nbsp; 많은 상황에서 쓰였을 때 일반적으로 좋은 성능을 보이는 알고리즘이기에 하나의 방법으로 채택하였다.  
  
 * <b>단순 선형 회귀(Linear Regression) :</b>  

    &nbsp; 데이터를 정의하는 가장 최적의 선을 찾아 분류하는 방법   

    &nbsp; 데이터셋을 고려했을때 특성(Feature)에 대한 목표값(Target)의 변화를 예측하는것에 적절히 사용될 수 있다고 생각해 채택하였다.  

* <b>랜덤 포레스트(Random Forest) :</b>    
      
    &nbsp; <span>의사결정 나무(Decision Tree)의 단점을 보완한 모델로 훈련으로 구성해놓은 다수의 의사결정 나무로부터 분류결과를 취합해 결론을 얻는 방법</span>

    &nbsp; 데이터가 잘 문서화되어있어야 효과적인 모델이므로 이번 프로젝트에 사용되기 부적합 할 수 있지만 다른 모델들과의 비교를 위하여 채택하였다.  <br>
    
  
### Features

  <b>목표값(Target Feature)</b> : 시간별 교통량에 따른 교통체증의 정도(1 ~ 0 로 표현)

<b>특성(Features)</b>   
  
  * 미개발 토지 : (값에대한 설명)

  * 추정 교통량 : (값에대한 설명)

  * 역세권 실거래 정보 : (값에대한 설명)

  * 평균 속도 : (값에대한 설명)

  * 속하는 시/군/구 의 상권 활성도 : (값에대한 설명)

## IV. Evaluation & Analysis
### Graphs, tables, any statistics

## V. Related Work
### Tools, libraries, blogs, or any documentation that we have used to do this project

## VI. Conclusion: Discussion
### Feel free to use any format we like
