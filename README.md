 
#로지스틱 회귀 분석은 Yes/No 처럼 두가지로 나뉘는 분류 문제를 다룬다.
#타이타닉 승객정보 데이터셋을 이용해 생존 여부 예측하기

 
#라이브러리 및 데이터 불러오기
import pandas as pd
file_url="https://media.githubusercontent.com/media/musthave-ML10/data_source/main/titanic.csv"
data=pd.read_csv(file_url)

 
data.head()
#	Pclass 티켓 클래스 
#	SibSp 함께 탑승한 형제 및 배우자 수
#	Parch 함께 탑승한 부모 및 자녀 수
#   Embarked 승선한 항구
#   Survived 생존 여부

 
data.info()

 
data.describe()
#pclass에서 min 부터 max 까지의 값을 보면 1/2/3 총 3가지 값이 있습니다.
#Age는 대부분 승객이 비교적 젊은 층이지만 일부 나이가 많은 승객이 있다.
#SibSp, Parch 를 보면 대부분 승객이 가족을 동반하지 않고 혼자 탑승했다.

 
data.corr()
#데이터 간의 상관 관계 분석
#Parch 와 SibSp 의 상관관계가 높게 나타났다. 가족을 동반할 경우 부모와 자녀, 형제 배우자를 함께 동반할 가능성이 높기 대문이다.

# 0.2 이하 상관관계가 거의 없다.
# 0.2~0.4  : 낮은 상관 관계
# 0.4~0.6 : 중간수준의 상관관계
# 0.6~0.8 : 높은 상관관계
# 0.8 이상 : 매우높은 상관관계


 
import matplotlib.pyplot as plt
import seaborn as sns

 
sns.heatmap(data.corr())
plt.show()

 
sns.heatmap(data.corr(),cmap='coolwarm')
plt.show()

 
sns.heatmap(data.corr(),cmap='coolwarm',vmin=-1,vmax=1)
plt.show()

 
sns.heatmap(data.corr(),cmap='coolwarm',vmin=-1,vmax=1,annot=True)
plt.show()

 
data["Name"].nunique()

 
data["Sex"].nunique()

 
data["Ticket"].nunique()

 
data["Embarked"].nunique()

 
data=data.drop(['Name','Ticket'],axis=1)

 
data.head()

 
pd.get_dummies(data,columns=['Sex','Embarked'])

 
pd.get_dummies(data,columns=['Sex','Embarked'],drop_first=True)

 
data=pd.get_dummies(data,columns=['Sex','Embarked'],drop_first=True)

 
from sklearn.model_selection import train_test_split
X=data.drop('Survived',axis=1) #데이터셋에서 종속변수 제거후 저장
y=data['Survived'] 
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=100) #학습셋과 시험셋 분리

 
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀 import

 
model=LogisticRegression() #로지스틱 회귀 모델 생성
model.fit(X_train, y_train) #모델 학습

 
pred=model.predict(X_test) #예측

 
#예측 모델 평가하기
from sklearn.metrics import accuracy_score #정확도 라이브러리 임포트
accuracy_score(y_test,pred) #실제값과 예측값으로 정확도 계산

# 78%이면 그렇게 나쁜 수준은 아니지만 엄청 잘 예측하는 모델이라고 할 수는 없습니다.


 
model.coef_

 
pd.Series(model.coef_[0],index=X.columns)

 
#PClass는 음의 계수를 가지고 있기 때문에 Pclass가 높을 수로 생존가능성이 난다. 3등급 class는 생존 가능성이 낮았다.
#성별은 여성이면 생존 가능성이 높았다. 라고 해석할 수 있다. 

 
#이해하기 : 피처 엔지니어링
#피처엔지니어링이란 기존 데이터를 손보아 더 나은 변수를 만드는 기법입니다.


data['family']=data['SibSp']+data['Parch'] #SibSp와 Parch 변수 합하기
data.drop(['SibSp','Parch'],axis=1,inplace=True)
data.head()



X=data.drop('Survived',axis=1) #데이터셋에서 종속변수 제거후 저장
y=data['Survived'] 
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=100) #학습셋과 시험셋 분리
model=LogisticRegression() #로지스틱 회귀 모델 생성
model.fit(X_train, y_train) #모델 학습
pred=model.predict(X_test) #예측
accuracy_score(y_test,pred) #실제값과 예측값으로 정확도 계산

# 피쳐엔지니어링으로 변수를 조정하여 조금 더 정확도를 높혔다


pred


X_test


import numpy as np
#신규 데이터로 예측 해보기


jack=np.array([3,20.0,0,0,0,0])
Rose=np.array([1,19.0,1,0,1,1])
ME=np.array([2,48.0,1,1,1,5])


sample_pred=np.array([jack,Rose,ME])
import pandas as pd

sample_df = pd.DataFrame(sample_pred, columns=['Pclass', 'Age', 'Sex_male', 'Embarked_Q', 'Embarked_S', 'family'])
pred_result=model.predict(sample_df)


pred_result
#나만 죽는 걸로





