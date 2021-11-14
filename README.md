# 2018년 빅데이터 콘테스트 주제로 모델 구현
## NCsoft 리니지 유저이탈여부 예측 모델

### 데이터 전처리

먼저 기본적인 데이터프레임을 형성한 후 제일먼저 활동내역을 기록한 train_activity.csv 파일을 불러온다. 그리고 시작하기에 앞서 이 데이터는 4만개의 계정(즉 4만명의 유저)의 데이터를 다루기때문에 모든 파일에서 4만개의 아이디에 대해 그룹화를 진행할 것이다.

~~~
def make_basic_form(activity):

    df = activity.groupby(['acc_id']).sum()
  #  features = df.columns
  #  df = df.drop(features, axis = 1)
    return df

activity = pd.read_csv('train_activity.csv')
df = make_basic_form(activity)

df.to_csv('train_xdata.csv')
~~~

train_activity.csv 파일에는 각 유저가 플레이한 날도 기록이 되어있다. 그렇기에 밑에 함수는 먼저 어떤유저가 어떤날에 플레이 했는지를 그룹화하고 그 플레이 한 날의 간격 그중에서도 가장 긴 간격의 날을 추출해 내는 함수이다. 매일 했다면 플레이 간격 날짜는 1이 추출이 되는것이고 마지막에는 28에서 간격을 빼게 되는데 28에서 빼는 이유는 일단 이 데이터 자체가 28일을 기준으로 데이터가 만들어 졌으며 자주 플레이 할수록 생존했다는 레이블에 더욱 긍정적인 요소를 미치기에 가중치를 더 크게 하기위해서 뺀것이다.

~~~
def add_act_day_diff_feature(df, activity):

    df['act_day_diff'] = 0
    activity = activity.groupby(['acc_id', 'day']).count()
    activity = activity.reset_index()

    for i in df.index:
        days = list(activity[activity['acc_id'] ==i]['day'])
        days.append(28)
        days.insert(0,0)
        if len(days) ==1:
            df.loc[i, 'act_day_diff'] = 0
        else:
            max_diff = 0
            temp = days[0]
            for j in range(len(days)-1):
                if days[j+1] - temp > max_diff:
                    max_diff=days[j+1] - temp
                temp = days[j+1]
            df.loc[i,'act_day_diff']=28 - max_diff
    return df


activity = pd.read_csv('train_activity.csv')
df = pd.read_csv('train_xdata.csv').set_index('acc_id', drop=True)
df = add_act_day_diff_feature(df, activity)

df.to_csv('train_xdata.csv')
~~~

각 csv 파일마다 쓸모없다고 여겨지는 피쳐들을 제거해준다. 그리고 데이터프레임에 합쳐준다.

![1](https://user-images.githubusercontent.com/65812122/141679609-abccd46a-9601-4585-b77b-e6a76cf9b1dc.JPG)

전체적으로 전처리를 한 데이터를 해석해보자면 계정ID가 2인 사람은 활동이 다른사람들보다 훨씬 적으며 혈맹간의 전투 활동조차도 존재 하지 않는다 하지만 플레이시간이 전체적으로 다른사람보다 월등히 높고 계정간의 아이템 판매가격이 높은편에 속한다. 후에 이계정의 생존유무를 보면 계정ID가 2인 사람은 잔존한다.반면에 계정ID가 8인 사람을 보자면 2인 사람과 비교했을때 플레이타임과 거래가격총량은 월등히 낮은반면 활동이력이 월등히 높다. 후에 8의 생존유무를 보았을때에도 이 계정도 잔존한다. 결론적으로 나의경험으로 비추어 보아 2인사람은 아마도 장사를 주로 하는 성향의 사람일 확률이 높다. 또한 8인 사람은 짧은 플레이타임에 월등히 높은 활동과 혈맹간의 활동이력으로 보아 직장인에 하드코어하게 게임을 즐기는 성향의 사람일 확률이 높다. 이런 극단적인 활동을 하는 사람들이 현실에서도 장기적 유저일 확률이 높으며 나의 생각에는 어중간한 포지션에 속해있는 유저들(예를들어, 아이템을 과금할 능력이 부족해 적당히 하고 컨텐츠만 즐기는 라이트 유저)이 이탈할 확률이 높고,아예 플레이타임도 적고 활동자체가 적은 유저도 이탈할 확률이 높기에 이 모델에서도 피쳐들이 평균적으로 중간 값에 포진해있는 계정ID와 모든 피쳐값이 낮은 계정ID가 이탈할것이라고 예측할것같다.


### 모델 만들기와 학습

~~~
def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(file_path,
                                                    batch_size=32,
                                                    label_name=LABEL_COLUMN,
                                                    na_value="?",
                                                    num_epochs=1,
                                                    ignore_errors=True,
                                                    **kwargs)
    return dataset
    
class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_freatures = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_freatures]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features
        
        matches =tf.equal('Y', labels)
        onehot = tf.cast(matches, tf.float32)
        labels = onehot
       
        
       
        
        return features, labels
        
desc=feature[NUMERIC_FEATURES].describe()
MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])
~~~

전처리한 데이터에서 feature값들을 불러오고 해당 데이터셋에 대한 모델을 학습시킬 때, 필요한 파라미터 값을 넘겨준다. 또한 인코딩 방식도 적어주며, 평균과 표준편차 또한 계산한다.


~~~
model = tf.keras.Sequential([
    numeric_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
    
model.fit(train_data, epochs=500)
~~~

relu 함수와 sigmoid 함수를 이용하여 4개의 hidden layer를 추가 했다.
loss 함수는 binary_crossentropy, optimizer는 adam optimizer를 사용하였으며,
epoch는 500으로 설정하고 훈련을 진행.ㅣ
![2](https://user-images.githubusercontent.com/65812122/141679801-b7741768-1cb5-4396-b504-9e5a533244f2.JPG)


~~~
for prediction, delay in zip(predictions[:32], mydata[0][1]):  
    print("Predicted ALIVE: {:.5%}".format(prediction[0]),
        " | Actual ALIVE: ",
        ("Y" if bool(delay) else "N"))
~~~

![3](https://user-images.githubusercontent.com/65812122/141679857-0bfab7db-6c3c-4475-84d1-6086fe2eac3e.JPG)

전체적으로 몇개의 ID를 제외하곤 준수한 정확도를 보여준다. 여기서 정확도를 더욱 높이려면 나의 생각에는 12 일 23 일...27~28일의 일 간격들에 해당하는 활동이력들을 취합하여 매핑 하게 되면 더욱 정확한 모델이 나오지 않을까 예측해본다. 그리고 사실 게임에서는 예외 케이스가 많이 있다. 예를들어 한사람이 2개의 명의로 2개의 계정을 이용하여 하나는 플레이용 하나는 장사용 계정으로 이용할수가 있다. 이런것을 파악하기 위해서는 train_trade.csv 파일에 각 어느 계정끼리 거래를 했는지까지도 자세히 살펴봐야 할것이다. 왜냐하면 위의 예로 따져보았을때 장사용 계정은 분명히 플레이용 계정과 거래가 많았고 아이템 거래량또한 많았을것이기 때문이다. (장사용 계정에서 플레이용 계정에 아이템을 판 돈을 주면서,플레이용 계정에서는 그동안 새롭게 얻은아이템을 옮기는 형식의 거래. 그렇기에 실제 거래수에 비해 제대로된 게임화폐 거래량이 불규칙할 것) 이런 예외적인 케이스들을 다 따져보아 쓸데없는 계정의 데이터도 제거해 나가다보면 더욱 정확한 예측이 가능할 것이다.

또한 이 예측모델의 목적은 이탈할 유저를 정확히 예측하여 이탈할 유저에 맞춰 프로모션을 진행하여 이탈을 막고자 함이다. 그렇기에 이탈할 유저의 예측 정확도가 더욱 중요하다.(N인 경우)
