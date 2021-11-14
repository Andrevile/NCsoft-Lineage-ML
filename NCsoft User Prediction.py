import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import models,layers
import pandas as pd
import csv
import os

def make_basic_form(activity):

    df = activity.groupby(['acc_id']).sum()
  #  features = df.columns
  #  df = df.drop(features, axis = 1)
    return df

activity = pd.read_csv('train_activity.csv')
df = make_basic_form(activity)

df.to_csv('train_xdata.csv')

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

columns=['day',"char_id","private_shop","death","game_money_change","enchant_count","exp_recovery"]
df = df.drop(columns,axis = 1)

combat=pd.read_csv("train_combat.csv")
gain_features = ['pledge_cnt', 'num_opponent']
combat = combat.groupby('acc_id').sum()[gain_features]

df=df.join(combat)

payment=pd.read_csv("train_payment.csv")
payment=payment.groupby(['acc_id']).sum()
del payment["day"]

df=df.join(payment)

pledge=pd.read_csv("train_pledge.csv")
pledge=pledge.groupby(["acc_id"]).sum()
columns=["day","char_id","pledge_id"]
pledge= pledge.drop(columns,axis = 1)

df=df.join(pledge)

trade=pd.read_csv("train_trade.csv")
trade=trade.groupby(["source_acc_id"]).sum()
columns=["day","type","source_char_id","target_acc_id","target_char_id","item_amount"]
trade=trade.drop(columns,axis=1)

df=df.join(trade)

df=df.fillna(0)

label=pd.read_csv("train_label.csv")
label=label.groupby(["acc_id"]).sum()
del label["amount_spent"]

df=df.join(label)
df.loc[df.survival_time<64,"survival_time"]='N'
df.loc[df.survival_time==64,"survival_time"]='Y'

df.to_csv("feature.csv",header=True,index=False)
df=pd.read_csv("feature.csv")


def load_feature_data(path):
    csv_path=os.path.join(path,"feature.csv")
    return pd.read_csv(csv_path)

feature=load_feature_data(".")

LABEL_COLUMN="survival_time"
LABELS=[0,1]

def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(file_path,
                                                    batch_size=32,
                                                    label_name=LABEL_COLUMN,
                                                    na_value="?",
                                                    num_epochs=1,
                                                    ignore_errors=True,
                                                    **kwargs)
    return dataset


raw_train_data=get_dataset("feature.csv")

def show_batch(dataset):
    for batch,label in dataset.take(1):
        for key,value in batch.items():
            print("{:20s}:{}".format(key,value.numpy()))
        print("label: ",label.numpy())

what=pd.read_csv("feature.csv")

conti_var=feature.columns[feature.dtypes!='object']
list(conti_var)


class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_freatures = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_freatures]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        matches = tf.equal('Y', labels)
        onehot = tf.cast(matches, tf.float32)
        labels = onehot

        return features, labels

NUMERIC_FEATURES=list(conti_var)
packed_train_data=raw_train_data.map(PackNumericFeatures(NUMERIC_FEATURES))


show_batch(packed_train_data)


example_batch,labels_batch=next(iter(packed_train_data))

desc=feature[NUMERIC_FEATURES].describe()
MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])

def normalize_numeric_data(data, mean, std):
    return (data-mean)/std

import functools
normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=None, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]



numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
numeric_layer(example_batch).numpy()

train_data = packed_train_data.take(900).shuffle(100)
test_data = packed_train_data.skip(900)

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

test_loss, test_accuracy = model.evaluate(test_data)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

mydata=list(test_data)
predictions = model.predict(mydata[0][0])


for prediction, delay in zip(predictions[:32], mydata[0][1]):
    print("Predicted ALIVE: {:.5%}".format(prediction[0]),
        " | Actual ALIVE: ",
        ("Y" if bool(delay) else "N"))