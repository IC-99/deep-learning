import pandas as pd
from datetime import datetime

def date_to_int(date):
    d1 = datetime.strptime('2016-04-01', "%Y-%m-%d")
    d2 = datetime.strptime(date, "%Y-%m-%d")
    return abs((d2 - d1).days)

raw_dataset = pd.read_csv('./data.csv', usecols=['user_id', 'date', 'data_type', 'data_value'])

feature_types = [10, 11, 12, 14, 15, 18, 19, 26]
feature_names = ["sleepduration", "bedin", "bedout", "awakeduration", "timetosleep", "remduration", "deepduration", "stepsgaitspeed"]

df_array = []

for i in range(len(feature_types)):
    df_feature = raw_dataset[raw_dataset['data_type'] == feature_types[i]].copy(deep=True)
    df_feature.drop(raw_dataset.columns[[2]], axis=1, inplace=True)
    df_feature.columns = ["user_id", "date", feature_names[i]]
    df_array.append(df_feature)

dataset = df_array[0]
for i in range(1, len(df_array)):
    dataset = dataset.merge(df_array[i], on=["user_id", "date"], how="outer")

dataset = dataset.dropna(subset=['sleepduration', 'stepsgaitspeed'])
dataset['date'] = dataset['date'].apply(date_to_int)
#print(dataset.head())

dataset.to_csv('dataset_new2.csv')