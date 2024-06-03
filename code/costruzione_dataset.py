import pandas as pd

raw_dataset = pd.read_csv('./data.csv', usecols=['user_id', 'date', 'data_type', 'data_value'])

feature_types = [10, 11, 12, 14, 15, 18, 19, 26]
feature_names = ["sleepduration", "bedin", "bedout", "awakeduration", "timetosleep", "remduration", "deepduration", "stepsgaitspeed"]

dataset_array = []

for i in range(len(feature_types)):
    dataset_feature = raw_dataset[raw_dataset['data_type'] == feature_types[i]].copy(deep=True)
    dataset_feature.drop(raw_dataset.columns[[2]], axis=1, inplace=True)
    dataset_feature.columns = ["user_id", "date", feature_names[i]]
    dataset_array.append(dataset_feature)

dataset = dataset_array[0]
for i in range(1, len(dataset_array)):
    dataset = dataset.merge(dataset_array[i], on=["user_id", "date"], how="outer")

dataset = dataset.dropna(subset=['sleepduration', 'stepsgaitspeed'])

dataset.set_index(['user_id','date'], inplace=True)

dataset.to_csv('dataset.csv')