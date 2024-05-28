import pandas as pd

raw_dataset = pd.read_csv('./sample.csv', usecols=['user_id', 'date', 'data_type', 'data_value'])

feature_types = [10, 11, 12, 14, 15, 18, 19, 26, 27]
feature_names = ["sleepduration", "bedin","bedout", "awakeduration", "timetosleep", "remduration", "deepduration", "stepsgaitspeed", "distancegaitspeed"]

df_array = []

for i in range(len(feature_types)):
    df_feature = raw_dataset[raw_dataset['data_type'] == feature_types[i]].copy(deep=True)
    df_feature.drop(raw_dataset.columns[[2]], axis=1, inplace=True)
    df_feature.columns=["user_id","date", feature_names[i]]
    df_array.append(df_feature)

dataset = df_array[0]
for i in range(1, len(df_array)):
    dataset = dataset.merge(df_array[i], on=["user_id", "date"], how="outer")

dataset.set_index(['user_id','date'], inplace=True)
print(dataset.head())

#dataset = dataset.dropna()
#print(dataset.head())

dataset.to_csv('dataset_new.csv')