import pandas as pd
from datetime import datetime

def date_to_int(date):
    d1 = datetime.strptime('2016-04-01', "%Y-%m-%d")
    d2 = datetime.strptime(date, "%Y-%m-%d")
    return abs((d2 - d1).days)

dataset = pd.read_csv('./dataset.csv', nrows=10000)

dataset['date'] = dataset['date'].apply(date_to_int) # trasforma le date in interi

#selection = df.loc[df['data_type'] == 18]

print(dataset.head())

distributions = {}

last = None
count = 1

for index, row in dataset.iterrows():
    date = row['date']
    if last != None:
        if date == last + 1 or date == last + 2:
            count += 1
        else:
            distributions[count] = distributions.get(count, 0) + 1
            count = 1
    
    last = date

print(distributions)

def num_of_seq_of_len(target_len):
    tot_seq = 0
    for seq_len in range(target_len, 367):
        if seq_len in distributions:
            tot_seq += distributions[seq_len] * (seq_len - target_len + 1)
    return tot_seq

for seq_len in range(1, 30):
    print('abbiamo', num_of_seq_of_len(seq_len), 'sequenze di lunghezza', seq_len)