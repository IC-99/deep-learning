import pandas as pd
from datetime import datetime

def date_to_int(date):
    d1 = datetime.strptime('2016-04-01', "%Y-%m-%d")
    d2 = datetime.strptime(date, "%Y-%m-%d")
    return abs((d2 - d1).days)

dataset = pd.read_csv('./dataset.csv')

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

"""
abbiamo 799164 sequenze di lunghezza 1
abbiamo 686673 sequenze di lunghezza 2
abbiamo 610522 sequenze di lunghezza 3
abbiamo 552533 sequenze di lunghezza 4
abbiamo 506354 sequenze di lunghezza 5
abbiamo 468830 sequenze di lunghezza 6
abbiamo 437578 sequenze di lunghezza 7
abbiamo 410421 sequenze di lunghezza 8
abbiamo 386383 sequenze di lunghezza 9
abbiamo 364903 sequenze di lunghezza 10
abbiamo 345721 sequenze di lunghezza 11
abbiamo 328506 sequenze di lunghezza 12
abbiamo 312951 sequenze di lunghezza 13
abbiamo 298779 sequenze di lunghezza 14
abbiamo 285678 sequenze di lunghezza 15
abbiamo 273477 sequenze di lunghezza 16
abbiamo 262115 sequenze di lunghezza 17
abbiamo 251538 sequenze di lunghezza 18
abbiamo 241740 sequenze di lunghezza 19
abbiamo 232623 sequenze di lunghezza 20
abbiamo 224057 sequenze di lunghezza 21
abbiamo 215943 sequenze di lunghezza 22
abbiamo 208261 sequenze di lunghezza 23
abbiamo 201011 sequenze di lunghezza 24
abbiamo 194184 sequenze di lunghezza 25
abbiamo 187739 sequenze di lunghezza 26
abbiamo 181610 sequenze di lunghezza 27
abbiamo 175807 sequenze di lunghezza 28
abbiamo 170287 sequenze di lunghezza 29
"""