import pandas as pd
from numpy import random

data = {'user_id': [], 'date': [], 'sleepduration': [], 'bedin': [], 'bedout': [], 'awakeduration': [],
        'timetosleep': [], 'remduration': [], 'deepduration': [], 'stepsgaitspeed': [], 'distancegaitspeed': []}

def get_random_float(loc, scale):
    return round(abs(random.normal(loc, scale)), 2)

for user_id in range(1000):
    for date in range(365):
        data['user_id'].append(user_id)
        data['date'].append(date)

        data['sleepduration'].append(get_random_float(7, 4))
        data['bedin'].append(get_random_float(24, 4))
        data['bedout'].append(get_random_float(9, 3))
        data['awakeduration'].append(get_random_float(0.5, 0.25))
        data['timetosleep'].append(get_random_float(0.5, 0.25))
        data['remduration'].append(get_random_float(2, 1))
        data['deepduration'].append(get_random_float(3, 2))
        data['stepsgaitspeed'].append(get_random_float(95, 15))
        data['distancegaitspeed'].append(get_random_float(4.5, 1))

for col in data:
    print(len(data[col]))

dataset = pd.DataFrame(data)

dataset.to_csv('dataset_fake.csv')