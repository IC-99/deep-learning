import pandas as pd
from numpy import random
from random import randrange

data = {'user_id': [], 'date': [], 'sleepduration': [], 'bedin': [], 'bedout': [], 'awakeduration': [],
        'timetosleep': [], 'remduration': [], 'deepduration': [], 'stepsgaitspeed': [], 'distancegaitspeed': []}

def get_random_float(loc, scale):
    return round(abs(random.normal(loc, scale)), 2)

for user_id in range(1000):
    for date in range(365):
        data['user_id'].append(user_id)
        data['date'].append(date)

        bedin = get_random_float(24, 1)
        bedout = get_random_float(9, 1)
        timetosleep = get_random_float(0.5, 0.1)
        awakeduration = get_random_float(0.5, 0.1)
        sleepduration = round(bedout + 24 - bedin - timetosleep - awakeduration, 2)
        rem_percentage = randrange(15, 30) / 100
        deep_percentage = randrange(5, 15) / 100

        data['sleepduration'].append(sleepduration)
        data['bedin'].append(bedin)
        data['bedout'].append(bedout)
        data['awakeduration'].append(awakeduration)
        data['timetosleep'].append(timetosleep)
        data['remduration'].append(round(sleepduration * rem_percentage, 2))
        data['deepduration'].append(round(sleepduration * deep_percentage, 2))
        data['stepsgaitspeed'].append(get_random_float(95, 4))
        data['distancegaitspeed'].append(get_random_float(4.5, 0.2))

for col in data:
    print(len(data[col]))

dataset = pd.DataFrame(data)

dataset.to_csv('dataset_fake.csv')