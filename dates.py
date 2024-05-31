from datetime import datetime

def date_to_int(date):
    d1 = datetime.strptime('2016-04-01', "%Y-%m-%d")
    d2 = datetime.strptime(date, "%Y-%m-%d")
    return abs((d2 - d1).days)

print(date_to_int('2016-04-01'))