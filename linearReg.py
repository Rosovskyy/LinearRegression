%matplotlib inline

import re
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


data = pd.read_csv('train.csv')

regions = list(set(data["Район"].tolist()))
for col, row in data.iterrows():
    if regions.index(row["Район"]) == 0:
        continue
    data.at[col, "Район"] = regions.index(row["Район"])

data_prepared = data.copy(True)

data_prepared.loc[data_prepared['Тип ціни'] == 'грн', 'Ціна'] /= 28.1
data_prepared.loc[data_prepared['Тип ціни'] == '€', 'Ціна'] /= 0.88

for col in ['Загальна площа', 'Житлова площа', 'Кімнат']:
    data_prepared[col] = data_prepared[col].fillna(0)
    data_prepared[col] = pd.to_numeric(data_prepared[col].apply(lambda x: re.sub(',', '.', str(x))))
    data_prepared.loc[data[col] == 0, col] = data_prepared.loc[data[col] != 0, col].mean(skipna=True)

x1 = data_prepared[['Загальна площа', 'Житлова площа', 'Кімнат']]

y = data_prepared['Ціна']

lr = LinearRegression()
lr.fit(x1, y)

test = pd.read_csv('test.csv')

regions = list(set(test["Район"].tolist()))
for col, row in test.iterrows():
    if regions.index(row["Район"]) == 0:
        continue
    test.at[col, "Район"] = regions.index(row["Район"])

for col in ['Загальна площа', 'Житлова площа', 'Кімнат']:
    test[col] = test[col].fillna(0)
    test[col] = pd.to_numeric(test[col].apply(lambda x: re.sub(',', '.', str(x))))
    test.loc[test[col] == 0, col] = test.loc[test[col] != 0, col].mean(skipna=True)

x_test = test[['Загальна площа', 'Житлова площа', 'Кімнат']]

result = pd.DataFrame(columns=['Ціна'])
result['Ціна'] = lr.predict(x_test)

result['Тип ціни'] = test['Тип ціни']

result.loc[result['Тип ціни'] == 'грн', 'Ціна'] *= 28.1
result.loc[result['Тип ціни'] == '€', 'Ціна'] *= 0.88

result = result['Ціна']
result.to_csv('submission.csv', sep=',')
