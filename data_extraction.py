import pandas as pd
import numpy as np
import pickle

# apply the maximum absolute scaling in Pandas using the .abs() and .max() methods
def maximum_absolute_scaling(df):
    # copy the dataframe
    df_scaled = df.copy()

    # apply maximum absolute scaling
    for column in df_scaled.columns:
        df_scaled[column] = df_scaled[column]  / df_scaled[column].abs().max()

    return df_scaled

data_csv = pd.read_csv('heart.csv')
data_csv = maximum_absolute_scaling(data_csv)

#att = ['age', 'sex', 'fbs', 'thalach', 'exang', 'chol', 'trestbps']

att = ['age', 'sex', 'fbs', 'thalach', 'exang', 'chol', 'trestbps']
target = ['target']

X = []
y = []

for i in range(data_csv.shape[0]):
    X.append(list(data_csv[att].iloc[i]))
    y.append(list(data_csv[target].iloc[i]))
    print(f'[EXTRACT] Row {i} finished')

X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)

with open('X.pkl', 'wb') as f:
    print('WRITING X')
    pickle.dump(X, f)

with open('y.pkl', 'wb') as f:
    print('WRITING y')
    pickle.dump(y, f)

print('Done.')