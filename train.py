#Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
import pickle

# Parameters
n_estimators=120
max_depth=20
output_file = 'model_RF.bin'

# Data preparetion
#Loading Datas
df_initial = pd.read_csv('Invistico_Airline.csv')
# Changes to column names
df_initial.columns = df_initial.columns.str.lower().str.replace(' ', '_')
# Unify categorical values
categorical_columns = list(df_initial.dtypes[df_initial.dtypes == 'object'].index)
for c in categorical_columns:
    df_initial[c] = df_initial[c].str.lower().str.replace(' ', '_')
# Fill NaN values with mean
df_initial['arrival_delay_in_minutes'].fillna(df_initial['arrival_delay_in_minutes'].mean(),inplace=True)
# Convert "departure_delay_in_minutes" to float
df_initial['departure_delay_in_minutes']=df_initial['departure_delay_in_minutes'].astype('float')
# Lets change target variable values to: 'satisfied' = 1 and 'dissatisfied' = 0
df_initial.satisfaction = (df_initial.satisfaction == 'satisfied').astype(int)
# Split
df_full_train, df_test = train_test_split(df_initial, test_size=0.2, random_state=42)

df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train.satisfaction.values
del df_full_train['satisfaction']

df_test = df_test.reset_index(drop=True)
y_test = df_test.satisfaction.values
del df_test['satisfaction']

# Training
def train(df_train, y_train, n_estimators=120, max_depth=20):
    dicts = df_train.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df.to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict(X)

    return y_pred

# training the final model
print('training the final model')

dv, model = train(df_full_train, y_full_train, n_estimators=n_estimators, max_depth=max_depth)
y_pred = predict(df_test, dv, model)

auc = np.round(roc_auc_score(y_test, y_pred),3)
print(f'auc={auc}')

# Save the model
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')