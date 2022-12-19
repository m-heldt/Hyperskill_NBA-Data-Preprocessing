import pandas as pd
import numpy as np
import os
import requests
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"

# write your code here
def clean_data(datapath: str):
    df = pd.read_csv(datapath)
    df['b_day'] = pd.to_datetime(df['b_day'], format='%m/%d/%y')
    df['draft_year'] = pd.to_datetime(df['draft_year'], format='%Y')
    df['team'].fillna('No Team', inplace=True)
    df['weight'] = df['weight'].str.extract(r'/ (\d*.\d*)', expand=False).astype('float')
    df['height'] = df['height'].str.extract(r'/ (\d*.\d*)', expand=False).astype('float')
    df['salary'] = df['salary'].str.extract(r'(\d+)', expand=False).astype('float')
    df.loc[df.country != 'USA', 'country'] = 'Not-USA'
    df.loc[df.draft_round == 'Undrafted', 'draft_round'] = '0'
    return df

def feature_data(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.version == 'NBA2k20', 'version'] = '2020'
    df.loc[df.version == 'NBA2k21', 'version'] = '2021'
    df['version'] = pd.to_datetime(df['version'], format='%Y')
    df['age'] = ((df['version'] - df['b_day']) / np.timedelta64(1, "Y")).apply(np.ceil).astype('int')
    df['experience'] = ((df['version'] - df['draft_year']) / np.timedelta64(1, "Y")).round().astype('int')
    df['bmi'] = df.weight/(df.height)**2
    df.drop(columns=['weight', 'height', 'version', 'b_day', 'draft_year'], inplace=True)

    # high cardinality features
    df.drop(columns=['college', 'jersey', 'draft_peak', 'full_name'], inplace=True)

    return df

def multicol_data(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=['age'], inplace=True)
    return df

def transform_data(df: pd.DataFrame):
    numerical = df[['rating', 'experience', 'bmi']]
    scaler = StandardScaler()
    numerical = pd.DataFrame(scaler.fit_transform(numerical))
    numerical.columns = ['rating', 'experience', 'bmi']

    categorical = df.select_dtypes(exclude='number')
    encoder = OneHotEncoder()
    categorical_out = encoder.fit_transform(categorical)
    categorical = pd.DataFrame.sparse.from_spmatrix(categorical_out)

    categories = []
    for cats in encoder.categories_:
        categories.extend(list(cats))
    categorical.columns = categories

    X = pd.concat([numerical, categorical], axis=1)
    y = df['salary']

    return X, y


df_cleaned = clean_data(data_path)
df_featured = feature_data(df_cleaned)
df = multicol_data(df_featured)
X, y = transform_data(df)

answer = {
    'shape': [X.shape, y.shape],
    'features': list(X.columns),
    }
print(answer)
