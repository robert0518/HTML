import pandas as pd
df = pd.read_csv("train_data.csv")
df['date'] = pd.to_datetime(df['date'])

df = df.sort_values(by='date')
train_df = df.iloc[:8854]
test_df = df.iloc[8854:]

train_df.to_csv('train_split.csv')
test_df.to_csv('test_split.csv')