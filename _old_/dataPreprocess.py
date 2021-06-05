import pandas as pd
import numpy as np

df = pd.read_csv("data/dataSetNew/featuresDataMerged.csv", sep=',',
                 names=["EAR", "MAR", "Circularity", "MOE", "LIP_DIS", "Drowsy"])
print(df.shape)

cols = ["EAR", "MAR", "Circularity", "MOE", "LIP_DIS"]
df[cols] = df[cols].replace({0: np.nan})
df = df.dropna()

df_new = df.drop(df[(df['EAR'] <= 0.22) & (df['Drowsy'] == 0.0) & (df['LIP_DIS'] > 17)].index)
df_new = df_new.drop(df_new[(df_new['EAR'] > 0.22) & (df_new['Drowsy'] == 1.0)].index)

dfAlert = df_new.loc[df_new['Drowsy'] == 0.0]
dfDrowsy = df_new.loc[df_new['Drowsy'] == 1.0]

print(dfAlert.shape)
print(dfDrowsy.shape)

df.to_csv("data/dataSetNew/featuresDataFiltered.csv", index = False, header=False)
