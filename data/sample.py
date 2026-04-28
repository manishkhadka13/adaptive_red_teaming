import pandas as pd

df=pd.read_csv("data/AdvBench.csv",index_col=0)

sampled_df=df.sample(n=100,random_state=42)

sampled_df.to_csv("AdvBench_100.csv",index=False)
print("Sampled 100 rows and saved")