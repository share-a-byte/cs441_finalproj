import pandas as pd

df = pd.read_csv("trial.csv")
df = df[["id", "duration", "type"]]

new_df = pd.DataFrame(columns=["uid", "type", "offset"])

for index, row in df.iterrows():
    t_id, duration, t_type = row.values
    duration = int(duration[:-2])

    for start in range(duration // 60):
        new_df.loc[len(new_df)] = [t_id, t_type, (start * 60)]

new_df.to_csv("duration_split.csv", index=False)