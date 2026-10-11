import pandas as pd

cols = ["USSCIDN", "OFFGUIDE", "DISTRICT", "MONRACE", "HISPORIG", "NEWRACE", "MONSEX"]
ussc = pd.read_csv("2018 Crime Statistics.csv", usecols=cols, low_memory=False)
print(ussc.shape)   # should be roughly 69k rows
print(ussc.isna().mean())

ussc.to_csv("ussc2018_slim.csv", index=False)