import pandas as pd


print("hi")

df1 = pd.read_csv("2018 Crime Statistics.csv")


# Open the file in write mode with UTF-8 encoding
with open("output.txt", "w", encoding="utf-8") as file:
    file.write(str(df1.columns.to_list()))


print("done")
# df = pd.read_csv("2018 Crime Statistics.csv", usecols=["OFFGUIDE", "DISTRICT", "MONRACE"])
# print(df.head())
