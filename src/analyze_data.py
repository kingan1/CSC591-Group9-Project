import os
import pandas as pd


for f in os.listdir("../data"):
    print(f)

    df = pd.read_csv("../data/" + f)

    print(df.describe())
    


