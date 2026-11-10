import numpy as np
import pandas as pd


def load_clean(path="../data/raw/listings.csv.gz", price_min=20, price_max=2000):
    df = pd.read_csv(path)
    df["price"] = (df["price"]
                   .str.replace("$", "", regex=False)
                   .str.replace(",", "", regex=False)
                   .astype(float))
    df = df[df["price"].between(price_min, price_max)].copy()
    df["log_price"] = np.log1p(df["price"])
    return df.reset_index(drop=True)