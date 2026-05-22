import pickle
import os
import pandas as pd
from tqdm import tqdm

READ_FROM = "test_embeddings"

if __name__ == "__main__":
    rows = []
    for file in tqdm(os.listdir(READ_FROM)):
        with open(os.path.join(READ_FROM, file), "rb") as f:
            batch = pickle.load(f)
            rows.extend(batch)

    df = pd.DataFrame(rows)
    print(df.head())
    df.to_parquet(READ_FROM + ".parquet")
