import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class KneeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.t = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = Image.open(row["image_path"]).convert("L")  # X-ray usein 1-kanavainen
        if self.t: x = self.t(x)
        y = int(row["KL_grade"])
        return x, y

def load_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    assert {"image_path","KL_grade"}.issubset(df.columns)
    return df
