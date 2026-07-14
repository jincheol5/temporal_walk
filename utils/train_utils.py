import pandas as pd
import torch
from typing import Literal
from torch.utils.data import Dataset

class StaticGraphDataset(Dataset):
    def __init__(self,df:pd.DataFrame):
        self.src=torch.tensor(df["u"].values,dtype=torch.long)
        self.dst=torch.tensor(df["i"].values,dtype=torch.long)

    def __len__(self):
        return len(self.src)

    def __getitem__(self,idx):
        return self.src[idx],self.dst[idx]


class TrainUtils:
    @staticmethod
    def split_graph_df(
            df:pd.DataFrame,
            train_ratio:float=0.7,
            val_ratio:float=0.1,
            is_temporal:bool=True,
            random_state:int=42
        ):
        """
        Input:
            df
            train_ratio
            val_ratio
            is_temporal
            random_state
        Return:
            train_df
            val_df
            test_df
        """
        if not is_temporal:
            # 전체 행을 무작위로 섞고 인덱스 초기화
            df=df.sample(
                frac=1,
                random_state=random_state,
            ).reset_index(drop=True)
        n=len(df)
        train_end=int(n*train_ratio)
        val_end=int(n*(train_ratio+val_ratio))
        train_df=df.iloc[:train_end].reset_index(drop=True)
        val_df=df.iloc[train_end:val_end].reset_index(drop=True)
        test_df=df.iloc[val_end:].reset_index(drop=True)
        return train_df,val_df,test_df
    
    @staticmethod
    def get_element_of_static_graph(
            df:pd.DataFrame
        ):
        """
        """
        src=torch.as_tensor(
            df["u"].to_numpy(),
            dtype=torch.long,
        )
        dst=torch.as_tensor(
            df["i"].to_numpy(),
            dtype=torch.long,
        )
        return {
            "src":src,
            "dst":dst
        }
    
    @staticmethod
    def split_graph(
            src:torch.Tensor,
            dst:torch.Tensor,
            train_ratio:float=0.7,
            val_ratio:float=0.1,
        ):
        """
        """