import pandas as pd
from typing import Literal

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