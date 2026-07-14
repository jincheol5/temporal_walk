import os
import pandas as pd
import numpy as np
from typing import Literal

class DataUtils:
    base_path=os.path.join("..","data")

    @staticmethod
    def preprocess_static_graph(
            dataset_name:Literal[
                "email-Enron"
            ]="email-Enron"
        ):
        dataset_path=os.path.join(DataUtils.base_path,"static_graph",dataset_name)
        graph_path=os.path.join(dataset_path,f"{dataset_name}.txt")
        graph_df=pd.read_csv(
            graph_path,
            sep="\t", # 열 구분
            comment="#", # #으로 시작하는 행 무시
            header=None, # 실제 데이터에 열 이름이 없다고 지정
            names=["u","i"], # 두 열의 이름 지정
        )

        ### remove self-loop
        graph_df=(
            graph_df[
                graph_df["u"]!=graph_df["i"]
            ]
            .copy()
            .reset_index(drop=True)
        )

        ### remapping node id
        used_node_ids=np.sort(
            pd.concat([graph_df["u"],graph_df["i"]])
            .unique()
            .astype(int)
        )
        node_id_map={
            old_id:new_id
            for new_id,old_id in enumerate(used_node_ids,start=1)
        }
        graph_df["u"]=graph_df["u"].map(node_id_map).astype(int)
        graph_df["i"]=graph_df["i"].map(node_id_map).astype(int)

        return {
            "graph_df":graph_df,
            "n_node":max(graph_df["u"].max(),graph_df["i"].max()),
            "bipartite":False,
            "max_u":graph_df["u"].max()
        }

    @staticmethod
    def preprocess_temporal_graph(
            dataset_name:Literal[
                "enron",
                "wikipedia",
                "reddit"
            ]
        ):
        """
        Temporal graph dataset
            - ml_dataset.csv
                - col: u,i,ts,label,idx
                - ts 기준 오름차순 정렬된 상태
                - self-loop 제거 필요
        """
        bipartite_dataset={
            "enron":False,
            "wikipedia":True,
            "reddit":True
        }
        bipartite=bipartite_dataset[dataset_name]

        dataset_path=os.path.join(DataUtils.base_path,"temporal_graph",dataset_name)
        graph_path=os.path.join(dataset_path,f"ml_{dataset_name}.csv")
        graph_df=pd.read_csv(graph_path,index_col=0)
        graph_df=graph_df[
            ["u","i","ts"]
        ].copy()

        ### remove self-loop
        graph_df=(
            graph_df[
                graph_df["u"]!=graph_df["i"]
            ]
            .copy()
            .reset_index(drop=True)
        )

        ### remapping node id
        used_node_ids=np.sort(
            pd.concat([graph_df["u"],graph_df["i"]])
            .unique()
            .astype(int)
        )
        node_id_map={
            old_id:new_id
            for new_id,old_id in enumerate(used_node_ids,start=1)
        }
        graph_df["u"]=graph_df["u"].map(node_id_map).astype(int)
        graph_df["i"]=graph_df["i"].map(node_id_map).astype(int)

        return {
            "graph_df":graph_df,
            "n_node":max(graph_df["u"].max(),graph_df["i"].max()),
            "bipartite":bipartite,
            "max_u":graph_df["u"].max()
        }
