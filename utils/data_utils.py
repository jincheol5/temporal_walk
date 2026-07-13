import os
import pandas as pd
import numpy as np
from typing import Literal

class DataUtils:
    base_path=os.path.join("..","data","temporal_graph")

    @staticmethod
    def preprocess_dataset(
            dataset_name:Literal[
                "enron",
                "wikipedia",
                "reddit"
            ],
            graph_type:Literal[
                "static",
                "temporal"
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

        dataset_path=os.path.join(DataUtils.base_path,dataset_name)
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

        ### convert to static graph
        if graph_type=="static":
            graph_df=(
                graph_df[["u","i"]]
                    .drop_duplicates()
                    .reset_index(drop=True)
            )

        return {
            "graph_df":graph_df,
            "n_node":max(graph_df["u"].max(),graph_df["i"].max()),
            "bipartite":bipartite,
            "max_u":graph_df["u"].max()
        }
