import random
import pandas as pd
import numpy as np
import torch
from collections import defaultdict

class Graph:
    """
    node_id=0: padding node
    """
    def __init__(self,
            graph_df:pd.DataFrame,
            embed_dim:int
        ):
        """
        Input:
            graph_df: pd.DataFrame
        """
        # set adj
        self.adj=defaultdict(list)
        for event in graph_df.itertuples(index=False): # col: [u,i,label]
            src=int(event.u)
            dst=int(event.i)

            # edge 양방향 저장
            self.adj[dst].append(src)
            self.adj[src].append(dst)

        # set n_node
        self.n_node=max(graph_df["u"].max(),graph_df["i"].max())

        # set node_ft
        self.set_node_ft(embed_dim=embed_dim)

    def set_node_ft(self,
            embed_dim:int=32
        ):
        """
        Input:
            embedding_dim: int
        """
        self.node_ft=torch.randn(self.n_node+1,embed_dim)

    def random_walk(self,
            source:int,
            walk_length:int
        ):
        """
        Input:
            source
            walk_length
        Output:
            walk_seq
        """
        walk_seq=[source]
        cur_node=source
        for _ in range(walk_length-1):
            neighbors=self.adj.get(cur_node,[])
        
            # 이웃이 없으면 종료
            if len(neighbors)==0:
                break

            # 현재 노드의 이웃 중 하나를 균등 랜덤 선택
            next_node=random.choice(neighbors)
            walk_seq.append(next_node)
            cur_node=next_node
        return walk_seq

    def generate_walk_seq_list(self,
            num_walk:int,
            walk_length:int,
            shuffle_nodes:bool=True
        ):
        """
        Input
            num_walk: int, 각 노드마다 생성할 walk 개수
            walk_length: int, 각 walk sequence의 길이
        Output:
            walk_seq_list: list of walk_seq
        """
        nodes=list(self.adj.keys())
        walk_seq_list=[]
        for _ in range(num_walk):
            if shuffle_nodes:
                random.shuffle(nodes)
            for node in nodes:
                walk_seq=self.random_walk(
                    source=node,
                    walk_length=walk_length
                )
                walk_seq_list.append(walk_seq)
        return walk_seq_list