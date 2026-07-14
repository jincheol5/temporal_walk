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
            graph_df:pd.DataFrame
        ):
        """
        Input:
            graph_df: pd.DataFrame
        """
        # set adj
        self.adj=defaultdict(list)
        for event in graph_df.itertuples(index=False): # col: [u,i]
            src=int(event.u)
            dst=int(event.i)

            # edge 양방향 저장
            self.adj[dst].append(src)
            self.adj[src].append(dst)

        # 각 노드의 인접 리스트에서 중복 제거
        for node in self.adj:
            self.adj[node]=list(dict.fromkeys(self.adj[node]))

        # set n_node
        self.n_node=max(graph_df["u"].max(),graph_df["i"].max())

    def random_walk(self,
            source:int,
            walk_len:int
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
        for _ in range(walk_len-1):
            neighbors=self.adj.get(cur_node,[])
        
            # 이웃이 없으면 종료
            if len(neighbors)==0:
                break

            # 현재 노드의 이웃 중 하나를 균등 랜덤 선택
            next_node=random.choice(neighbors)
            walk_seq.append(next_node)
            cur_node=next_node
        return walk_seq

    def generate_walks(self,
            n_walk:int,
            walk_len:int,
            shuffle_nodes:bool=True
        ):
        """
        Input
            num_walk: int, 각 노드마다 생성할 walk 개수
            walk_length: int, 각 walk sequence의 길이
        Output:
            walks: list of walk_seq
        """
        nodes=list(self.adj.keys())
        walks=[]
        for _ in range(n_walk):
            if shuffle_nodes:
                random.shuffle(nodes)
            for node in nodes:
                walk_seq=self.random_walk(
                    source=node,
                    walk_len=walk_len
                )
                walks.append(walk_seq)
        return walks