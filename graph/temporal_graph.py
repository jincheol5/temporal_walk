import random
import math
import pandas as pd
from collections import defaultdict
from bisect import bisect_right
from utils import DataUtils

class TemporalGraph:
    """
    node_id=0: padding node
    """
    def __init__(self,
            graph_df:pd.DataFrame,
            bipartite:bool=False
        ):
        """
        Input:
            graph_df: pd.DataFrame
            bipartite: bool
        """
        # set adj, adj_t, edge_events
        self.adj=defaultdict(list)
        self.adj_t=defaultdict(list)
        self.edge_events=[]
        for event in graph_df.itertuples(index=False): # col: [u,i,ts]
            src=int(event.u)
            dst=int(event.i)
            t=float(event.ts)
            # edge 양방향 저장
            self.adj[dst].append(src)
            self.adj_t[dst].append(t)
            self.adj[src].append(dst)
            self.adj_t[src].append(t)
            self.edge_events.append((src,dst,t))
            self.edge_events.append((dst,src,t))

        # set n_node, bipartite, max_u
        self.n_node=max(graph_df["u"].max(),graph_df["i"].max())
        self.bipartite=bipartite
        self.max_u=graph_df["u"].max()
        self.max_t=graph_df["ts"].max()

    def select_temporal_edge_random(self,
            seed:int
        ):
        """
        모든 temporal edge를 동일한 확률로 선택
        """
        sampling=random.Random(seed)
        return sampling.choice(self.edge_events)

    def select_temporal_edge_linear(self,
            seed:int
        ):
        """
        시간순 rank에 비례해 temporal edge 선택
        """
        sampling=random.Random(seed)
        n_edge=len(self.edge_events)
        return sampling.choices(
            population=self.edge_events,
            weights=range(1,n_edge+1), # 가장 최근 edge가 가중치 높도록 설정
            k=1,
        )[0]

    def select_temporal_edge_exponential(self,
            seed:int,
            temperature:float=1.0
        ):
        """
        최근 edge일수록 지수적으로 높은 확률로 선택
        P(e) ∝ exp((t_e - t_max) / temperature)
        """
        if temperature<=0:
            raise ValueError("temperature는 0보다 커야 합니다.")
        sampling=random.Random(seed)
        weights=[
            math.exp((t-self.max_t)/temperature)
            for _,_,t in self.edge_events
        ]
        return sampling.choices(
            population=self.edge_events,
            weights=weights,
            k=1,
        )[0]

    def select_temporal_neighbor_random(self,
            seed:int,
            source:int,
            current_t:float,
        ):
        """
        현재 source 노드에서 timestamp >= current_t인 temporal edge 중
        하나를 균등한 확률로 선택한다.

        self.adj_t[source]는 timestamp 오름차순으로 정렬되어 있다고 가정한다.
        Input:
        
        Output:
            neighbor_id
            neighbor_t
        """
        sampling=random.Random(seed)
        timestamps=self.adj_t[source]
        if not timestamps:
            return None

        # timestamp > current_t를 처음 만족하는 위치
        start_idx=bisect_right(timestamps,current_t)
        if start_idx==len(timestamps):
            return None

        # [start_idx,len(timestamps)-1]에서 균등하게 index 선택
        selected_idx=sampling.randrange(
            start_idx,
            len(timestamps),
        )
        return self.adj[source][selected_idx],self.adj_t[source][selected_idx]

    def temporal_random_walk(self,
            source:int,
            start_t:float,
            walk_len:int
        ):
        """
        """