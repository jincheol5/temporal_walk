import random
import math
import numpy as np
import networkx as nx
from typing import Literal
from .temporal_graph import TemporalGraph

class ATDGEB_Graph(TemporalGraph):
    def __init__(self,
            graph_df,
            bipartite:bool=False
        ):
        super().__init__(
            graph_df=graph_df,
            bipartite=bipartite
        )
        self.topological_graph=nx.from_pandas_edgelist(
            self.graph_df,
            source="u",
            target="i",
            create_using=nx.Graph() # Undirected Graph
        )

    def detect_k_core_community(self,
            w_list:list
        )->dict[int,set[int]]:
        """
        Input:
            w_list: list of w
        Output:
            dict
                key: level
                value: node set
        """
        G=self.topological_graph.copy()
        core_numbers=nx.core_number(G) # 각 노드의 core_number dict
        return {
            level: {
                node
                for node,core_number in core_numbers.items()
                if core_number>=level
            }
            for level in w_list
        }

    def detect_k_truss_community(self,
            y_list:list
        )->dict[int, set[int]]:
        """
        Input:
            y: list of y
        Output:
            dict
                key: level
                value: node set
        """
        if any(level<2 for level in y_list):
            raise ValueError(
                "k-truss level은 2 이상이어야 합니다."
            )
        G=self.topological_graph.copy()
        # 간선이 없는 고립 노드 제거
        G.remove_nodes_from(
            list(nx.isolates(G))
        )
        return {
            level:set(
                nx.k_truss(
                    G,
                    k=level
                ).nodes()
            )
            for level in y_list
        }

    def detect_k_clique_community(self,
            z_list:list
        )->dict[int,set[int]]:
        """
        Input:
            z: list of z
        Output:
            dict
                key: level
                value: node set
        """
        if any(level<2 for level in z_list):
            raise ValueError(
                "k-clique level은 2 이상이어야 합니다."
            )
        G=self.topological_graph.copy()

        # maximal clique 탐지
        maximal_cliques=list(
            nx.find_cliques(G)
        )

        return {
            level: {
                node
                for clique in maximal_cliques
                if len(clique)>=level
                for node in clique
            }
            for level in z_list
        }

    def generate_init_local_struct_vec(self,
            k_list:list
        ):
        """
        Input:
            k_list: list of k
        Output:
            local_struct_vec:
                길이: self.n_node + 1

                struct_vec[node_id]:
                    해당 노드의 초기 local structure vector

                struct_vec[0]:
                    padding node의 zero vector

                각 노드 벡터의 길이:
                    3 * len(k_list)

                벡터 구성:
                    [k-core | k-truss | k-clique]
        """
        if len(k_list)!=len(set(k_list)):
            raise ValueError(
                "k_list에는 중복된 값이 없어야 합니다."
            )

        # 세 종류의 community 탐지
        k_core_com=self.detect_k_core_community(
            w_list=k_list
        )
        k_truss_com=self.detect_k_truss_community(
            y_list=k_list
        )
        k_clique_com=self.detect_k_clique_community(
            z_list=k_list
        )
        n_level=len(k_list)
        struct_dim=3*n_level

        # index가 node ID와 같도록 self.n_node + 1개 생성
        # node 0은 padding node이므로 zero vector로 유지
        local_struct_vec=[
            np.zeros(
                struct_dim,
                dtype=np.float32
            )
            for _ in range(self.n_node+1)
        ]

        level_to_idx={
            k: idx
            for idx,k in enumerate(k_list)
        }
        community_results=[
            k_core_com,
            k_truss_com,
            k_clique_com,
        ]

        for community_idx,community_result in enumerate(
                community_results
            ):
            offset=community_idx*n_level
            for k,node_set in community_result.items():
                level_idx=level_to_idx[k]
                for node in node_set:
                    local_struct_vec[node][offset+level_idx]=k
        return local_struct_vec

    def compute_similarity(self,
            vec_a:np.ndarray,
            vec_b:np.ndarray
        )->float:
        """
        두 local structure vector의 cosine similarity를 계산합니다.
        두 벡터 중 하나가 zero vector이면 0을 반환합니다.
        """
        norm_a=np.linalg.norm(vec_a)
        norm_b=np.linalg.norm(vec_b)
        if norm_a==0 or norm_b==0:
            return 0.0
        similarity=np.dot(
            vec_a,
            vec_b
        ) / (norm_a * norm_b)
        # 부동소수점 오차 및 음수 가중치 방지
        return float(
            np.clip(
                similarity,
                0.0,
                1.0
            )
        )

    def aggregate_local_struct_vec(self,
            init_stru:list[np.ndarray],
            L:int
        )->list[np.ndarray]:
        """
        Input:
            init_stru: init local structure vector list
            L: aggregate 반복 횟수
        Output:
            stru: aggregated local structure vector list
        """

        # 원본 init_stru를 변경하지 않도록 복사
        stru=[
            np.asarray(
                vec,
                dtype=np.float32
            ).copy()
            for vec in init_stru
        ]

        # node 0은 padding node
        stru[0].fill(0.0)

        for _ in range(L):
            # 현재 layer 계산에는 이전 layer의 벡터만 사용
            prev_stru=stru
            next_stru=[
                vec.copy()
                for vec in prev_stru
            ]
            for node in range(1,self.n_node+1):
                neighbors=list(
                    self.topological_graph.neighbors(node)
                )
                # 고립 노드는 이전 벡터 유지
                if not neighbors:
                    continue

                similarities=np.asarray([
                    self.compute_similarity(
                        prev_stru[node],
                        prev_stru[neighbor]
                    )
                    for neighbor in neighbors
                ],dtype=np.float32)

                similarity_sum=float(
                    similarities.sum()
                )

                # 모든 이웃과의 유사도가 0인 경우
                # 논문의 weight 식을 계산할 수 없으므로 집계하지 않음
                if similarity_sum==0.0:
                    continue

                weights=(
                    similarities / similarity_sum
                )
                aggregated_vec=np.zeros_like(
                    prev_stru[node]
                )
                for neighbor,weight in zip(
                    neighbors,
                    weights
                ):
                    aggregated_vec+=(
                        weight*prev_stru[neighbor]
                    )
                next_stru[node]=(
                    prev_stru[node]+aggregated_vec
                )
            next_stru[0].fill(0.0) # padding node는 모든 layer에서 zero vector 유지
            stru=next_stru
        return stru