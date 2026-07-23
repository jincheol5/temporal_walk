import random
import math
from typing import Literal
from bisect import bisect_right
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