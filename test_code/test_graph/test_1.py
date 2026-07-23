import argparse
from utils import DataUtils
from graph import CTDNE_Graph

"""
<< Test >> 
graph.CTDNE_graph.CTDNE_Graph
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. 
                CTDNE_Graph.select_temporal_edge_uniform()
                CTDNE_Graph.select_temporal_edge_linear()
                CTDNE_Graph.select_temporal_edge_exponential()
            """
            data=DataUtils.preprocess_graph(dataset_name=f"enron")
            graph_df=data["graph_df"]
            bipartite=data["bipartite"]
            graph=CTDNE_Graph(
                graph_df=graph_df,
                bipartite=bipartite
            )
            graph.set_random_seed(seed=1)
            edge_1=graph.select_temporal_edge_uniform()
            edge_2=graph.select_temporal_edge_linear()
            edge_3=graph.select_temporal_edge_exponential()
            print(f"select temporal edge using uniform: {edge_1}")
            print(f"select temporal edge using linear: {edge_2}")
            print(f"select temporal edge using exponential: {edge_3}")

        case 2:
            """
            Test. 
                CTDNE_Graph.select_temporal_neighbor_uniform()
                CTDNE_Graph.select_temporal_neighbor_linear()
                CTDNE_Graph.select_temporal_neighbor_exponential()
            """
            data=DataUtils.preprocess_graph(dataset_name=f"enron")
            graph_df=data["graph_df"]
            bipartite=data["bipartite"]
            graph=CTDNE_Graph(
                graph_df=graph_df,
                bipartite=bipartite
            )
            graph.set_random_seed(seed=1)
            node_1,node_1_t=graph.select_temporal_neighbor_uniform(node=1,cur_t=0.0)
            node_2,node_2_t=graph.select_temporal_neighbor_linear(node=1,cur_t=0.0)
            node_3,node_3_t=graph.select_temporal_neighbor_exponential(node=1,cur_t=0.0)
            print(f"neighbors of node 1: {graph.adj[1]}")
            print(f"select temporal neighbor using uniform: {node_1} at {node_1_t}")
            print(f"select temporal neighbor using linear: {node_2} at {node_2_t}")
            print(f"select temporal neighbor using exponential: {node_3} at {node_3_t}")

        case 3:
            """
            Test. 
                CTDNE_Graph.temporal_random_walk()
            """
            data=DataUtils.preprocess_graph(dataset_name=f"enron")
            graph_df=data["graph_df"]
            bipartite=data["bipartite"]
            graph=CTDNE_Graph(
                graph_df=graph_df,
                bipartite=bipartite
            )
            graph.set_random_seed(seed=1)
            walk_1=graph.temporal_random_walk(
                source=1,
                start_t=0.0,
                walk_len=5,
                neighbor_sampling="uniform"
            )
            walk_2=graph.temporal_random_walk(
                source=1,
                start_t=0.0,
                walk_len=5,
                neighbor_sampling="linear"
            )
            walk_3=graph.temporal_random_walk(
                source=1,
                start_t=0.0,
                walk_len=5,
                neighbor_sampling="exponential"
            )
            print(f"walk of uniform: {walk_1}")
            print(f"walk of linear: {walk_2}")
            print(f"walk of exponential: {walk_3}")

        case 4:
            """
            Test. 
                CTDNE_Graph.generate_walks()
            """
            data=DataUtils.preprocess_graph(dataset_name=f"enron")
            graph_df=data["graph_df"]
            bipartite=data["bipartite"]
            graph=CTDNE_Graph(
                graph_df=graph_df,
                bipartite=bipartite
            )
            graph.set_random_seed(seed=1)
            walks=graph.generate_walks(
                walk_len=5,
                min_walk_len=3,
                n_context_window=100,
                edge_sampling="uniform",
                neighbor_sampling="uniform"
            )
            print(walks)

if __name__=="__main__":
    """
    Execute test_fn
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_num",type=int,default=1)
    args=parser.parse_args()
    test_config={
        "test_num":args.test_num
    }
    test_fn(**test_config)