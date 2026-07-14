import argparse
from utils import DataUtils
from graph import Graph

"""
<< Test >> 
graph.graph.Graph
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. Graph.random_walk()
            """
            data=DataUtils.preprocess_static_graph(
                dataset_name=f"email-Enron"
            )
            graph_df=data["graph_df"]
            graph=Graph(graph_df=graph_df)
            source=1
            walk=graph.random_walk(source=source,walk_len=5)
            print(f"walk sequence at source {source}: ")
            print(walk)

        case 2:
            """
            Test. Graph.generate_walks()
            """
            data=DataUtils.preprocess_static_graph(
                dataset_name=f"email-Enron"
            )
            graph_df=data["graph_df"]
            graph=Graph(graph_df=graph_df)
            n_walk=1
            walk_len=5
            walks=graph.generate_walks(
                n_walk=n_walk,
                walk_len=walk_len
            )
            print(f"walk sequence list: ")
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