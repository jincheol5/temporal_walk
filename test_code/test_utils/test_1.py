import argparse
from utils import DataUtils

"""
<< Test >> 
utils.data_utils.DataUtils
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. DataUtils.preprocess_dataset()
            """
            print(f"<< convert to static graph >>")
            data=DataUtils.preprocess_static_graph(
                dataset_name=f"email-Enron"
            )
            print(f"n_node: {data['n_node']}")
            print(f"bipartite: {data['bipartite']}")
            print(f"min_u: {data['graph_df']['u'].min()}")
            print(f"max_u: {data['max_u']}")
            print(f"min_i: {data['graph_df']['i'].min()}")
            print(f"max_i: {data['graph_df']['i'].max()}")
            print(f"graph_df:")
            print(data['graph_df'].head(5))

            print(f"\n<< convert to temporal graph >>")
            data=DataUtils.preprocess_temporal_graph(
                dataset_name=f"enron"
            )
            print(f"n_node: {data['n_node']}")
            print(f"bipartite: {data['bipartite']}")
            print(f"min_u: {data['graph_df']['u'].min()}")
            print(f"max_u: {data['max_u']}")
            print(f"min_i: {data['graph_df']['i'].min()}")
            print(f"max_i: {data['graph_df']['i'].max()}")
            print(f"graph_df:")
            print(data['graph_df'].head(5))

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