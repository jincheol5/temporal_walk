import argparse
from utils import DataUtils
from graph import TemporalGraph

"""
<< Test >> 
graph.temporal_graph.TemporalGraph
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. TemporalGraph.random_walk()
            """
            data=DataUtils.preprocess_temporal_graph(
                dataset_name=f"enron"
            )
            graph_df=data["graph_df"]

        case 2:
            """
            Test. 
            """

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