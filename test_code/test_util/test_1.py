import argparse
import torch
from util import TrainUtils

"""
<< Test >> 
utils.data_utils.TrainUtils
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. TrainUtils.padding_walks
            """
            walk_seq_list=[
                [1,3,5,2],
                [2,4],
                [3,1,6,7,8],
            ]
            walks=TrainUtils.padding_walks(
                walk_seq_list=walk_seq_list,
                device=torch.device("cpu")
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