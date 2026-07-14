import argparse
from utils import DataUtils,TrainUtils

"""
<< Test >> 
utils.data_utils.TrainUtils
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. TrainUtils.split_graph_df()
            """
            data=DataUtils.preprocess_dataset(
                dataset_name=kwargs["dataset_name"],
                is_temporal=False
            )
            graph_df=data["graph_df"]
            train_df,val_df,test_df=TrainUtils.split_graph_df(
                df=graph_df,
                is_temporal=False
            )
            print(f"all row len: {len(graph_df)}")
            print(f"train row len: {len(train_df)}")
            print(f"val row len: {len(val_df)}")
            print(f"test row len: {len(test_df)}")

if __name__=="__main__":
    """
    Execute test_fn
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_num",type=int,default=1)
    parser.add_argument("--dataset_name",type=str,default=f"enron")
    args=parser.parse_args()
    test_config={
        "test_num":args.test_num,
        "dataset_name":args.dataset_name
    }
    test_fn(**test_config)