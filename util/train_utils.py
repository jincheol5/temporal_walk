import torch

class TrainUtils:
    @staticmethod
    def padding_walks(
            walk_seq_list:list,
            device:torch.device=None
        ):
        """
        Input:
            walk_seq_list: list of walk_seq
        Output:
            walks: [len_walks,max_walk_seq_len] long tensor
        """
        max_walk_seq_len=max(len(walk_seq) for walk_seq in walk_seq_list)
        walks=torch.zeros(
            (len(walk_seq_list),max_walk_seq_len),
            dtype=torch.long,
            device=device
        )

        for i,walk_seq in enumerate(walk_seq_list):
            if len(walk_seq)==0:
                continue
            walks[i,:len(walk_seq)]=torch.tensor(
                walk_seq,
                dtype=torch.long,
                device=device,
            )
        return walks
