import pandas as pd
import torch

class Sampling:
    @staticmethod
    def random_negative_sampling_for_static_graph(
            src:torch.Tensor,
            dst:torch.Tensor,
            n_node:int,
            seed:int|None=None,
            bipartite:bool=False,
            max_u:int=None
        ):
        """
        Input: all positive edge element
            src: [E,]
            dst: [E,]
            n_node: int
        Output:
            neg_dst: [E,]
        """
        device=src.device
        dtype=src.dtype

        # local random generator
        generator=None
        if seed is not None:
            generator=torch.Generator(device=device)
            generator.manual_seed(seed)

        # sampling range 설정
        if bipartite:
            low=max_u+1
        else:
            low=1
        high=n_node+1

        # 전체 positive edge를 Python set으로 변환
        src_list=src.cpu().tolist()
        dst_list=dst.cpu().tolist()
        pos_edge_set:set[tuple[int,int]]=set()
        for u,v in zip(src_list,dst_list):
            u=int(u)
            v=int(v)
            pos_edge_set.add((u,v))
        
        # sampling (neg_edge 중복 허용)
        neg_dst=torch.empty_like(dst)
        for idx,(u,pos_v) in enumerate(
                zip(src_list,dst_list)
            ):
            while True:
                neg_v=torch.randint(
                    low=low,
                    high=high,
                    size=(1,),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                ).item()

                neg_edge=(u,neg_v)

                # self-loop 제외
                if neg_v==u:
                    continue

                # 그래프 전체에서 실제 edge인 경우 제외
                if neg_edge in pos_edge_set:
                    continue
                
                neg_dst[idx]=neg_v
                break
        return neg_dst # [E,]