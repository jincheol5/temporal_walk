import torch
import torch.nn as nn
from graph import Graph
from module import SkipGram

class DeepWalk_Base(nn.Module):
    def __init__(self,
            embed_dim:int,
            latent_dim:int,
            window:int,
            graph:Graph
        ):
        super().__init__()
        self.embed_dim=embed_dim
        self.latent_dim=latent_dim
        self.graph=graph

        self.skip_gram=SkipGram(
            vector_size=embed_dim,
            window=window
        )
        # row 0: padding, row 1 ~ N: 실제 node ID 1 ~ N
        self.node_ft=nn.Embedding(
            num_embeddings=self.graph.n_node+1,
            embedding_dim=self.embed_dim,
            padding_idx=0,
        )
        # downstream 학습 시 노드 임베딩을 고정
        self.node_ft.weight.requires_grad_(False)

    def convert_skipgram_to_torch_embedding(self):
        """
        Gensim Skip-gram의 center embedding을 PyTorch 임베딩으로 변환
        CPU에서 수행
        """

        embedding_weight=torch.zeros(
            size=(self.graph.n_node+1,self.embed_dim),
            dtype=torch.float32,
        )

        # row i에 실제 node ID i의 embedding 저장
        for node in range(1,self.graph.n_node+1):
            node_vector=torch.from_numpy(
                self.skip_gram.wv[node].copy()
            ).to(dtype=torch.float32)
            embedding_weight[node].copy_(node_vector)
        
        # 기존 nn.Embedding 객체는 유지하고 weight 값만 복사
        with torch.no_grad():
            self.node_ft.weight.copy_(embedding_weight)
            self.node_ft.weight[0].zero_() # row 0은 항상 padding zero vector
        self.skipgram_trained=False

    def train_skipgram(self,
            n_walk:int,
            walk_len:int,
            epoch:int=1
        ):
        """
        """
        ### Random Walk 생성
        walks=self.graph.generate_walks(
            n_walk=n_walk,
            walk_len=walk_len
        )

        ### vocabulary 생성
        self.skip_gram.build_vocab(walks)

        ### Skip-Gram 학습
        self.skip_gram.train(
            corpus_iterable=walks,
            total_examples=self.skip_gram.corpus_count,
            epochs=epoch
        )

        ### Gensim 임베딩을 PyTorch nn.Embedding으로 변환 후 저장
        self.convert_skipgram_to_torch_embedding()
        self.skipgram_trained=True

    def forward(self):
        return NotImplemented

class DeepWalk_Link_Prediction(DeepWalk_Base):
    def __init__(self, 
            embed_dim:int,
            latent_dim:int,
            window:int,
            graph:Graph
        ):
        super(DeepWalk_Link_Prediction,self).__init__(
            embed_dim=embed_dim, 
            latent_dim=latent_dim, 
            window=window,
            graph=graph 
        )
        # decoder
        self.decoder=nn.Sequential(
            nn.Linear(
                in_features=embed_dim+embed_dim,
                out_features=latent_dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=latent_dim,
                out_features=1
            )
        )
    def forward(self,
            pos_edge,
            neg_edge
        ):
        """
        Input:
            pos_edge
            neg_edge
        """