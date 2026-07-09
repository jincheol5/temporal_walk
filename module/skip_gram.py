import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    """
    Skip-gram with Negative Sampling 모델.

    negative sampling 자체는 외부에서 수행한다고 가정한다.

    입력:
        center_node: [B]
        pos_context_node: [B]
        neg_context_node: [B, K]

    출력:
        loss: scalar
    """

    def __init__(
        self,
        n_node: int,
        embed_dim: int,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.n_node = n_node
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx

        # 중심 노드 embedding
        self.center_embedding = nn.Embedding(
            num_embeddings=n_node,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )

        # 주변 노드 embedding
        self.context_embedding = nn.Embedding(
            num_embeddings=n_node,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Word2Vec류 모델에서 자주 쓰는 간단한 초기화.
        """
        init_range = 0.5 / self.embed_dim

        self.center_embedding.weight.data.uniform_(
            -init_range,
            init_range,
        )

        self.context_embedding.weight.data.uniform_(
            -init_range,
            init_range,
        )

        if self.padding_idx is not None:
            self.center_embedding.weight.data[self.padding_idx].fill_(0)
            self.context_embedding.weight.data[self.padding_idx].fill_(0)

    def forward(
        self,
        center_node: torch.Tensor,
        pos_context_node: torch.Tensor,
        neg_context_node: torch.Tensor,
    ) -> torch.Tensor:
        """
        center_node:
            shape [B]

        pos_context_node:
            shape [B]

        neg_context_node:
            shape [B, K]

        return:
            scalar loss
        """

        # [B, D]
        center_embed = self.center_embedding(center_node)

        # [B, D]
        pos_context_embed = self.context_embedding(pos_context_node)

        # [B, K, D]
        neg_context_embed = self.context_embedding(neg_context_node)

        # positive score: [B]
        pos_score = torch.sum(
            center_embed * pos_context_embed,
            dim=1,
        )

        # negative score: [B, K]
        neg_score = torch.bmm(
            neg_context_embed,
            center_embed.unsqueeze(2),
        ).squeeze(2)

        # positive pair는 score가 커지도록
        pos_loss = F.logsigmoid(pos_score)

        # negative pair는 score가 작아지도록
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)

        # maximize를 minimize 문제로 바꾸기 위해 - 부호
        loss = -(pos_loss + neg_loss).mean()

        return loss

    def get_node_embedding(self) -> torch.Tensor:
        """
        학습 후 사용할 node embedding.

        일반적으로 center embedding을 사용한다.
        """
        return self.center_embedding.weight

    def get_context_embedding(self) -> torch.Tensor:
        """
        context embedding을 따로 확인하고 싶을 때 사용.
        """
        return self.context_embedding.weight