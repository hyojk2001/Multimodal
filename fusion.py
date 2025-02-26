import torch
import torch.nn as nn
import random
import numpy as np

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AttentionFusionVariableLength(nn.Module):
    def __init__(self, dim1, dim2, fusion_dim, num_classes):
        super(AttentionFusionVariableLength, self).__init__()
        
        self.fusion_dim = fusion_dim

        # ✅ 선형 변환: A와 B를 동일한 차원으로 변환
        self.query_transform = nn.Linear(dim1, fusion_dim)
        self.key_transform = nn.Linear(dim2, fusion_dim)
        self.value_transform = nn.Linear(dim2, fusion_dim)

        # ✅ Softmax와 Classifier
        self.softmax = nn.Softmax(dim=-1)
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, embedding_a, embedding_b):
        """
        embedding_a: (1, 64)  
        embedding_b: (n, 256) where 1 <= n <= 17
        """

        # Step 1: Linear Transformation
        Q = self.query_transform(embedding_a)  # (1, fusion_dim)
        K = self.key_transform(embedding_b)    # (n, fusion_dim)
        V = self.value_transform(embedding_b)  # (n, fusion_dim)

        # Step 2: Attention Score 계산 (QK^T)
        # (1, fusion_dim) x (fusion_dim, n) → (1, n)
        # attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (fusion_dim ** 0.5)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.fusion_dim ** 0.5)
        attention_weights = self.softmax(attention_scores)  # (1, n)

        # Step 3: Weighted Sum 적용 (어텐션 가중합 계산)
        # (1, n) x (n, fusion_dim) → (1, fusion_dim)
        weighted_sum = torch.matmul(attention_weights, V)

        # Step 4: 최종 합성 임베딩 계산 (Q + 가중합)
        fused_embedding = Q + weighted_sum

        # Step 5: 라벨 예측
        # logits = self.classifier(fused_embedding)  # (1, num_classes)
        # return fused_embedding, logits, attention_weights
        return fused_embedding