import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(GraphConvLayer, self).__init__()
        self.A = A + torch.eye(A.size(0))
        D = torch.sum(self.A, dim=1)
        self.D_inv_sqrt = torch.diag(torch.pow(D, -0.5))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # A_normalized = torch.matmul(torch.matmul(self.D_inv_sqrt, self.A), self.D_inv_sqrt)
        A_normalized = torch.matmul(torch.matmul(self.D_inv_sqrt, self.A), self.D_inv_sqrt).to(x.device)
        x = torch.einsum('nctv,vw->nctw', (x.float(), A_normalized))
        x = self.conv(x)
        return x

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, dropout_prob=0.5):
        super(STGCNBlock, self).__init__()
        self.gcn = GraphConvLayer(in_channels, out_channels, A)
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.bn(x)
        x = self.dropout(x)
        return F.relu(x)

class STGCN(nn.Module):
    def __init__(self, num_classes, matrix, in_channels=3, dropout_prob=0.5):
        super(STGCN, self).__init__()
        self.stgcn1 = STGCNBlock(in_channels, 64, matrix, dropout_prob=dropout_prob)
        self.stgcn2 = STGCNBlock(64, 64, matrix, dropout_prob=dropout_prob)
        self.stgcn3 = STGCNBlock(64, 64, matrix, dropout_prob=dropout_prob)
        self.stgcn4 = STGCNBlock(64, 128, matrix, stride=2, dropout_prob=dropout_prob)
        self.stgcn5 = STGCNBlock(128, 128, matrix, dropout_prob=dropout_prob)
        self.stgcn6 = STGCNBlock(128, 128, matrix, dropout_prob=dropout_prob)
        self.stgcn7 = STGCNBlock(128, 256, matrix, stride=2, dropout_prob=dropout_prob)
        self.stgcn8 = STGCNBlock(256, 256, matrix, dropout_prob=dropout_prob)
        self.stgcn9 = STGCNBlock(256, 256, matrix, dropout_prob=dropout_prob)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x의 shape: (batch_size, actions, frames, nodes, channels)
        batch_size, actions, frames, nodes, channels = x.shape

        # 1. 행동(action) 축을 batch와 병합 -> (batch_size * actions, frames, nodes, channels)
        # x = x.view(batch_size * actions, frames, nodes, channels)
        x = x.reshape(batch_size * actions, frames, nodes, channels)

        # 2. 차원 재배치 -> (batch_size * actions, channels, frames, nodes)
        x = x.permute(0, 3, 1, 2).contiguous()

        # 3. ST-GCN 블록을 순차 적용
        x = self.stgcn1(x)
        x = self.stgcn2(x)
        x = self.stgcn3(x)
        x = self.stgcn4(x)
        x = self.stgcn5(x)
        x = self.stgcn6(x)
        x = self.stgcn7(x)
        x = self.stgcn8(x)
        x = self.stgcn9(x)

        # 4. 평균 풀링 후 flatten
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.view(batch_size * actions, -1)  # (batch_size * actions, 256)

        # 5. 행동 축을 다시 합쳐 (batch_size, actions, 256)
        x = x.view(batch_size, actions, -1)

        # 6. 모든 행동에 대해 평균 임베딩 계산 -> (batch_size, 256)
        embeddings = x.mean(dim=1)

        return embeddings


class TabNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TabNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 최종 예측 레이어

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        embedding = x  # 임베딩 반환
        
        return embedding


class MultimodalClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultimodalClassifier, self).__init__()
        
        # Fully Connected Layers (Feature Extractor)
        self.fc1 = nn.Linear(input_dim, 256)
        # self.in1 = nn.InstanceNorm1d(256)  # BatchNorm1d > InstanceNorm1d로 변경
        self.in1 = nn.InstanceNorm1d(256)  # BatchNorm1d > InstanceNorm1d로 변경
        # self.in1 = nn.InstanceNorm1d(256, affine=False, track_running_stats=False)  # BatchNorm1d > InstanceNorm1d로 변경
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.in2 = nn.InstanceNorm1d(128)  # BatchNorm1d > InstanceNorm1d로 변경
        # self.in2 = nn.InstanceNorm1d(128, affine=False, track_running_stats=False)  # BatchNorm1d > InstanceNorm1d로 변경
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        # Residual Block (256 > 128)
        self.residual_fc = nn.Linear(256, 256)
        # self.residual_fc = nn.Linear(256, 128)
        
        # Final Classifier
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # 배치 크기가 1인 경우 InstanceNorm 사용을 위해 view 적용
        # InstanceNorm은 (N, C, L) 형태로 입력을 받으므로 변환 필요
        
        # out = self.fc1(x)
        # out = self.in1(out.unsqueeze(2)).squeeze(2)  # (N, C, L) > (N, C)
        
        if x.dim() == 2:
            out = self.fc1(x)
        else:
            out = self.in1(out.unsqueeze(2)).squeeze(2)  # (N, C, L) > (N, C)

        # print(out.shape)        
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Residual Connection (256 > 256)
        residual = out
        residual = self.residual_fc(residual)
        out = out + residual  # Skip Connection 적용
        # print(out.shape)
        # print(residual.shape)
        
        # Feature Refinement
        if x.dim() == 2:
            out = self.fc2(x)
        else:
            out = self.in2(out.unsqueeze(2)).squeeze(2)  # (N, C, L) > (N, C)
        
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Classification Head
        logits = self.classifier(out)
        return logits
    

