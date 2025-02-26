import torch
import torch.nn as nn
import os
import pandas as pd
from dataset2 import MultiModalDataset, custom_collate_fn
from model_final import STGCN, TabNet, MultimodalClassifier
from fusion import AttentionFusionVariableLength
from torch.utils.data import DataLoader
import pickle
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')

    status = {
            '정상': 0, 
            '오십견': 1, 
            '회전근개': 2, 
            '퇴행성무릎': 3, 
            '척수손상': 4, 
            '파킨슨': 5, 
            '근감소증': 6, 
            '뇌졸중': 7
            }

    with open('joint_matrix.pkl', 'rb') as f:
        adj_matrix = pickle.load(f)

    A_tensor = torch.tensor(adj_matrix).float()

    ## 모델 로드
    # 1. 경로
    tabular_model_path = './weights/tabular_cnuh1_cnuh2.pth'
    skeleton_model_path = './weights/skeleton_cnuh1_cnuh2.pth'
    fusion_path = './weights/fusion_cnuh1_cnuh2.pth'
    multimodal_model_path = './weights/classifier_cnuh1_cnuh2.pth'

    # 2. 각 task별 선언
    tabular_model = TabNet(input_dim=18, hidden_dim=64, output_dim=8)
    tabular_model.load_state_dict(torch.load(tabular_model_path))
    tabular_model.to(device)
    tabular_model.eval()
    # tabular_model = torch.load(tabular_model_path)
    
    skeleton_model = STGCN(num_classes=8, matrix=A_tensor)
    skeleton_model.load_state_dict(torch.load(skeleton_model_path))
    skeleton_model.to(device)
    skeleton_model.eval()
    # skeleton_model = torch.load(skeleton_model_path)
    
    dim1 = 64   # tabular embedding shape
    dim2 = 256  # skeleton embeddng shape
    fusion_dim = 256    # fusion shape
    num_classes = 8
    fusion = AttentionFusionVariableLength(dim1, dim2, fusion_dim, num_classes)
    fusion.load_state_dict(torch.load(fusion_path))
    fusion.to(device)
    fusion.eval()
    # fusion = torch.load(fusion_path)
    
    multimodal_model = MultimodalClassifier(fusion_dim, num_classes)
    multimodal_model.load_state_dict(torch.load(multimodal_model_path))
    multimodal_model.to(device)
    multimodal_model.eval()
    # multimodal_model = torch.load(multimodal_model_path)

    ## 데이터 로드
    test_tabular_path = './dataset/cnuh1/test/tabular/data.xlsx'
    test_skeleton_path = './dataset/cnuh1/test/skeleton'

    test_dataset = MultiModalDataset(test_tabular_path, test_skeleton_path)

    batch_size = 1

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 pin_memory=True,
                                 collate_fn=custom_collate_fn,
                                 num_workers=4)

    ## 모델 기반 데이터 추출
    # 1. 클래스 추출
    # 2. 확률 추출

    with torch.no_grad():
        for tabular, skeleton, label in test_dataloader:
            tabular = torch.nan_to_num(tabular, nan=0.0).to(device)
            tabular_embedding = tabular_model(tabular)

            skeleton_padded = pad_sequence([tensor.permute(1,0,2,3) for tensor in skeleton], 
                                batch_first=True, padding_value=0)
            
            skeleton_padded = skeleton_padded.permute(0,2,1,3,4).to(device)

            skeleton_embedding = skeleton_model(skeleton_padded)

            fused_embedding = fusion(tabular_embedding, skeleton_embedding)
            logit = multimodal_model(fused_embedding)

    # prob = F.softmax(logit, dim=-1)
    prob = F.softmax(logit, dim=-1)

    # prob = max(F.softmax(logit, dim=-1))
    # pred = torch.argmax(prob, dim=-1)

    max_prob, predicted_class = torch.max(prob, dim=-1)

    # print(max_prob)
    # print(predicted_class)

    status_reverse = {v:k for k,v in status.items()}

    # print(f'{round(max_prob.item()*100)}% 의 확률로 {status_reverse.get(predicted_class.item())}')
    print(f'예측 : {status_reverse.get(predicted_class.item())}, 확률 : {round(max_prob.item()*100)}%')
    print(f'실제 : {status_reverse.get(label.item())}')

    # 결과 xlsx화
    df = pd.DataFrame({'예측' : [status_reverse.get(predicted_class.item())],
                       '확률' : [round(max_prob.item()*100)]})

    df.to_excel('test_inference.xlsx', index=False)    