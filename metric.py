import torch
import torch.nn as nn
import os
import pandas as pd
# from dataset2 import MultiModalDataset, custom_collate_fn
from dataset2_new import MultiModalDataset, custom_collate_fn
from model_final import STGCN, TabNet, MultimodalClassifier
from fusion import AttentionFusionVariableLength
from torch.utils.data import DataLoader
import pickle
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix

# 데이터셋 정의
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')
    
    # model_path = '/home/khj/workspace/udjet/weights/cross_validation/model_epoch8_fold1.pth'
    # model_path = '/home/khj/workspace/udjet/weights/134_B_model.pth'
    # model_path = '/home/khj/workspace/udjet/weights/cross_validation/134_B_new_model_epoch17_fold3.pth'
    # model_path = '/home/khj/workspace/udjet/weights/cross_validation/134_B_new_model_epoch28_fold3.pth'
    # model_path = '/home/khj/workspace/udjet/weights/cross_validation/134_B_new_model_epoch16_fold2.pth'
    # model_path = '/home/khj/workspace/udjet/weights/cross_validation/134_B_new_model_epoch21_fold1.pth'
    model_path = '/home/khj/workspace/udjet/weights/cross_validation/134_B_new_model_epoch27_fold2.pth'
    
    checkpoint = torch.load(model_path, weights_only=True)

    with open('joint_matrix.pkl', 'rb') as f:
        adj_matrix = pickle.load(f)

    A_tensor = torch.tensor(adj_matrix).float()

    # 2. 각 task별 선언
    # tabular_model = TabNet(input_dim=18, hidden_dim=64, output_dim=8)
    tabular_model = TabNet(input_dim=15, hidden_dim=64, output_dim=8)
    tabular_model = nn.DataParallel(tabular_model)
    tabular_model.load_state_dict(checkpoint['tabular_model'])
    tabular_model.to(device)
    tabular_model.eval()

    skeleton_model = STGCN(num_classes=8, matrix=A_tensor)
    skeleton_model = nn.DataParallel(skeleton_model)
    skeleton_model.load_state_dict(checkpoint['skeleton_model'])
    skeleton_model.to(device)
    skeleton_model.eval()
    
    dim1 = 64   # tabular embedding shape
    dim2 = 256  # skeleton embeddng shape
    fusion_dim = 256    # fusion shape
    num_classes = 8
    fusion = AttentionFusionVariableLength(dim1, dim2, fusion_dim, num_classes)
    fusion = nn.DataParallel(fusion)
    fusion.load_state_dict(checkpoint['fusion'])
    fusion.to(device)
    fusion.eval()
    # fusion = torch.load(fusion_path)
    
    multimodal_model = MultimodalClassifier(fusion_dim, num_classes)
    multimodal_model = nn.DataParallel(multimodal_model)
    multimodal_model.load_state_dict(checkpoint['classifier'])
    multimodal_model.to(device)
    multimodal_model.eval()

    # tabular_path = './dataset/cnuh1_cnuh2/valid/tabular/data.xlsx'
    # skeleton_path = './dataset/cnuh1_cnuh2/valid/skeleton'
    # tabular_path = './dataset/cnuh1_cnuh2_sejong1_gachon1_134_B/valid/tabular/data.xlsx'
    # skeleton_path = './dataset/cnuh1_cnuh2_sejong1_gachon1_134_B/valid/skeleton'
    tabular_path = './dataset/cnuh1_cnuh2_sejong1_gachon1_134_B_new/valid/tabular/data.xlsx'
    skeleton_path = './dataset/cnuh1_cnuh2_sejong1_gachon1_134_B_new/valid/skeleton'

    dataset = MultiModalDataset(tabular_path, skeleton_path)
    
    batch_size = 8

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            pin_memory=True,
                            collate_fn=custom_collate_fn,
                            num_workers=32)
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for tabular, skeleton, label in dataloader:
            tabular = torch.nan_to_num(tabular, nan=0.0).to(device)
            tabular_embedding = tabular_model(tabular)

            skeleton_padded = pad_sequence([tensor.permute(1,0,2,3) for tensor in skeleton], 
                                batch_first=True, padding_value=0)
            
            skeleton_padded = skeleton_padded.permute(0,2,1,3,4).to(device)

            skeleton_embedding = skeleton_model(skeleton_padded)

            fused_embedding = fusion(tabular_embedding, skeleton_embedding)
            logit = multimodal_model(fused_embedding)

            preds = torch.argmax(logit, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(label)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # confusion matrix
    confusion_m = confusion_matrix(all_labels, all_preds)
    print(confusion_m)

    # Accuracy
    correct_pred = (all_preds == all_labels).sum().item()
    total_pred = all_labels.size(0)
    accuracy = correct_pred / total_pred

    # F1 score
    # f1_micro = f1_score(all_labels.numpy(), all_preds.numpy(), average='micro')
    f1_micro = f1_score(all_labels.numpy(), all_preds.numpy(), average='micro')
    f1_macro = f1_score(all_labels.numpy(), all_preds.numpy(), average='macro')
    f1_weighted = f1_score(all_labels.numpy(), all_preds.numpy(), average='weighted')

    print(f'Accuracy : {round(accuracy, 2)}')
    print(f'F1 score(micro) : {round(f1_micro, 2)}')
    print(f'F1 score(macro) : {round(f1_macro, 2)}')
    print(f'F1 score(weighted) : {round(f1_weighted, 2)}')
    # print(f'Accuracy : {accuracy}')
    # print(f'F1 score(micro) : {f1_micro}')
