from dataset import MultiModalDataset, custom_collate_fn
from models import STGCN, TabNet, MultimodalClassifier
from fusion import AttentionFusionVariableLength
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
from torch.nn.utils.rnn import pad_sequence
import os
import sys

sys.path.append(os.getcwd())

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ST-GCN 네트워크에 파라미터로 들어가는 인접행렬 선언
    with open('joint_matrix.pkl', 'rb') as f:
        adj_matrix = pickle.load(f)

    A_tensor = torch.tensor(adj_matrix).float()
    # A_tensor = A_tensor.to(device)

    # 모달 임베딩 추출을 위한 모델(네트워크) 선언
    # tabular_model = TabNet(input_dim=18, hidden_dim=64, output_dim=8)
    # skeleton_model = STGCN(num_classes=8, matrix=A_tensor)
    tabular_model = TabNet(input_dim=18, hidden_dim=64, output_dim=8).to(device)
    skeleton_model = STGCN(num_classes=8, matrix=A_tensor).to(device)

    # fusion 생성
    dim1 = 64
    dim2 = 256
    # fusion_dim = 128
    fusion_dim = 256
    num_classes = 8
    fusion = AttentionFusionVariableLength(dim1, dim2, fusion_dim, num_classes).to(device)
    
    # 최종 멀티모달 모델 선언
    multimodal_model = MultimodalClassifier(fusion_dim, num_classes).to(device)

    # optimizer 생성 > L2 정규화 적용(데이터 과적합 방지용)
    optimizer = optim.SGD(
        list(tabular_model.parameters()) +
        list(skeleton_model.parameters()) +
        list(fusion.parameters()) + 
        list(multimodal_model.parameters()),
        lr=0.001,
        weight_decay=1e-4
    )

    # loss function 생성
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    
    # 1. tabular_tensor, skeleton 경로 지정
    # tabular_train_path = './dataset/cnuh1_cnuh2/train/tabular/data.xlsx'
    # skeleton_train_path = './dataset/cnuh1_cnuh2/train/skeleton'
    # tabular_valid_path = './dataset/cnuh1_cnuh2/valid/tabular/data.xlsx'
    # skeleton_valid_path = './dataset/cnuh1_cnuh2/valid/skeleton'

    tabular_train_path = './dataset/cnuh1_cnuh2_sejong1_gachon1_134_B/train/tabular/data.xlsx'
    skeleton_train_path = './dataset/cnuh1_cnuh2_sejong1_gachon1_134_B/train/skeleton'
    tabular_valid_path = './dataset/cnuh1_cnuh2_sejong1_gachon1_134_B/valid/tabular/data.xlsx'
    skeleton_valid_path = './dataset/cnuh1_cnuh2_sejong1_gachon1_134_B/valid/skeleton'
    
    # 2. dataset 클래스에서 불러오기
    train_dataset = MultiModalDataset(tabular_train_path, skeleton_train_path)
    valid_dataset = MultiModalDataset(tabular_valid_path, skeleton_valid_path)

    # dataloader 작성
    batch_size = 8

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  pin_memory=True,
                                  collate_fn=custom_collate_fn, 
                                  num_workers=8)
    
    valid_dataloader = DataLoader(valid_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=False, 
                                  pin_memory=True,
                                  collate_fn=custom_collate_fn, 
                                  num_workers=8)
    
    A_tensor = A_tensor.to(device)

    # epoch 지정
    num_epochs = 15
    
    # 가중치 저장 > validation loss가 전 대비 적어진 경우에만 저장
    best_valid_loss = float('inf')

    # early stopping > patience 만큼의 epoch 동안 validation loss가 나아지지 못하면 훈련 종료
    patience = 10
    counter = 0

    # # 훈련 시작
    print('Training Started')
    for epoch in range(num_epochs):
        ## training ##
        tabular_model.train()
        skeleton_model.train()
        multimodal_model.train()
        fusion.train()

        train_loss = 0
        train_samples = 0

        for batch_idx, (tabular, skeleton, label) in enumerate(train_dataloader):
            
            # 정형데이터가 nan이 나오는 경우를 방지
            tabular = torch.nan_to_num(tabular, nan=0.0)
            
            tabular = tabular.to(device)
            label = label.to(device)

            tabular_embedding = tabular_model(tabular)
            # print('tabular embedding shape :', tabular_embedding.shape)

            skeleton_padded = pad_sequence([tensor.permute(1,0,2,3) for tensor in skeleton], 
                                           batch_first=True, padding_value=0)
            
            skeleton_padded = skeleton_padded.permute(0,2,1,3,4).to(device)
            # print(skeleton_padded.shape)

            skeleton_embedding = skeleton_model(skeleton_padded)
            # print('skeleton embedding shape :', skeleton_embedding.shape)
            
            fused_embedding = fusion(tabular_embedding, skeleton_embedding)
            pred = multimodal_model(fused_embedding)

            loss = criterion(pred, label)
            # print('loss :', loss)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item() * tabular.size(0)
            train_samples += tabular.size(0)
        train_loss /= train_samples
        # train_loss += loss

        ## validation ##
        tabular_model.eval()
        skeleton_model.eval()
        multimodal_model.eval()
        fusion.eval()

        valid_loss = 0
        valid_samples = 0

        with torch.no_grad():
            for batch_idx, (tabular, skeleton, label) in enumerate(valid_dataloader):
                # 정형데이터가 혹시 nan이 나오는 경우를 방지
                tabular = torch.nan_to_num(tabular, nan=0.0)
                
                tabular = tabular.to(device)
                label = label.to(device)

                tabular_embedding = tabular_model(tabular)
                # print('tabular embedding shape :', tabular_embedding.shape)

                skeleton_padded = pad_sequence([tensor.permute(1,0,2,3) for tensor in skeleton], 
                                            batch_first=True, padding_value=0)
                
                skeleton_padded = skeleton_padded.permute(0,2,1,3,4).to(device)
                # print(skeleton_padded.shape)

                skeleton_embedding = skeleton_model(skeleton_padded)
                # print('skeleton embedding shape :', skeleton_embedding.shape)
                
                fused_embedding = fusion(tabular_embedding, skeleton_embedding)
                pred = multimodal_model(fused_embedding)

                loss = criterion(pred, label)
                # print('loss :', loss)
                
                valid_loss += loss.item() * tabular.size(0)
                valid_samples += tabular.size(0)
        valid_loss /= valid_samples        
        
        print(f'epoch {epoch}/{num_epochs} | training loss : {train_loss:4f} | validation loss : {valid_loss:4f}')

        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            print(f'Weight saved at epoch {epoch}.')
            torch.save({
                        'tabular_model' : tabular_model.state_dict(),
                        'skeleton_model' : skeleton_model.state_dict(),
                        'fusion' : fusion.state_dict(),
                        'classifier' : multimodal_model.state_dict()
                        },
                        'weights/134_B_model.pth'
                        )
            
        else:
            counter += 1

            if counter >= patience:
                print(f'Early Stopping triggered at epoch {epoch}.')
                break

