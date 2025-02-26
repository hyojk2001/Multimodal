from dataset import MultiModalDataset, custom_collate_fn
from models import STGCN, TabNet, MultimodalClassifier
from fusion import AttentionFusionVariableLength
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
import pickle
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold, StratifiedKFold
from torchmetrics.classification import MulticlassAccuracy

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tabular_path = './dataset/cnuh1_cnuh2_sejong1_gachon1_134_B_new/train/tabular/data.xlsx'
    skeleton_path = './dataset/cnuh1_cnuh2_sejong1_gachon1_134_B_new/train/skeleton'
    
    dataset = MultiModalDataset(tabular_path, skeleton_path)

    # print(train_dataset.__getitem__(3)[0])
    # print(train_dataset.__getitem__(3)[1].shape)

    # kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    labels = [dataset.__getitem__(i)[2] for i in range(dataset.__len__())]

    # ST-GCN 의 인접행렬 tensor 전환
    with open('joint_matrix.pkl', 'rb') as f:
        adj_matrix = pickle.load(f)

    A_tensor = torch.tensor(adj_matrix).float()

    # for fold, (train_idx, valid_idx) in enumerate(skf.split(dataset)):
    for fold, (train_idx, valid_idx) in enumerate(skf.split(dataset, labels)):

        print(f"Fold {fold + 1}")

        train_subset = Subset(dataset, train_idx)
        valid_subset = Subset(dataset, valid_idx)

        train_loader = DataLoader(train_subset,
                                  batch_size=8,
                                  shuffle=False, 
                                  pin_memory=True,
                                  collate_fn=custom_collate_fn, 
                                  num_workers=8)
        
        valid_loader = DataLoader(valid_subset,
                                  batch_size=8,
                                  shuffle=False, 
                                  pin_memory=True,
                                  collate_fn=custom_collate_fn, 
                                  num_workers=8)
        
        ## 모델, optimizer, loss 초기화 ##
        
        # 모달 임베딩 추출을 위한 모델(네트워크) 선언
        tabular_model = TabNet(input_dim=15, hidden_dim=64, output_dim=8)
        skeleton_model = STGCN(num_classes=8, matrix=A_tensor)

        # fusion 생성
        dim1 = 64
        dim2 = 256
        fusion_dim = 256
        num_classes = 8
        fusion = AttentionFusionVariableLength(dim1, dim2, fusion_dim, num_classes)
        
        # 최종 멀티모달 모델 선언
        multimodal_model = MultimodalClassifier(fusion_dim, num_classes)
        
        # 각 모델별 DP 적용
        tabular_model = nn.DataParallel(tabular_model)
        skeleton_model = nn.DataParallel(skeleton_model)
        fusion = nn.DataParallel(fusion)
        multimodal_model = nn.DataParallel(multimodal_model)

        # GPU 이동
        tabular_model = tabular_model.to(device)
        skeleton_model = skeleton_model.to(device)
        fusion = fusion.to(device)
        multimodal_model = multimodal_model.to(device)

        # optimizer 생성 > L2 정규화 적용(데이터 과적합 방지용)
        optimizer = optim.Adam(
                                list(tabular_model.parameters()) +
                                list(skeleton_model.parameters()) +
                                list(fusion.parameters()) + 
                                list(multimodal_model.parameters()),
                                lr=0.001,
                                weight_decay=1e-4
                                )

        criterion = nn.CrossEntropyLoss()

        train_accuracy = MulticlassAccuracy(num_classes=8).to(device)
        valid_accuracy = MulticlassAccuracy(num_classes=8).to(device)
        ## 모델, optimizer, loss 초기화 ##
        
        
        num_epochs = 30

        print('Training Started!')
        for epoch in range(num_epochs):
            tabular_model.train()
            skeleton_model.train()
            multimodal_model.train()
            fusion.train()

            train_loss = 0
            train_samples = 0

            train_accuracy.reset()

            for batch_idx, (tabular, skeleton, label) in enumerate(train_loader):
                
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
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item() * tabular.size(0)
                train_samples += tabular.size(0)
                train_accuracy.update(pred, label)

            train_loss /= train_samples
            train_acc = train_accuracy.compute()
            # train_loss += loss

            ## validation ##
            tabular_model.eval()
            skeleton_model.eval()
            multimodal_model.eval()
            fusion.eval()

            valid_loss = 0
            valid_samples = 0

            valid_accuracy.reset()

            with torch.no_grad():
                for batch_idx, (tabular, skeleton, label) in enumerate(valid_loader):
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
                    valid_accuracy.update(pred, label)
            valid_loss /= valid_samples  
            valid_acc = valid_accuracy.compute()

            print(f'epoch {epoch}/{num_epochs} | training loss : {train_loss:4f} | training accuracy : {train_acc:4f}')
            print(f'epoch {epoch}/{num_epochs} | validation loss : {valid_loss:4f} | validation accuracy : {valid_acc:4f}')

            ## 모델 저장 ##
            torch.save({
                        'tabular_model' : tabular_model.state_dict(),
                        'skeleton_model' : skeleton_model.state_dict(),
                        'fusion' : fusion.state_dict(),
                        'classifier' : multimodal_model.state_dict()
                        },
                        f'weights/cross_validation/model_epoch{epoch}_fold{fold+1}.pth'
                        )
            ## 모델 저장 ##

