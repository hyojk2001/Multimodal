import optuna
# from dataset2 import MultiModalDataset, custom_collate_fn
from dataset2_new import MultiModalDataset, custom_collate_fn
from model_final import STGCN, TabNet, MultimodalClassifier
from fusion import AttentionFusionVariableLength
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
import pickle
from torch.nn.utils.rnn import pad_sequence
import os

# 모델 내 bayesian optimization 파라미터 확인

# 1.TabNet
# 2. ST-GCN
# 3. Fusion
# 4. Classifier

def train_loop(tabular_model,
               skeleton_model,
               fusion,
               multimodal_model,
               optimizer,
               criterion,
               train_loader,
               valid_loader,
               device,
               epochs):
    
    tabular_model.to(device)
    skeleton_model.to(device)
    fusion.to(device)
    multimodal_model.to(device)

    ## loop ##
    # for epoch in range(epochs):
    for epoch in range(epochs):
        tabular_model.train()
        skeleton_model.train()
        fusion.train()
        multimodal_model.train()

        train_loss = 0
        train_samples = 0

        for _, (tabular, skeleton, label) in enumerate(train_loader):
            
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
        train_loss /= train_samples

        tabular_model.eval()
        skeleton_model.eval()
        multimodal_model.eval()
        fusion.eval()

        valid_loss = 0
        valid_samples = 0

        with torch.no_grad():
            for _, (tabular, skeleton, label) in enumerate(valid_loader):
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

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}")
    ## loop ##
    
    return valid_loss

# def objective(trial):
def objective(trial, train_loader, valid_loader, device):

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    # lr = trial.suggest_loguniform('lr', 0.001, 0.1)
    # weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    # dropout = trial.suggest_float('dropout', 0.1, 0.6)
    lr = trial.suggest_float('lr', 0.001, 0.1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.6, log=True)

    optimizer_category = trial.suggest_categorical("optimizer", ['SGD', 'Adam', 'AdamW'])
    # fusion_dim = trial.suggest_int('fusion_dimension', 64, 256)

    with open('joint_matrix.pkl', 'rb') as f:
        adj_matrix = pickle.load(f)

    adj_matrix_tensor = torch.tensor(adj_matrix).float()
    
    num_classes = 8

    dim1 = 64
    dim2 = 256
    fusion_dim = 256

    tabular_model = TabNet(input_dim=18, 
                           hidden_dim=64, 
                           output_dim=8).to(device)
    
    skeleton_model = STGCN(num_classes=8, 
                           matrix=adj_matrix_tensor, 
                           dropout_prob=dropout).to(device)
    
    fusion = AttentionFusionVariableLength(dim1, dim2, 
                                           fusion_dim, 
                                           num_classes).to(device)
    
    multimodal_model = MultimodalClassifier(fusion_dim, 
                                            num_classes).to(device)

    # optimizer = optim.Adam(list(tabular_model.parameters()) +
    #                        list(skeleton_model.parameters()) +
    #                        list(fusion.parameters()) + 
    #                        list(multimodal_model.parameters()),
    #                        lr=lr,
    #                        weight_decay=weight_decay
    #                        )
    
    if optimizer_category == 'SGD':
        optimizer = optim.SGD(list(tabular_model.parameters()) +
                              list(skeleton_model.parameters()) +
                              list(fusion.parameters()) + 
                              list(multimodal_model.parameters()),
                              lr=lr,
                              weight_decay=weight_decay
                              )
    if optimizer_category == 'Adam':
        optimizer = optim.Adam(list(tabular_model.parameters()) +
                               list(skeleton_model.parameters()) +
                               list(fusion.parameters()) + 
                               list(multimodal_model.parameters()),
                               lr=lr,
                               weight_decay=weight_decay
                               )
    if optimizer_category == 'AdamW':
        optimizer = optim.AdamW(list(tabular_model.parameters()) +
                                list(skeleton_model.parameters()) +
                                list(fusion.parameters()) + 
                                list(multimodal_model.parameters()),
                                lr=lr,
                                weight_decay=weight_decay
                                )
    
    criterion = nn.CrossEntropyLoss()

    epochs = 20

    final_loss = train_loop(tabular_model,
                            skeleton_model,
                            fusion,
                            multimodal_model,
                            optimizer,
                            criterion,
                            train_loader,
                            valid_loader,
                            device,
                            epochs
                            )
    
    trial.report(final_loss, step=0)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return final_loss
    
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')

    ## 데이터 정의 ##
    # 1. tabular_tensor, skeleton 경로 지정
    tabular_train_path = './dataset/cnuh1_cnuh2/train/tabular/data.xlsx'
    skeleton_train_path = './dataset/cnuh1_cnuh2/train/skeleton'

    tabular_valid_path = './dataset/cnuh1_cnuh2/valid/tabular/data.xlsx'
    skeleton_valid_path = './dataset/cnuh1_cnuh2/valid/skeleton'
    
    # 2. dataset 클래스에서 불러오기
    train_dataset = MultiModalDataset(tabular_train_path, skeleton_train_path)
    valid_dataset = MultiModalDataset(tabular_valid_path, skeleton_valid_path)

    # 3. dataloader 작성
    batch_size = 8

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, 
                              shuffle=True, 
                              pin_memory=True,
                              collate_fn=custom_collate_fn,
                              prefetch_factor=4, 
                              num_workers=8)
    
    val_loader = DataLoader(valid_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            pin_memory=True,
                            collate_fn=custom_collate_fn, 
                            prefetch_factor=4, 
                            num_workers=8)
    ## 데이터 정의 ##

     
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, device), 
                   n_trials=12)
    # study.optimize(objective, n_trials=30)

    print(f"Best Parameters: {study.best_params}")