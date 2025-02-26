from dataset import TabularDataset, SkeletonDataset, MultiModalDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import os
import shutil
import pandas as pd
import numpy as np

if __name__ == '__main__':

    ### tabular dataset에서 label을 기준으로 train/valid 등 split 진행 ###
    
    # dataset 생성
    tabular_path = '/home/khj/workspace/udjet/data/tabular/cnuh1_cnuh2_sejong1_gachon1_134_B_new.xlsx'
    skeleton_path = '/home/khj/workspace/udjet/data/skeleton_cnuh1_cnuh2_sejong1_gachon1_B'
    
    # 2. dataset2에서 불러오기
    tabular_dataset = TabularDataset(tabular_path, skeleton_path)
    
    # dataloader 생성
    indices = list(map(int, os.listdir(skeleton_path)))
    labels = [tabular_dataset.__getitem__(i)[1]['상태'].iloc[0] for i,_ in enumerate(indices)]

    train_indices, valid_indices, train_labels, valid_labels = train_test_split(
    indices, labels, test_size=0.2, random_state=42
    )

    train_data_index = [indices.index(i) for i in (train_indices)]
    valid_data_index = [indices.index(i) for i in (valid_indices)]

    # print(indices)
    # print(train_indices)
    # print(valid_indices)
    # print(train_data_index)
    # print(valid_data_index)

    # skeleton_path와 train_indices를 이용
    # dataset/train/skeleton에 데이터 생성
    for i in train_indices:
        p = os.path.join(skeleton_path, str(i))
        # print(p)
        # new_path = '/home/khj/workspace/udjet/dataset/train/skeleton'
        new_path = '/home/khj/workspace/udjet/dataset/cnuh1_cnuh2_sejong1_gachon1_134_B_new/train/skeleton'
        os.makedirs(new_path, exist_ok=True)
        new_p = os.path.join(new_path, str(i))
        print(new_p)
        shutil.copytree(p, new_p)

    # skeleton_path와 valid_indices를 이용
    # dataset/valid/skeleton에 데이터 생성
    for i in valid_indices:
        p = os.path.join(skeleton_path, str(i))
        # print(p)
        # new_path = '/home/khj/workspace/udjet/dataset/valid/skeleton'
        new_path = '/home/khj/workspace/udjet/dataset/cnuh1_cnuh2_sejong1_gachon1_134_B_new/valid/skeleton'
        os.makedirs(new_path, exist_ok=True)
        new_p = os.path.join(new_path, str(i))
        # print(new_p)
        shutil.copytree(p, new_p)

    df = pd.read_excel(tabular_path)

    # dataset/train/tabular에 정형데이터 생성 
    df_train = df[df['number'].isin(train_indices)]
    # train_tabular_path = '/home/khj/workspace/udjet/dataset/train/tabular/data.xlsx'
    train_tabular_path = '/home/khj/workspace/udjet/dataset/cnuh1_cnuh2_sejong1_gachon1_134_B_new/train/tabular/data.xlsx'
    os.makedirs('/home/khj/workspace/udjet/dataset/cnuh1_cnuh2_sejong1_gachon1_134_B_new/train/tabular', exist_ok=True)
    df_train.to_excel(train_tabular_path, index=None)
    
    # dataset/valid/tabular에 정형데이터 생성 
    df_valid = df[df['number'].isin(valid_indices)]
    # valid_tabular_path = '/home/khj/workspace/udjet/dataset/valid/tabular/data.xlsx'
    valid_tabular_path = '/home/khj/workspace/udjet/dataset/cnuh1_cnuh2_sejong1_gachon1_134_B_new/valid/tabular/data.xlsx'
    os.makedirs('/home/khj/workspace/udjet/dataset/cnuh1_cnuh2_sejong1_gachon1_134_B_new/valid/tabular', exist_ok=True)
    df_valid.to_excel(valid_tabular_path, index=None)



