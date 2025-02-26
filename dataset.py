## transform : .csv -> dataframe -> torch.tensor
## call : 17 action dataset
## timestamp : no preprocessing

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import os
import numpy as np
import torch
import pandas as pd
import glob as glob
from datetime import datetime
from data_preprocess.angle_stand import stand
# from data_preprocess.angle import shoulder_flexion_left, shoulder_flexion_right, shoulder_extension_left, shoulder_extension_right
# from data_preprocess.number_tapping import stand_from_chair
# from data_preprocess.point_cal import hand_2_back_head_left, hand_2_back_head_right, hand_2_back_left, hand_2_back_right
from data_preprocess.walk import walk_8
from data_preprocess.preprocessing.correction import correction_skeleton, correction_skeleton_gpu

def calculate_age(birthdate):
    today = datetime.today()
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    return age

def custom_collate_fn(batch):
    # 배치 내의 데이터를 분리
    tabular_batch, skeleton_batch, label_batch = zip(*batch)

    max_actions = max([tensor.shape[0] for tensor in skeleton_batch])  # 최대 동작 수 
    max_frames = max([tensor.shape[1] for tensor in skeleton_batch])   # 최대 프레임 수 

    # 이중 패딩 적용 (동작 수 & 프레임 수)
    padded_skeleton_batch = []
    for tensor in skeleton_batch:
        pad_actions = max_actions - tensor.shape[0]
        pad_frames = max_frames - tensor.shape[1]
        
        # 동작 수와 프레임 수 모두 패딩 적용
        padded_tensor = torch.nn.functional.pad(
            tensor, 
            (0, 0, 0, 0, pad_frames, 0, pad_actions, 0)
        )
        padded_skeleton_batch.append(padded_tensor)

    # 다른 데이터는 그대로 스택
    tabular_batch = torch.stack([torch.tensor(item, dtype=torch.float32) for item in tabular_batch])
    label_batch = torch.tensor(label_batch, dtype=torch.long)
    
    return tabular_batch, padded_skeleton_batch, label_batch
    # return tabular_batch, skeleton_batch_final, label_batch

# 기본 tabular 데이터에 skeleton 기반 정량 데이터 추가
def tabular_reinforcement(time, skeleton_dataset):
    # time : 딕셔너리
    # {'동작_0' : [시간 리스트], '동작_1' : [시간 리스트], ....}
    # skeleton_dataset : 딕셔너리
    # {'동작_0' : [스켈레톤 np array], '동작_1' : [스켈레톤 np array], ....}

    # 추가할 정량 데이터 기반 데이터프레임 초기화
    a = pd.DataFrame()
    
    # 동작에 따라 a['컬럼명'] = 데이터 형태로 추가
    # 현재는 13개 데이터
    # 추후 정량데이터 협의 및 고도화를 통해 변경
    
    # 8자 보행 데이터만 추가
    try:
        t = time['Figure of 8 walk test']
        s = skeleton_dataset['Figure of 8 walk test']
        a['WALK'] = [walk_8(t,s)]
        # print('8자 보행 :', walk_8(t,s))
    except:
        a['WALK'] = 0

    return a


# 정형 데이터
# 1.이름 or 식별정보
# 2.생년월일 > 나이
# 3. 성별 > 0(남자) / 1(여자)
# 4. BMI 신설 > 몸무게(kg)/신장(m)^2
## ==> 정형 데이터를 통해 나이, 성별, 신장, 몸무게, BMI 이렇게 총 5개의 데이터

# 정형데이터와 스켈레톤 데이터는 index를 통해 매핑되어야 함

class TabularDataset():
    # def __init__(self, tabular_data_dir):
    def __init__(self, tabular_data_dir, skeleton_data_dir):
        self.tabular_data_dir = tabular_data_dir
        self.skeleton_data_dir = skeleton_data_dir

    def __len__(self):
        return len(pd.read_excel(self.tabular_data_dir))

    def __getitem__(self, index):     
        
        data = pd.read_excel(self.tabular_data_dir)
        
        mapping_id = int(os.listdir(self.skeleton_data_dir)[index])
        # print('mapping id :', mapping_id)

        data['생년월일'] = pd.to_datetime(data['생년월일'])
        data['나이'] = data['생년월일'].apply(calculate_age)

        gender = {
            '남' : 0,
            '여' : 1
        }

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

        data = data.drop(columns=['이름'])
        data['성별'] = data['성별'].map(gender)
        data['상태'] = data['상태'].map(status)
        data['BMI'] = round(data['몸무게'] / ((data['신장']/100)**2), 1)

        labels = data[['number','상태']]

        data = data.drop(columns=['생년월일', '상태'])

        # 그 외 17개 동작에서 총 13개 데이터를 계산 후 data 에 추가하기
        # ROM_1/2(+L/R), MFT_1/2(+L/R), SPPB_1/2/3/4, WALK 

        # return data[data['number'] == index], labels[labels['number'] == index]
        return data[data['number'] == mapping_id], labels[labels['number'] == mapping_id]


# 17개 동작에 대해 ST-GCN 기반 임베딩 준비
# 최종 return은 17개 동작에 대해 카메라 1/2번 중 하나를 선택해서 시계열 데이터셋으로 재정비
class SkeletonDataset():
    def __init__(self, skeleton_data_dir):
        self.skeleton_data_dir = skeleton_data_dir

    def __len__(self):
        # return 0
        return len(os.listdir(self.skeleton_data_dir))
    
    def __getitem__(self, index):

        mapping_id = os.listdir(self.skeleton_data_dir)[index]
        df_path = glob.glob(os.path.join(self.skeleton_data_dir, str(mapping_id), '*'))
        # df_path = glob.glob(os.path.join(self.skeleton_data_dir, str(index), '*'))
        # print(df_path)

        # total_skeleton = np.array([])
        total_skeleton = {}
        total_time = {}
        # print(f'df_path : {df_path}')

        for df_p in df_path:
            # 이름 아래 최대 17개 .csv 파일을 읽음
            # df = pd.read_csv(self.skeleton_data_dir, header=None)
            # print('skeleton_path :', df_p)
            # df = pd.read_csv(df_p, header=None)``
            # print(f'mapping_id: {mapping_id}')
            # print(f'df_path: {df_path}')
            # print(f'df_p: {df_p}')
            # df = pd.read_csv(df_p, header=None, skip_blank_lines=True)
            df = pd.read_csv(df_p, header=None)

            time = sorted(list(set(df[0])))

            # 2번 카메라 데이터를 사용
            # df = df[df[1] == 2] -> 원 데이터(0,1,2,3카메라 모두 포함) 시 사용 -> 이미 전처리 시 2번만 사용

            timestamp = sorted(list(set(df[0])))
            skeleton = []

            for t in timestamp:
                df_frame = df[df[0] == t]       # 32개 설정 -> df_frame = 1프레임 당 스켈레톤
                skeleton_frame = np.array([])
                for _, row in df_frame.iterrows():
                    skeleton_frame = np.append(skeleton_frame, np.array([row[3:6]]))
                
                # skeleton_frame_reshape = skeleton_frame.reshape(32,3)
                skeleton_frame_reshape = correction_skeleton(skeleton_frame.reshape(32,3))
                # skeleton_frame_reshape = correction_skeleton_gpu(skeleton_frame.reshape(32,3))
                # print('skeleton_frame_reshape :', skeleton_frame_reshape.shape)
                skeleton.append(skeleton_frame_reshape)
            
            key_name = (df_p.split('/')[-1]).split('.')[0]
            # total_skeleton[df_p.split('/')[-1]] = np.array(skeleton)
            # total_time[df_p.split('/')[-1]] = np.array(time)
            total_skeleton[key_name] = np.array(skeleton)
            total_time[key_name] = np.array(time)

        # 최종 return format
        # total_skeleton = {'앉아서....' : (frame수, 32, 3), ....}
        # total_skeleton의 총 key는 17개

        return total_time, total_skeleton
    # return 시 0번은 동작 별 timeline, 1번은 동작 별 skeleton[N, Point, Dim]



# tabular, skeleton에서 정형데이터/임베딩 준비
# skeletondataset에서 17개 데이터를 받아와서 데이터 분석 후 tabulardataset에 보강하기
# 5,6,7,8 > ROM
# 9,10,11,12 > MFT
# 13,14,15,16 > SPPB
# 17 > GAIT
# 18 > 상태(어노테이션) - 0:정상, 1:오십견, 2:회전근개, 3:퇴행성무릎, 4:척수손상, 5:파킨슨, 6:근감소증, 7:뇌졸중
class MultiModalDataset():
    def __init__(self, tabular_path, skeleton_path):
        # self.TabularDataset = TabularDataset.__init__(self, tabular_path, skeleton_path)
        # self.SkeletonDataset = SkeletonDataset.__init__(self, skeleton_path)
        # self.tabular_path = tabular_path
        # self.skeleton_path = skeleton_path
        self.tabular_dataset = TabularDataset(tabular_path, skeleton_path)
        # self.tabular_dataset = TabularDataset(tabular_path)
        self.skeleton_dataset = SkeletonDataset(skeleton_path)

        assert self.tabular_dataset.__len__() == self.skeleton_dataset.__len__(), "Data Unequal"

    def __len__(self):
        return min((self.tabular_dataset.__len__(), self.skeleton_dataset.__len__()))

    def __getitem__(self, index):
        # tabular_data, label = TabularDataset.__getitem__(index)
        # skeleton_data = SkeletonDataset.__getitem__(index)
        
        # tabular_data, label = self.tabular_dataset.__getitem__(index)
        tabular_data_original, label = self.tabular_dataset.__getitem__(index)
        skeleton_time = self.skeleton_dataset.__getitem__(index)[0]
        skeleton_data = self.skeleton_dataset.__getitem__(index)[1]
        # skeleton_time, skeleton_data = dict

        tabular_data_reinforcement = tabular_reinforcement(skeleton_time, skeleton_data)

        # skeleton 데이터 쌓기 
        # [(48,32,3), (64,32,3), (78,32,3), ...]
        skeleton_stack = []
        for k in list(skeleton_data.keys()):
            # skeleton_stack.append(skeleton_data[k])
            skeleton_stack.append(torch.tensor(skeleton_data[k]))
        
        padded_skeleton = pad_sequence(skeleton_stack, batch_first=True, padding_value=0)
        # print('padded_skeleton shape :', padded_skeleton.shape)
        
        # print('tabular_data_reinforcement')
        # print(tabular_data_reinforcement)

        # tabular_data = 현재 5개의 column을 가진 정형데이터
        # skeleton_data = 동작명 : [전체 프레임수, 포인트수, 3] 으로 구성된 딕셔너리 
        # number와 '상태' 를 라벨링으로 가진 정형데이터

        # skeleton_data를 기반으로 tabular_data를 보강
        # 기존 5개 column > 18개 column(13 = 어깨 최대각도, 평균 시간 등등..)

        tabular_data_original = tabular_data_original.reset_index(drop=True)
        tabular_data_reinforcement = tabular_data_reinforcement.reset_index(drop=True)

        # final_tabular_data = pd.concat([tabular_data_original, tabular_data_reinforcement], axis=1)
        
        final_tabular_data = pd.concat([tabular_data_original, tabular_data_reinforcement], axis=1).iloc[0].tolist()[1:]
        
        # print('final_tabular_data :', final_tabular_data)

        annotation = label.iloc[0].iloc[1]
        # print(f'annotation: {annotation}')

        # return tabular_data, skeleton_data, label
        # return final_tabular_data, skeleton_data, label
        # return final_tabular_data, padded_skeleton, label
        
        # return tabular_data_original, padded_skeleton, annotation
        return final_tabular_data, padded_skeleton, annotation