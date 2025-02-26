# Multimodal Classification Model

### Modality
1. **Tab-Net with tabular data (from scratch)**
2. **ST-GCN with 3D serial skeleton data (from scratch)**
   > https://arxiv.org/abs/1801.07455

**************

### Embedding Fusion
Joint Embedding : **Attention-based fusion**

**************

### Data Preprocessing
1. Direction normalization
2. Location normalization

**************

### Data Augmentation
1. Mirroring
2. Reversing Time
3. Adding noise in skeleton
4. Dropping random frames 

**************

### Hyperparameter Tuning
Using Optuna for hyperparameters(L2 norm, lr, optimizer, dropout in ST-GCN)
