import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import copy
import random
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, matthews_corrcoef
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import ast

def hierarchy_filter(df_in, rank='kingdom', min_seq=5, wildcard_seed=False,
                     wildcard_list=None, wildcard_name=None, r=0.1):
    df = copy.deepcopy(df_in)
    original_ranks = ['species', 'genus', 'family', 'order',
                      'class', 'phylum', 'kingdom', 'domain']
    retain_columns = [rank, 'target', 'smiles','fingerprint']
    df = df[retain_columns]

    dedup_dfs = []
    for cls in df[rank].unique():
        class_df = df[df[rank] == cls]
        dedup_df = class_df.drop_duplicates(subset='target', keep='first')
        dedup_dfs.append(dedup_df)
    df = pd.concat(dedup_dfs)

    class_counts = df[rank].value_counts()
    valid_class_list = class_counts[class_counts >= min_seq].index.tolist()
    df = df[df[rank].isin(valid_class_list)].reset_index(drop=True)


    sorted_class_list = sorted(df[rank].unique())
    class_class_converter = {cls: i for i, cls in enumerate(sorted_class_list)}
    df['label'] = df[rank].map(class_class_converter)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state= 76)   #This is my girlfriend's birthday, you can change it to any number you like.
    for train_idx, val_idx in sss.split(df.target, df.label):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

    return (
        train_df['target'].tolist(),
        val_df['target'].tolist(),
        train_df['label'].tolist(),
        val_df['label'].tolist(),
        sorted_class_list,
        class_class_converter
    )

class FeatureGenerator:
    def __init__(self):
        self.smiles_cache = {}  

    def generate(self, target_list, df_ref):
        features = []
        valid_indices = []
        target_to_fp = df_ref.set_index('target')['fingerprint'].to_dict()
        for i, target in enumerate(target_list):
            fp_val = target_to_fp.get(target, None)
            if fp_val is None:
                continue
            try:
                fp = np.array(ast.literal_eval(fp_val), dtype=int)
            except Exception as e:
                print(f"转换指纹时出错，target: {target}, error: {e}")
                continue
            if fp is not None and len(fp) == 2072:
                features.append(fp)
                valid_indices.append(i)
        return np.array(features), valid_indices

class GlycanDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class GlycClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.7),

            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(0.6),

            nn.Linear(768, num_classes)

        )

    def forward(self, x):
        return self.layers(x)


class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.best_acc = 0.0
        self.history = {'train_loss': [], 'val_loss': []}

    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_mcc = matthews_corrcoef(all_labels, all_preds)
        self.history['train_loss'].append(epoch_loss)
        return epoch_loss, epoch_acc, epoch_mcc

    def validate(self, loader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_mcc = matthews_corrcoef(all_labels, all_preds)
        self.history['val_loss'].append(epoch_loss)
        return epoch_loss, epoch_acc, epoch_mcc

    def save_checkpoint(self, path):
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history
        }, path)



config = {
    'rank': 'kingdom',
    'min_seq': 5,
    'batch_size': 32,
    'epochs': 100,
}
df = (pd.read_csv('../data/MPM-fingerprint.csv'))
train_x, val_x, train_y, val_y, class_list, class_converter = hierarchy_filter(
    df, rank=config['rank'], min_seq=config['min_seq']
)


fgen = FeatureGenerator()
train_features, train_valid_idx = fgen.generate(train_x, df)
train_y = [train_y[i] for i in train_valid_idx]

val_features, val_valid_idx = fgen.generate(val_x, df)
val_y = [val_y[i] for i in val_valid_idx]


train_dataset = GlycanDataset(train_features, train_y)
val_dataset = GlycanDataset(val_features, val_y)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GlycClassifier(2072, len(class_list)).to(device)
# criterion = nn.CrossEntropyLoss().to(device)
#
# optimizer_ft = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer_ft,
#     mode='max',
#     factor=0.5,
#     patience=5,
#     verbose=True
# )
# best_acc = 0
# trainer = Trainer(model, criterion, optimizer_ft)
# for epoch in range(config['epochs']):
#     train_loss, train_acc, train_mcc = trainer.train_epoch(train_loader)
#     val_loss, val_acc, val_mcc = trainer.validate(val_loader)
#     scheduler.step(val_loss)
#     print(f"Epoch {epoch + 1}/{config['epochs']}")
#     print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | MCC: {train_mcc:.4f}")
#     print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | MCC: {val_mcc:.4f}")
#
#     if val_acc > trainer.best_acc:
#         trainer.best_acc = val_acc
#         # trainer.save_checkpoint(f'glyclass_test/best_model_{i}.pth')
#         print(f"Saved new best model with accuracy {val_acc:.4f}")
# print(trainer.best_acc)
model_fp = GlycClassifier(input_size=2072, num_classes=len(class_list))
checkpoint = torch.load(f'../weights/FP-Net-kingdom.pth', map_location=device)
model_fp.load_state_dict(checkpoint['model_state'])
model_fp = model_fp.to(device)

df_full = pd.read_csv('../data/MPM-fingerprint.csv')
all_targets = df_full['target'].tolist()
all_kingdoms = df_full['kingdom'].tolist()

fgen = FeatureGenerator()
all_features, valid_indices = fgen.generate(all_targets, df_full)
all_targets = [all_targets[i] for i in valid_indices]
all_kingdoms = [all_kingdoms[i] for i in valid_indices]

glycan_loader = DataLoader(torch.FloatTensor(all_features), batch_size=32, shuffle=False)
model_fp.eval()
res = []
with torch.no_grad():
    for inputs in glycan_loader:
        inputs = inputs.to(device)
        intermediate_model = nn.Sequential(*list(model_fp.layers.children())[:7])  
        out = intermediate_model(inputs)
        res.append(out.cpu())

res2 = [res[k].detach().numpy() for k in range(len(res))]
res2 = pd.DataFrame(np.concatenate(res2))

from sklearn.manifold import TSNE

tsne_emb = TSNE(random_state=42).fit_transform(res2)
plt.figure(figsize=(9, 9))
import seaborn as sns
ax = sns.scatterplot(x=tsne_emb[:, 0], y=tsne_emb[:, 1], s=40, alpha=0.4,
                hue=all_kingdoms, palette='colorblind', rasterized=True)
# sns.scatterplot(x=tsne_emb[:, 0], y=tsne_emb[:, 1], s=50, alpha=0.8, edgecolor="w", linewidth=0.3,
#                 hue=taxonomic_glycans.kingdom.values.tolist(), palette='colorblind', rasterized=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('t-SNE Dim1')
plt.ylabel('t-SNE Dim2')
plt.title(f'FP-Net')
plt.tight_layout()
# plt.savefig("sine_wave.svg", format="svg")  
plt.savefig("FP-Net.svg", format="svg", bbox_inches='tight')
plt.show()


