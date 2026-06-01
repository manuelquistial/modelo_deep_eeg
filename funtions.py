import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np


class LocalFeatureLearner(nn.Module):
    def __init__(self, in_channels=1, eeg_channels=8, F1=7, dropout=0.25):
        super().__init__()

        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(in_channels, F1, kernel_size=(1, 10), padding=(0, 5)),
            nn.BatchNorm2d(F1),
            nn.ELU()
        )

        # Spatial convolution
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F1, kernel_size=(eeg_channels, 1)),
            nn.BatchNorm2d(F1),
            nn.ELU()
        )

        # Pooling
        self.pool = nn.AvgPool2d(kernel_size=(1, 10))

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x  # (B, F1, 1, T/k)


class EEGMeModel(nn.Module):
    def __init__(self, C=8, T=1000, f=7, e=32, num_classes=2):
        super().__init__()

        self.T = T

        # Local
        self.local = LocalFeatureLearner(
            in_channels=1,
            eeg_channels=C,
            F1=f
        )

        # Embedding (Conv2D projection)
        self.proj = nn.Conv2d(f, e, kernel_size=(1,1))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=e,
            nhead=4,
            dropout=0.25,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # Embedding dimension final
        # self.embedding_dim = e * (T // 10)

        # Clasificador
        # self.fc = nn.Linear(self.embedding_dim, num_classes)

        self.embedding_dim = e
        self.fc = nn.Linear(e, num_classes)

    def forward(self, x):

        # ================= LOCAL FEATURES =================
        local_features = self.local(x)      # (B,f,1,T/k)

        # ================= PROJECTION =================
        x = self.proj(local_features)       # (B,e,1,T/k)

        x = x.squeeze(2)                    # (B,e,T/k)

        x = x.permute(0,2,1)                # (B,T/k,e)

        # ================= GLOBAL FEATURES =================
        global_features = self.transformer(x)   # (B,T/k,e)

        x = global_features

        # ================= POOLING =================
        x = torch.mean(x, dim=1)

        # ================= EMBEDDINGS =================
        embeddings = F.normalize(x, p=2, dim=1)

        # ================= CLASSIFICATION =================
        logits = self.fc(embeddings)

        return logits, embeddings, local_features, global_features


def compute_pairwise_distances(embeddings):
    return torch.cdist(embeddings, embeddings, p=2)


def semi_hard_triplet_mining(embeddings, labels, margin=1.0):
    dist_matrix = compute_pairwise_distances(embeddings)

    anchors, positives, negatives = [], [], []

    for i in range(len(labels)):
        anchor = embeddings[i]
        label = labels[i]

        pos_mask = (labels == label)
        neg_mask = (labels != label)

        pos_indices = pos_mask.nonzero(as_tuple=True)[0]
        neg_indices = neg_mask.nonzero(as_tuple=True)[0]

        for p_idx in pos_indices:
            if p_idx == i:
                continue

            d_ap = dist_matrix[i, p_idx]

            # semi-hard negatives
            valid_neg = []
            for n_idx in neg_indices:
                d_an = dist_matrix[i, n_idx]

                if d_ap < d_an < d_ap + margin:
                    valid_neg.append(n_idx)

            if len(valid_neg) > 0:
                n_idx = valid_neg[0]

                anchors.append(anchor)
                positives.append(embeddings[p_idx])
                negatives.append(embeddings[n_idx])

    if len(anchors) == 0:
        return None, None, None

    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

'''
def train_step(model, x, labels, optimizer, margin=1.0, lambda_tri=0.5):
    model.train()

    logits, embeddings = model(x)

    # Cross Entropy
    ce_loss = nn.CrossEntropyLoss()(logits, labels)

    # Triplet Loss
    A, P, N = semi_hard_triplet_mining(embeddings, labels, margin)

    if A is not None:
        triplet_fn = nn.TripletMarginLoss(margin=margin)
        tri_loss = triplet_fn(A, P, N)
    else:
        tri_loss = torch.tensor(0.0, device=x.device)

    # Loss total
    loss = ce_loss + lambda_tri * tri_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), ce_loss.item(), tri_loss.item()
'''


def loso_split(data, labels, subjects, test_subject):

    train_idx = [i for i, s in enumerate(subjects) if s != test_subject]
    test_idx  = [i for i, s in enumerate(subjects) if s == test_subject]

    X_train = data[train_idx]
    y_train = labels[train_idx]
    s_train = subjects[train_idx]

    X_test = data[test_idx]
    y_test = labels[test_idx]
    s_test = subjects[test_idx]

    return (X_train, y_train, s_train), (X_test, y_test, s_test)


def get_loader(dataset, batch_size=32):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y, _ in loader:
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")

def segment_signal(eeg, window_size, step):
    """
    eeg: (C, T)
    """
    segments = []

    for i in range(0, eeg.shape[1] - window_size, step):
        seg = eeg[:, i:i+window_size]
        segments.append(seg)

    return np.array(segments)  # (N_segments, C, window_size)


class EEGDataset(Dataset):
    def __init__(self, data, labels, subjects):
        self.data = data
        self.labels = labels
        self.subjects = subjects

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]  # (C,T)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1,C,T)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        s = self.subjects[idx]

        return x, y, s
    
'''
def train_loso(model_class, data, labels, subjects, num_epochs=10):

    unique_subjects = np.unique(subjects)

    for test_subject in unique_subjects:

        print(f"\n Test subject: {test_subject}")

        # SPLIT
        (X_train, y_train, s_train), (X_test, y_test, s_test) = loso_split(data, labels, subjects, test_subject)

        train_dataset = EEGDataset(X_train, y_train, s_train)
        test_dataset  = EEGDataset(X_test, y_test, s_test)

        train_loader = get_loader(train_dataset)
        test_loader  = get_loader(test_dataset)

        # NUEVO MODELO
        model = model_class()

        optimizer = optim.Adam(
            model.parameters(),
            lr=1e-3,
            betas=(0.5, 0.999)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=20
        )

        best_val_loss = float('inf')
        patience_counter = 0
        early_patience = 50

        # TRAIN
        for epoch in range(100):

            model.train()
            total_loss = 0

            for x, y, _ in train_loader:

                logits, embeddings = model(x)
               
                ce_loss = nn.CrossEntropyLoss()(logits, y)

                A, P, N = semi_hard_triplet_mining(embeddings, y)

                if A is not None:
                    tri_loss = nn.TripletMarginLoss(margin=1.0)(A, P, N)
                else:
                    tri_loss = torch.tensor(0.0)

                loss = ce_loss
                # loss = ce_loss + 0.5 * tri_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # VALIDACIÓN (temporal con train_loader)
            val_loss = 0
            model.eval()

            with torch.no_grad():
                for x, y, _ in train_loader:
                    logits, _ = model(x)
                    ce_loss = nn.CrossEntropyLoss()(logits, y)
                    val_loss += ce_loss.item()

            print(f"Epoch {epoch} | Train: {total_loss:.4f} | Val: {val_loss:.4f}")

            # scheduler
            scheduler.step(val_loss)

            # early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_patience:
                print("Early stopping")
                break

        # TEST (FUERA del loop de epochs)
        evaluate(model, test_loader)
'''

from collections import defaultdict, Counter
import torch


def evaluate_subject_level(model, loader):

    model.eval()

    # guardar predicciones por sujeto
    subject_preds = defaultdict(list)
    subject_labels = {}

    with torch.no_grad():

        for x, y, s in loader:

            logits, _, _, _ = model(x)

            preds = torch.argmax(logits, dim=1)

            # recorrer batch
            for pred, label, subject in zip(preds, y, s):

                subject = subject.item()

                subject_preds[subject].append(pred.item())

                # guardar label real del sujeto
                subject_labels[subject] = label.item()

    # ================= MAJORITY VOTING =================
    correct = 0
    total = 0

    for subject in subject_preds:

        preds = subject_preds[subject]

        # voto mayoritario
        final_pred = Counter(preds).most_common(1)[0][0]

        true_label = subject_labels[subject]

        print(f"Sujeto {subject} | Pred final: {final_pred} | Real: {true_label}")

        if final_pred == true_label:
            correct += 1

        total += 1

    acc = correct / total

    print(f"\nSubject-Level Accuracy: {acc:.4f}")

    return acc

def train_loso(model_class, data, labels, subjects, num_epochs=100):

    #unique_subjects = np.unique(subjects)[:1]
    unique_subjects = np.unique(subjects)
    all_accs = []

    for test_subject in unique_subjects:

        print(f"\n Test subject: {test_subject}")

        # SPLIT LOSO
        (X_train, y_train, s_train), (X_test, y_test, s_test) = \
            loso_split(data, labels, subjects, test_subject)

        # split train/val (IMPORTANTE)
        split = int(0.8 * len(X_train))

        X_val = X_train[split:]
        y_val = y_train[split:]
        s_val = s_train[split:]

        X_train = X_train[:split]
        y_train = y_train[:split]
        s_train = s_train[:split]

        # datasets
        train_dataset = EEGDataset(X_train, y_train, s_train)
        val_dataset   = EEGDataset(X_val, y_val, s_val)
        test_dataset  = EEGDataset(X_test, y_test, s_test)

        # loaders
        train_loader = get_loader(train_dataset)
        val_loader   = get_loader(val_dataset)
        test_loader  = get_loader(test_dataset)

        # modelo
        model = model_class()

        optimizer = optim.Adam(model.parameters(), lr=1e-3)   #algoritmo que actualiza los pesos
        
        #reducir automáticamente el learning rate
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10 )

        best_val_loss = float('inf')
        patience_counter = 0
        early_patience = 20

        # ================= TRAIN =================
        for epoch in range(num_epochs):

            model.train()
            total_loss = 0

            for x, y, _ in train_loader:

                logits, embeddings, local_features, global_features = model(x)

                ce_loss = nn.CrossEntropyLoss()(logits, y)

                # activar triplet después (opcional)
                use_triplet = False

                if use_triplet:
                    A, P, N = semi_hard_triplet_mining(embeddings, y)

                    if A is not None:
                        tri_loss = nn.TripletMarginLoss(margin=1.0)(A, P, N)
                    else:
                        tri_loss = torch.tensor(0.0)

                    loss = ce_loss + 0.1 * tri_loss
                else:
                    loss = ce_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # ================= VALIDATION =================
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for x, y, _ in val_loader:
                    logits, _, local_features, global_features  = model(x)
                    ce_loss = nn.CrossEntropyLoss()(logits, y)
                    val_loss += ce_loss.item()

            train_loss = total_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
            
            scheduler.step(val_loss)

            # early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_patience:
                print("Early stopping")
                break

        
        # ================= TEST =================
        
        temporal_weights = model.local.temporal_conv[0].weight.data
        np.save(f"temporal_weights_subject_{test_subject}.npy",
        temporal_weights.cpu().numpy())
        # guardar embeddings
        all_embeddings = []
        all_labels_emb = []
        all_subjects_emb = []
        all_preds_emb = []
        all_local_features = []
        all_global_features = []
        model.eval()

        with torch.no_grad():

            for x, y, s in test_loader:

                logits, embeddings, local_features, global_features = model(x)

                preds = torch.argmax(logits, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())
                all_labels_emb.append(y.cpu().numpy())
                all_subjects_emb.append(s.cpu().numpy())
                all_preds_emb.append(preds.cpu().numpy())
                all_local_features.append(local_features.cpu().numpy())
                all_global_features.append(global_features.cpu().numpy())

        # concatenar
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels_emb = np.concatenate(all_labels_emb, axis=0)
        all_subjects_emb = np.concatenate(all_subjects_emb, axis=0)
        all_preds_emb = np.concatenate(all_preds_emb, axis=0)
        all_local_features = np.concatenate(all_local_features,axis=0)
        all_global_features = np.concatenate(all_global_features,axis=0)


        # guardar
        np.save(f"embeddings_subject_{test_subject}.npy", all_embeddings)
        np.save(f"labels_subject_{test_subject}.npy", all_labels_emb)
        np.save(f"subjects_subject_{test_subject}.npy", all_subjects_emb)
        np.save(f"preds_subject_{test_subject}.npy", all_preds_emb)
        np.save(f"all_local_features_{test_subject}.npy", all_local_features)
        np.save(f"all_global_features_{test_subject}.npy", all_global_features)
        # evaluar
        acc = evaluate_subject_level(model, test_loader)

        all_accs.append(acc)

    # ================= RESULTADOS FINALES =================
    print("\n==== RESULTADOS FINALES ====")
    print("Accuracies:", all_accs)
    print("Mean:", np.mean(all_accs))
    print("Std:", np.std(all_accs))

    



