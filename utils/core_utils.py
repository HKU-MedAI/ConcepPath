import numpy as np
import torch
from utils.utils import *
import os
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F

class TOPDataset(Dataset):

    def __init__(self, data_fps, labels, transform):

        self.transform = transform
        self.data_fps = data_fps
        label_set = sorted(list(set(labels)))
        label_map = dict(zip(
            label_set, range(len(label_set))
        ))
        self.labels = [label_map[i] for i in labels]

    def __len__(self):
        return len(self.data_fps)

    def __getitem__(self, idx):
        
        with open(self.data_fps[idx], "rb") as f:
            data = pickle.load(f)["data"]
            
        label = self.labels[idx]
        return self.transform(data), label
    
    
def train_loop(epoch, model, loader, optimizer, n_classes, scheduler, loss_fn = None):   

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    
    train_loss = 0.

    print('\n')
    
    all_probs = []
    all_labels = []
    
    for batch_idx, (data, label) in enumerate(loader):
        
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, loss = model(data, label)
        
        probs = Y_prob.detach()
        labels = label.item()
        losses = loss.item()
        
        all_probs.append(probs)
        all_labels.append(label)
        train_loss += losses
        
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, losses, labels, data.size(0)))

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate metrics for epoch
    train_loss /= len(loader)

    y_true = torch.concat(all_labels, dim=0)
    y_pred = torch.concat(all_probs, dim=0)
    
    acc, correct_counts, total_counts, micro_f1, macro_f1, micro_auc, macro_auc, avg_sensitivity, avg_specificity = evaluate_metrics(y_true=y_true, y_pred=y_pred, num_classes=n_classes)
    
    scheduler.step()

    correct_info_list = [f"Class {i}: acc {correct_counts[i]/total_counts[i]}, {correct_counts[i]}/{total_counts[i]}"for i in range(len(correct_counts))]
    
    print('\nEpoch: {}, train_loss: {:.4f}, train_acc: {:.4f}'.format(epoch, train_loss, acc))
    for correct_info in correct_info_list:
        print(correct_info)
        
    return train_loss, acc, micro_f1, macro_f1, micro_auc, macro_auc, avg_sensitivity, avg_specificity

def test(model, loader, n_classes, test_name_list, attn_score_fp, vlm_model, test=False):
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_probs = []
    all_labels = []

    patient_results = {}
    
    for batch_idx, (data, label) in enumerate(loader):
        
        test_name = test_name_list[batch_idx].split("/")[-1].replace(".pkl","")
        
        # print(test_name, "\t",  label)
        
        result_fp = os.path.join(attn_score_fp, vlm_model)
        if test:
            if not os.path.exists(result_fp):
                os.makedirs(result_fp)
        result_fp = os.path.join(result_fp, f"{test_name}.pkl")
        
        data, label = data.to(device), label.to(device)
        
        with torch.no_grad():
            logits, Y_prob, Y_hat, loss = model(data, label, result_fp, test)

        probs = Y_prob.detach()
        labels = label.item()
        losses = loss.item()
        
        all_probs.append(probs)
        all_labels.append(label)
        test_loss += losses
        
        patient_results[test_name] = f"is {Y_hat.item()==labels}, pred: {Y_hat.item()}, label: {labels}"
    
    test_loss /= len(loader)
    
    y_true = torch.concat(all_labels, dim=0)
    y_pred = torch.concat(all_probs, dim=0)
    
    acc, correct_counts, total_counts, micro_f1, macro_f1, micro_auc, macro_auc, avg_sensitivity, avg_specificity = evaluate_metrics(y_true=y_true, y_pred=y_pred, num_classes=n_classes)
    
    correct_info_list = [f"Class {i}: acc {correct_counts[i]/total_counts[i]}, correct {correct_counts[i]}/{total_counts[i]}"for i in range(len(correct_counts))]
    for correct_info in correct_info_list:
        print(correct_info)

    return test_loss, acc, micro_f1, macro_f1, micro_auc, macro_auc, avg_sensitivity, avg_specificity, patient_results


def validate(epoch, model, loader, n_classes, early_stopping = None, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    val_loss = 0.
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, loss = model(data, label)
            
            loss = loss_fn(logits, label)

            probs = Y_prob.detach()
            labels = label.item()
            losses = loss.item()
            
            all_probs.append(probs)
            all_labels.append(label)
            val_loss += losses
            
    val_loss /= len(loader)
    
    y_true = torch.concat(all_labels, dim=0)
    y_pred = torch.concat(all_probs, dim=0)
    
    acc, correct_counts, total_counts, micro_f1, macro_f1, micro_auc, macro_auc, avg_sensitivity, avg_specificity = evaluate_metrics(y_true=y_true, y_pred=y_pred, num_classes=n_classes)

    print('\nVal Set, val_loss: {:.4f}, val_acc: {:.4f}'.format(val_loss, acc))
    
    correct_info_list = [f"Class {i}: acc {correct_counts[i]/total_counts[i]}, {correct_counts[i]}/{total_counts[i]}"for i in range(len(correct_counts))]
    for correct_info in correct_info_list:
        print(correct_info)
        
    if early_stopping:
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(os.path.join(f"best_model.pt")))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True, val_loss, acc, micro_f1, macro_f1, micro_auc, macro_auc, avg_sensitivity, avg_specificity

    return False, val_loss, acc, micro_f1, macro_f1, micro_auc, macro_auc, avg_sensitivity, avg_specificity

def evaluate_metrics(y_true, y_pred, num_classes):
    
    y_true_np = y_true.cpu().numpy()
    y_pred_np = torch.argmax(y_pred, dim=1).cpu().numpy()

    
    acc = (y_true_np == y_pred_np).mean()
    cm = confusion_matrix(y_true_np, y_pred_np)
    
    correct_counts = np.diag(cm)
    total_counts = np.sum(cm, axis=1)

    micro_f1 = f1_score(y_true_np, y_pred_np, average='micro')
    macro_f1 = f1_score(y_true_np, y_pred_np, average='macro')

    y_onehot = y_true.cpu().numpy()
    y_pred_prob = y_pred.cpu().numpy()
    
    if num_classes>2:
        micro_auc = roc_auc_score(y_onehot, y_pred_prob, average='micro', multi_class="ovr")
        macro_auc = roc_auc_score(y_onehot, y_pred_prob, average='macro', multi_class="ovr")
    else:
        micro_auc = roc_auc_score(y_onehot, y_pred_prob[:,1], average='micro')
        macro_auc = micro_auc

    
    cm = confusion_matrix(y_true_np, y_pred_np)
    sensitivity = np.diag(cm) / np.sum(cm, axis=1)
    specificity = (np.sum(cm) - np.sum(cm, axis=0) - np.sum(cm, axis=1) + np.diag(cm)) / (np.sum(cm) - np.sum(cm, axis=0))

    avg_sensitivity = np.mean(sensitivity)
    avg_specificity = np.mean(specificity)

    return acc, correct_counts, total_counts, micro_f1, macro_f1, micro_auc, macro_auc, avg_sensitivity, avg_specificity

def update_best_metrics(model_id, metrics, csv_path):

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    if 'id' in df.columns and model_id in df['id'].values:
        idx = df[df['id'] == model_id].index[0]
        for key, value in metrics.items():
            df.at[idx, key] = value
    else:
        metrics['id'] = model_id
        df = df._append(metrics, ignore_index=True)

    df.to_csv(csv_path, index=False)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss