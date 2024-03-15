import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset_log_AD import multi_task_dataset_AD,CustomDataCollator_AD



def log_AD(tokenizer, model, device, log_file, max_length,
                        dataset_name="BGL"):
    model.to(device)
    model.eval()
    data_collator = CustomDataCollator_AD(tokenizer, pad_to_multiple_of=None)
    logdata_test = multi_task_dataset_AD(log_file, [dataset_name], tokenizer, train=False)
    test_dataloader = DataLoader(logdata_test, shuffle=False,collate_fn=data_collator, batch_size=20)

    y_true = []
    y_pred = []
    for step, batch in enumerate(test_dataloader):
        input_ids = batch['input_ids'].to(device)
        atten_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].squeeze(1).detach().cpu().numpy()
        y_true.extend(labels.tolist())
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=atten_mask)
            predictions = outputs.logits.argmax(dim=-1)
            pre = predictions.detach().cpu().numpy().tolist()
            y_pred.extend(pre)

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    p = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    eval_metric = {'acc': acc, 'recall': recall, 'prec': p, 'f1': f1}
    return eval_metric

def write_pd():
    pass