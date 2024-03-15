import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def evaluate(model, eval_dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    for step, batch in enumerate(eval_dataloader):
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

    eval_metric = {'acc':acc,'recall':recall,'prec':p,'f1':f1}

    return eval_metric