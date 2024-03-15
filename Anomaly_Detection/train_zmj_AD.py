import logging
import math
import os
import random
import datasets
import torch
import numpy as np
import transformers
from datasets import load_metric
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from transformers import (
    HfArgumentParser,
    AutoConfig,
    default_data_collator,
    get_scheduler,
    set_seed,
    RobertaForTokenClassification,
    RobertaForSequenceClassification,
    AutoTokenizer,

)
transformers.logging.set_verbosity_error()
from dataset_log_AD import multi_task_dataset_AD,CustomDataCollator_AD,multi_task_dataset_buffer_AD
from eval_AD import evaluate
from Log_anomaly_detection import log_AD
device = 'cuda:0'
random.seed(2023)
torch.cuda.manual_seed(2023)
np.random.seed(2023)


ori_data_path = 'dataset_cl/data_AD'


def fine_tuning_woCL(args):
    files_SmallParse = args.permutation
    task_output_dir = args.save+'/fine_tuning_woCL'
    os.makedirs(task_output_dir, exist_ok=True)

    model_path = "roberta-base"
    all_metric = {}

    for idx,data_name in enumerate(files_SmallParse):
        print(f'idx={idx} data_name={data_name}')

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True, do_lower_case=False)
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        tokenizer.model_max_length = model.config.max_position_embeddings - 2
        data_collator = CustomDataCollator_AD(tokenizer, pad_to_multiple_of=None)
        logdata_train = multi_task_dataset_AD(ori_data_path, [data_name], tokenizer)
        train_dataloader = DataLoader(logdata_train, shuffle=True,collate_fn=data_collator, batch_size=20)
        logdata_eval = multi_task_dataset_AD(ori_data_path, [data_name], tokenizer,train_eval_test='train')
        eval_dataloader = DataLoader(logdata_eval, shuffle=False, collate_fn=data_collator,batch_size=20)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=5e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        num_train_epochs = 5
        lr_scheduler = get_scheduler(
            name='polynomial',
            optimizer=optimizer,
            num_warmup_steps=num_train_epochs * len(train_dataloader)//10,
            num_training_steps=num_train_epochs * len(train_dataloader),
        )

        max_train_steps = 3200
        completed_steps = 0
        for epoch in range(num_train_epochs):
            model.train()
            total_loss = []
            for step, batch in enumerate(train_dataloader):
                # input_ids = batch[0].to(device)
                # atten_mask = batch[1].to(device)
                # labels = batch[2].squeeze(1).to(device)
                # outputs = model(input_ids=input_ids,attention_mask=atten_mask,labels=labels)
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

                loss = outputs.loss
                # print(step,' ',loss)
                total_loss.append(float(loss))
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
                # if completed_steps % 5 == 0:
                #     best_metric = evaluate(model,  eval_dataloader, device)
                #     print("Finish training, best metric: ")
                #     print(best_metric)
            print('epoch = ', epoch, '   loss = ', sum(total_loss) / len(total_loss))

            if epoch % 1 == 0:
                best_metric = evaluate(model,  eval_dataloader, device)
                print("Finish training, best metric: ")
                print(best_metric)


        log_file = f'data_AD'
        metric = log_AD(tokenizer, model, device, log_file, max_length=256,dataset_name=data_name)
        print(f'data_name:{data_name}')
        print(metric)
        all_metric[data_name]=metric
    # sorted_dict_by_key = dict(sorted(all_metric.items()))
    # df = pd.DataFrame(sorted_dict_by_key)
    # df.to_csv(f'{task_output_dir}\\metrics.csv')



def train_MultiTask(args):
    files_SmallParse = args.permutation
    task_output_dir = args.save+'/multi-task'
    os.makedirs(task_output_dir, exist_ok=True)

    model_path = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True, do_lower_case=False)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    tokenizer.model_max_length = model.config.max_position_embeddings - 2

    data_collator = CustomDataCollator_AD(tokenizer, pad_to_multiple_of=None)
    logdata_train = multi_task_dataset_AD(ori_data_path, files_SmallParse, tokenizer)
    train_dataloader = DataLoader(logdata_train, shuffle=True, collate_fn=data_collator, batch_size=20)
    logdata_eval = multi_task_dataset_AD(ori_data_path, files_SmallParse, tokenizer, train_eval_test='eval')
    eval_dataloader = DataLoader(logdata_eval, shuffle=False, collate_fn=data_collator, batch_size=20)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=5e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    num_train_epochs = 2
    lr_scheduler = get_scheduler(
        name='polynomial',
        optimizer=optimizer,
        num_warmup_steps=num_train_epochs * len(train_dataloader)//10,
        num_training_steps=num_train_epochs * len(train_dataloader),
    )

    max_train_steps = 3200
    completed_steps=0
    for epoch in range(num_train_epochs):
        model.train()
        total_loss = []
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss.append(float(loss))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1


        print('epoch = ', epoch, '   loss = ', sum(total_loss) / len(total_loss))

        if epoch % 1 == 0:
            best_metric = evaluate(model,  eval_dataloader, device)
            print("Finish training, best metric: ")
            print(best_metric)


    all_metric = {}
    for file in files_SmallParse:
        log_file = f'data_AD'
        metric = log_AD(tokenizer, model, device, log_file, max_length=256, dataset_name=file)
        print(file)
        print(metric)
        all_metric[file] = metric
    sorted_dict_by_key = dict(sorted(all_metric.items()))
    df = pd.DataFrame(sorted_dict_by_key)
    df.to_csv(f'{task_output_dir}\\metrics.csv')


def sequential_fine_tuning(args):
    files_SmallParse = args.permutation
    task_output_dir = args.save+'/sequential_fine-tuning'
    os.makedirs(task_output_dir, exist_ok=True)
    with open(task_output_dir+'\order.txt','w') as files:
        files.write(str(files_SmallParse))
    # ori_data_path = 'D:\ZMJ\pythonProject\Log_continual_learning\datasets'
    model_path = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True, do_lower_case=False)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    tokenizer.model_max_length = model.config.max_position_embeddings - 2


    data_collator = CustomDataCollator_AD(
        tokenizer, pad_to_multiple_of=None
    )
    for idx,data_name in enumerate(files_SmallParse):
        print(f'idx={idx} data_name={data_name}')
        logdata_train = multi_task_dataset_AD(ori_data_path, files_SmallParse, tokenizer)
        train_dataloader = DataLoader(logdata_train, shuffle=True, collate_fn=data_collator, batch_size=20)
        logdata_eval = multi_task_dataset_AD(ori_data_path, files_SmallParse, tokenizer, train_eval_test='eval')
        eval_dataloader = DataLoader(logdata_eval, shuffle=False, collate_fn=data_collator, batch_size=20)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=5e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        num_train_epochs = 5
        lr_scheduler = get_scheduler(
            name='polynomial',
            optimizer=optimizer,
            num_warmup_steps=num_train_epochs * len(train_dataloader)//10,
            num_training_steps=num_train_epochs * len(train_dataloader),
        )

        max_train_steps = 3200
        completed_steps = 0
        for epoch in range(num_train_epochs):
            model.train()
            total_loss = []
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss.append(float(loss))
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

                # if completed_steps >= max_train_steps:
                #     break
            print('epoch = ', epoch, '   loss = ', sum(total_loss) / len(total_loss))

            if epoch % 1 == 0:
                best_metric = evaluate(model,  eval_dataloader, device)
                print("Finish training, best metric: ")
                print(best_metric)

        sub_task_output_dir = task_output_dir+f'/Num_{idx}_{data_name}'
        os.makedirs(sub_task_output_dir, exist_ok=True)
        all_metric = {}
        for file in files_SmallParse:
            log_file = f'data_AD'
            metric = log_AD(tokenizer, model, device, log_file, max_length=256, dataset_name=file)
            print(metric)
            all_metric[file] = metric
        sorted_dict_by_key = dict(sorted(all_metric.items()))
        df = pd.DataFrame(sorted_dict_by_key)
        df.to_csv(f'{sub_task_output_dir}\\metrics.csv')

def zero_shot(args):
    files_SmallParse = args.permutation
    model_path = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True, do_lower_case=False)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    tokenizer.model_max_length = model.config.max_position_embeddings - 2

    task_output_dir = args.save+'/zero-shot'
    os.makedirs(task_output_dir, exist_ok=True)
    all_metric = {}
    for file in files_SmallParse:
        log_file = f'data_AD'
        metric = log_AD(tokenizer, model, device, log_file, max_length=256, dataset_name=file)
        print(metric)
        all_metric[file] = metric
    sorted_dict_by_key = dict(sorted(all_metric.items()))
    df = pd.DataFrame(sorted_dict_by_key)
    df.to_csv(f'{task_output_dir}\\metrics.csv')


def inc_joint(args):
    files_SmallParse = args.permutation
    task_output_dir = args.save+'/incremental_joint_learning'
    os.makedirs(task_output_dir, exist_ok=True)
    with open(task_output_dir+'\order.txt','w') as files:
        files.write(str(files_SmallParse))

    model_path = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True, do_lower_case=False)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    tokenizer.model_max_length = model.config.max_position_embeddings - 2

    data_collator = CustomDataCollator_AD(
        tokenizer, pad_to_multiple_of=None
    )
    data_inc_list = []
    for idx, data_name in enumerate(files_SmallParse):
        print(f'idx={idx} data_name={data_name}')
        data_inc_list.append(data_name)
        logdata_train = multi_task_dataset_AD(ori_data_path, data_inc_list, tokenizer)
        train_dataloader = DataLoader(logdata_train, shuffle=True, collate_fn=data_collator, batch_size=25)
        logdata_eval = multi_task_dataset_AD(ori_data_path, data_inc_list, tokenizer, train_eval_test='eval')
        eval_dataloader = DataLoader(logdata_eval, shuffle=True, collate_fn=data_collator, batch_size=25)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.001,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=5e-6,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        num_train_epochs = 5
        lr_scheduler = get_scheduler(
            name='polynomial',
            optimizer=optimizer,
            num_warmup_steps=num_train_epochs * len(train_dataloader)//10,
            num_training_steps=num_train_epochs * len(train_dataloader),
        )

        max_train_steps = 3200
        completed_steps = 0
        for epoch in range(num_train_epochs):
            model.train()
            total_loss = []
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss.append(float(loss))
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

                # if completed_steps >= max_train_steps:
                #     break
            print('epoch = ', epoch, '   loss = ', sum(total_loss) / len(total_loss))

            if epoch % 1 == 0:
                best_metric = evaluate(model,  eval_dataloader, device)
                print("Finish training, best metric: ")
                print(best_metric)

        sub_task_output_dir = task_output_dir + f'/Num_{idx}_{data_name}'
        os.makedirs(sub_task_output_dir, exist_ok=True)
        all_metric={}
        for file in files_SmallParse:
            log_file = f'data_AD'
            metric = log_AD(tokenizer, model, device, log_file, max_length=256, dataset_name=file)
            print(metric)
            all_metric[file] = metric
        sorted_dict_by_key = dict(sorted(all_metric.items()))
        df = pd.DataFrame(sorted_dict_by_key)
        df.to_csv(f'{sub_task_output_dir}\\metrics.csv')

def sequential_keep_head(args):
    files_SmallParse = args.permutation
    task_output_dir = args.save+'/sequential_keep_head'
    os.makedirs(task_output_dir, exist_ok=True)
    with open(task_output_dir+'\order.txt','w') as files:
        files.write(str(files_SmallParse))
    # ori_data_path = 'D:\ZMJ\pythonProject\Log_continual_learning\datasets'
    model_path = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True, do_lower_case=False)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    tokenizer.model_max_length = model.config.max_position_embeddings - 2

    data_collator = CustomDataCollator_AD(
        tokenizer, pad_to_multiple_of=None
    )
    for idx,data_name in enumerate(files_SmallParse):
        print(f'idx={idx} data_name={data_name}')

        if idx != 0:
            print('finish load models')
            model.classifier = torch.load(task_output_dir+'/save_models/classifier.pth')
            for param in model.classifier.parameters():
                param.requires_grad = False

        logdata_train = multi_task_dataset_AD(ori_data_path, [data_name], tokenizer)
        train_dataloader = DataLoader(logdata_train, shuffle=True, collate_fn=data_collator, batch_size=25)
        logdata_eval = multi_task_dataset_AD(ori_data_path, [data_name], tokenizer, train_eval_test='eval')
        eval_dataloader = DataLoader(logdata_eval, shuffle=True, collate_fn=data_collator, batch_size=25)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=5e-6,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        num_train_epochs = 5
        lr_scheduler = get_scheduler(
            name='polynomial',
            optimizer=optimizer,
            num_warmup_steps=num_train_epochs * len(train_dataloader)//10,
            num_training_steps=num_train_epochs * len(train_dataloader),
        )

        max_train_steps = 3200
        completed_steps = 0
        for epoch in range(num_train_epochs):
            model.train()
            total_loss = []
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss.append(float(loss))
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

            print('epoch = ', epoch, '   loss = ', sum(total_loss) / len(total_loss))

            if epoch % 1 == 0:
                best_metric = evaluate(model,  eval_dataloader, device)
                print("Finish training, best metric: ")
                print(best_metric)

        if idx == 0:
            os.makedirs(task_output_dir+'/save_models',exist_ok=True)
            torch.save(model.classifier,task_output_dir+'/save_models/classifier.pth')

        sub_task_output_dir = task_output_dir+f'/Num_{idx}_{data_name}'
        os.makedirs(sub_task_output_dir, exist_ok=True)
        all_metric = {}
        for file in files_SmallParse:
            log_file = f'data_AD'
            metric = log_AD(tokenizer, model, device, log_file, max_length=256, dataset_name=file)
            print(metric)
            all_metric[file] = metric
        sorted_dict_by_key = dict(sorted(all_metric.items()))
        df = pd.DataFrame(sorted_dict_by_key)
        df.to_csv(f'{sub_task_output_dir}\\metrics.csv')

def sequential_keep_body(args):
    files_SmallParse = args.permutation
    task_output_dir = args.save+'/sequential_keep_body'
    os.makedirs(task_output_dir, exist_ok=True)
    with open(task_output_dir+'\order.txt','w') as files:
        files.write(str(files_SmallParse))

    model_path = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True, do_lower_case=False)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    tokenizer.model_max_length = model.config.max_position_embeddings - 2

    data_collator = CustomDataCollator_AD(
        tokenizer, pad_to_multiple_of=None
    )
    for idx,data_name in enumerate(files_SmallParse):
        print(f'idx={idx} data_name={data_name}')

        if idx != 0:
            print('finish load models')
            model.roberta.load_state_dict(torch.load(task_output_dir+'/save_models/roberta.pth'))
            for param in model.roberta.parameters():
                param.requires_grad = False

        logdata_train = multi_task_dataset_AD(ori_data_path, [data_name], tokenizer)
        train_dataloader = DataLoader(logdata_train, shuffle=True, collate_fn=data_collator, batch_size=25)
        logdata_eval = multi_task_dataset_AD(ori_data_path, [data_name], tokenizer, train_eval_test='eval')
        eval_dataloader = DataLoader(logdata_eval, shuffle=True, collate_fn=data_collator, batch_size=25)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=5e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        num_train_epochs = 50
        lr_scheduler = get_scheduler(
            name='polynomial',
            optimizer=optimizer,
            num_warmup_steps=num_train_epochs * len(train_dataloader)//10,
            num_training_steps=num_train_epochs * len(train_dataloader),
        )

        max_train_steps = 3200
        completed_steps = 0
        for epoch in range(num_train_epochs):
            model.train()
            total_loss = []
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss.append(float(loss))
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

                # if completed_steps >= max_train_steps:
                #     break
            print('epoch = ', epoch, '   loss = ', sum(total_loss) / len(total_loss))

            if epoch % 1 == 0:
                best_metric = evaluate(model,  eval_dataloader, device)
                print("Finish training, best metric: ")
                print(best_metric)

        if idx == 0:
            os.makedirs(task_output_dir + '/save_models', exist_ok=True)
            torch.save(model.roberta.state_dict(),task_output_dir+'/save_models/roberta.pth')

        sub_task_output_dir = task_output_dir+f'/Num_{idx}_{data_name}'
        os.makedirs(sub_task_output_dir, exist_ok=True)
        all_metric = {}
        for file in files_SmallParse:
            log_file = f'data_AD'
            metric = log_AD(tokenizer, model, device, log_file, max_length=256, dataset_name=file)
            # print(metric)
            all_metric[file] = metric
        sorted_dict_by_key = dict(sorted(all_metric.items()))
        df = pd.DataFrame(sorted_dict_by_key)
        df.to_csv(f'{sub_task_output_dir}\\metrics.csv')

def sequential_keep_body_wo_1_9(args):
    files_SmallParse = args.permutation
    task_output_dir = args.save+'/sequential_keep_body_wo_1_9'
    os.makedirs(task_output_dir, exist_ok=True)
    with open(task_output_dir+'\order.txt','w') as files:
        files.write(str(files_SmallParse))

    model_path = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True, do_lower_case=False)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    # print(model)
    # a = model.roberta.encoder.layer
    # b = model.roberta.encoder.layer[0]
    # print(a)
    model.to(device)
    tokenizer.model_max_length = model.config.max_position_embeddings - 2

    data_collator = CustomDataCollator_AD(
        tokenizer, pad_to_multiple_of=None
    )
    for idx,data_name in enumerate(files_SmallParse):
        print(f'idx={idx} data_name={data_name}')

        if idx != 0:
            print('finish load models')
            model.classifier = torch.load(task_output_dir+'/save_models/classifier.pth')
            for param in model.classifier.parameters():
                param.requires_grad = False
            for i in range(9,12):
                model.roberta.encoder.layer[i] = torch.load(task_output_dir+'/save_models/roberta_layers.pth')[i]
                for param in model.roberta.encoder.layer[i].parameters():
                    param.requires_grad = False

        logdata_train = multi_task_dataset_AD(ori_data_path, [data_name], tokenizer)
        train_dataloader = DataLoader(logdata_train, shuffle=True, collate_fn=data_collator, batch_size=25)
        logdata_eval = multi_task_dataset_AD(ori_data_path, [data_name], tokenizer, train_eval_test='eval')
        eval_dataloader = DataLoader(logdata_eval, shuffle=True, collate_fn=data_collator, batch_size=25)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=5e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        num_train_epochs = 5
        lr_scheduler = get_scheduler(
            name='polynomial',
            optimizer=optimizer,
            num_warmup_steps=num_train_epochs * len(train_dataloader)//10,
            num_training_steps=num_train_epochs * len(train_dataloader),
        )

        max_train_steps = 3200
        completed_steps = 0
        for epoch in range(num_train_epochs):
            model.train()
            total_loss = []
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss.append(float(loss))
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

                # if completed_steps >= max_train_steps:
                #     break
            print('epoch = ', epoch, '   loss = ', sum(total_loss) / len(total_loss))

            if epoch % 1 == 0:
                best_metric = evaluate(model,  eval_dataloader, device)
                print("Finish training, best metric: ")
                print(best_metric)

        if idx == 0:
            os.makedirs(task_output_dir + '/save_models', exist_ok=True)
            torch.save(model.roberta.encoder.layer,task_output_dir+'/save_models/roberta_layers.pth')
            torch.save(model.classifier,task_output_dir+'/save_models/classifier.pth')

        sub_task_output_dir = task_output_dir+f'/Num_{idx}_{data_name}'
        os.makedirs(sub_task_output_dir, exist_ok=True)
        all_metric = {}
        for file in files_SmallParse:
            log_file = f'data_AD'
            metric = log_AD(tokenizer, model, device, log_file, max_length=256, dataset_name=file)
            # print(metric)
            all_metric[file] = metric
        sorted_dict_by_key = dict(sorted(all_metric.items()))
        df = pd.DataFrame(sorted_dict_by_key)
        df.to_csv(f'{sub_task_output_dir}\\metrics.csv')

from CL_models.EWC import OnlineEWC
from copy import deepcopy
def sequential_ewc(args):
    files_SmallParse = args.permutation
    task_output_dir = args.save+'/sequential_EWC'
    os.makedirs(task_output_dir, exist_ok=True)
    with open(task_output_dir+'\order.txt','w') as files:
        files.write(str(files_SmallParse))
    # ori_data_path = 'D:\ZMJ\pythonProject\Log_continual_learning\datasets'
    model_path = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True, do_lower_case=False)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    tokenizer.model_max_length = model.config.max_position_embeddings - 2

    data_collator = CustomDataCollator_AD(
        tokenizer, pad_to_multiple_of=None
    )
    last_dataname= None
    for idx, data_name in enumerate(files_SmallParse):
        print(f'idx={idx} data_name={data_name}')

        logdata_train = multi_task_dataset_AD(ori_data_path, [data_name], tokenizer)
        train_dataloader = DataLoader(logdata_train, shuffle=True, collate_fn=data_collator, batch_size=16)

        logdata_eval = multi_task_dataset_AD(ori_data_path, [data_name], tokenizer, train_eval_test='eval')
        eval_dataloader = DataLoader(logdata_eval, shuffle=True, collate_fn=data_collator, batch_size=16)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=5e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        num_train_epochs = 5
        lr_scheduler = get_scheduler(
            name='polynomial',
            optimizer=optimizer,
            num_warmup_steps=num_train_epochs * len(train_dataloader)//10,
            num_training_steps=num_train_epochs * len(train_dataloader),
        )

        if idx == 1999:
            max_train_steps = 3200
            completed_steps = 0
            for epoch in range(num_train_epochs):
                model.train()
                total_loss = []
                for step, batch in enumerate(train_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    total_loss.append(float(loss))
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    completed_steps += 1


                print('epoch = ', epoch, '   loss = ', sum(total_loss) / len(total_loss))

                if epoch % 1 == 0:
                    best_metric = evaluate(model,  eval_dataloader, device)
                    print("Finish training, best metric: ")
                    print(best_metric)
            os.makedirs(task_output_dir+'/save_models', exist_ok=True)
            # torch.save(model.state_dict(), task_output_dir + f'/save_models/Num_{idx}_{data_name}_model.pth')
            # last_dataname = data_name

        else:
            print('+++++++++++++++++++++++++++++++++++++')
            model_old = deepcopy(model)
            # model_old.load_state_dict(torch.load(task_output_dir+f'/save_models/Num_{idx-1}_{last_dataname}_model.pth'))
            for p in model_old.parameters():
                p.requires_grad = False

            fisher = None
            ewc = OnlineEWC(model, model_old, "cuda", fisher=fisher)
            importance = 75000
            EPS = 1e-20
            phi = 0.95
            max_train_steps = 3200
            completed_steps = 0
            for epoch in range(num_train_epochs):
                model.train()
                total_loss = []
                for step, batch in enumerate(train_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()

                    loss_ewc = importance * ewc.penalty()
                    if loss_ewc != 0.:
                        loss_ewc.backward()
                    loss += loss_ewc
                    total_loss.append(float(loss))

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    completed_steps += 1
                print('epoch = ', epoch, '   loss = ', sum(total_loss) / len(total_loss))
                if epoch % 1 == 0:
                    best_metric = evaluate(model,  eval_dataloader, device)
                    print("Finish training, best metric: ")
                    print(best_metric)
            last_dataname = data_name
            # torch.save(model.state_dict(), task_output_dir + f'/save_models/Num_{idx}_{data_name}_model.pth')
            ewc.update(train_dataloader)

            if fisher is None:
                fisher = deepcopy(ewc.get_fisher())

                fisher = {n: (fisher[n] - fisher[n].min()) / (fisher[n].max() - fisher[n].min() + EPS) for n in fisher}

                print("\n New fisher (normalized):")
                print({n: (p.min().item(), p.median().item(), p.max().item()) for n, p in fisher.items()})
            else:
                new_fisher = ewc.get_fisher()
                for n in fisher:
                    new_fisher[n] = (new_fisher[n] - new_fisher[n].min()) / (
                                new_fisher[n].max() - new_fisher[n].min() + EPS)
                    fisher[n] = phi * fisher[n] + new_fisher[n]
                print("\n New fisher (normalized):")
                print({n: (p.min().item(), p.median().item(), p.max().item()) for n, p in fisher.items()})

        sub_task_output_dir = task_output_dir + f'/Num_{idx}_{data_name}'
        os.makedirs(sub_task_output_dir, exist_ok=True)
        all_metric = {}
        for file in files_SmallParse:
            log_file = f'data_AD'
            metric = log_AD(tokenizer, model, device, log_file, max_length=256, dataset_name=file)
            # print(metric)
            all_metric[file] = metric
        sorted_dict_by_key = dict(sorted(all_metric.items()))
        df = pd.DataFrame(sorted_dict_by_key)
        df.to_csv(f'{sub_task_output_dir}\\metrics.csv')


def sequential_er(args):
    files_SmallParse = args.permutation
    task_output_dir = args.save+'/sequential_er'
    os.makedirs(task_output_dir, exist_ok=True)
    with open(task_output_dir+'\order.txt','w') as files:
        files.write(str(files_SmallParse))
    # ori_data_path = 'D:\ZMJ\pythonProject\Log_continual_learning\datasets'
    model_path = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True, do_lower_case=False)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    tokenizer.model_max_length = model.config.max_position_embeddings - 2

    data_collator = CustomDataCollator_AD(
        tokenizer, pad_to_multiple_of=None
    )

    buffer_size = 200
    sample_K = 40
    buffer = []

    for idx, data_name in enumerate(files_SmallParse):
        print(f'idx={idx} data_name={data_name}')

        logdata_train = multi_task_dataset_buffer_AD(ori_data_path, [data_name], tokenizer)
        logdata_train.get_buffer()
        if idx != 0:
            if sample_K <= len(buffer):
                sample = random.sample(buffer,sample_K)
            else:
                sample = buffer
            logdata_train.add_buffer(sample)

        train_dataloader = DataLoader(logdata_train, shuffle=True, collate_fn=data_collator, batch_size=16)
        logdata_eval = multi_task_dataset_buffer_AD(ori_data_path, [data_name], tokenizer, train_eval_test='eval')
        eval_dataloader = DataLoader(logdata_eval, shuffle=True, collate_fn=data_collator, batch_size=16)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=5e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        num_train_epochs = 5
        lr_scheduler = get_scheduler(
            name='polynomial',
            optimizer=optimizer,
            num_warmup_steps=num_train_epochs * len(train_dataloader)//10,
            num_training_steps=num_train_epochs * len(train_dataloader),
        )

        max_train_steps = 3200
        completed_steps = 0

        for epoch in range(num_train_epochs):
            model.train()
            total_loss = []
            for step, batch in enumerate(train_dataloader):

                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss.append(float(loss))
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

                # if completed_steps >= max_train_steps:
                #     break
            print('epoch = ', epoch, '   loss = ', sum(total_loss) / len(total_loss))

            if epoch % 1 == 0:
                best_metric = evaluate(model,  eval_dataloader, device)
                print("Finish training, best metric: ")
                print(best_metric)


        # logdata_train.buffer(20)
        buffer.extend(logdata_train.buffer_data)
        if len(buffer) > buffer_size:
            random.shuffle(buffer)
            buffer = buffer[:buffer_size]


        sub_task_output_dir = task_output_dir + f'/Num_{idx}_{data_name}'
        os.makedirs(sub_task_output_dir, exist_ok=True)
        all_metric = {}
        for file in files_SmallParse:
            log_file = f'data_AD'
            metric = log_AD(tokenizer, model, device, log_file, max_length=256, dataset_name=file)
            # print(metric)
            all_metric[file] = metric
        sorted_dict_by_key = dict(sorted(all_metric.items()))
        df = pd.DataFrame(sorted_dict_by_key)
        df.to_csv(f'{sub_task_output_dir}\\metrics.csv')


from CL_models.KD import DistillKL,HintLoss
def simple_knowledge_distill(args):
    files_SmallParse = args.permutation
    task_output_dir = args.save+'/simple_knowledge_distill'
    os.makedirs(task_output_dir, exist_ok=True)
    with open(task_output_dir+'\order.txt','w') as files:
        files.write(str(files_SmallParse))

    # ori_data_path = 'D:\ZMJ\pythonProject\Log_continual_learning\datasets'
    model_path = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True, do_lower_case=False)


    data_collator = CustomDataCollator_AD(
        tokenizer, pad_to_multiple_of=None
    )

    for idx, data_name in enumerate(files_SmallParse):
        if idx == 0:
            model_student = RobertaForSequenceClassification.from_pretrained(model_path)
            model_student.to(device)
            tokenizer.model_max_length = model_student.config.max_position_embeddings - 2
        else:
            model_teacher = RobertaForSequenceClassification.from_pretrained(model_path)
            model_teacher.load_state_dict(torch.load(task_output_dir+f'/save_models/model_teacher_{idx-1}.pth'))
            for param in model_teacher.parameters():
                param.requires_grad = False
            model_student = RobertaForSequenceClassification.from_pretrained(model_path)

            model_teacher.to(device)
            model_student.to(device)
            tokenizer.model_max_length = model_student.config.max_position_embeddings - 2

        print(f'idx={idx} data_name={data_name}')
        logdata_train = multi_task_dataset_AD(ori_data_path, [data_name], tokenizer)
        train_dataloader = DataLoader(logdata_train, shuffle=True, collate_fn=data_collator, batch_size=16)

        logdata_eval = multi_task_dataset_AD(ori_data_path, [data_name], tokenizer, train_eval_test='eval')
        eval_dataloader = DataLoader(logdata_eval, shuffle=True, collate_fn=data_collator, batch_size=16)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model_student.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model_student.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=5e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        num_train_epochs = 3
        lr_scheduler = get_scheduler(
            name='polynomial',
            optimizer=optimizer,
            num_warmup_steps=num_train_epochs * len(train_dataloader)//10,
            num_training_steps=num_train_epochs * len(train_dataloader),
        )

        max_train_steps = 3200
        completed_steps = 0

        if idx == 0:
            for epoch in range(num_train_epochs):
                model_student.train()
                total_loss = []
                for step, batch in enumerate(train_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model_student(**batch)
                    loss = outputs.loss
                    total_loss.append(float(loss))
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    completed_steps += 1

                    # if completed_steps >= max_train_steps:
                    #     break
                print('epoch = ', epoch, '   loss = ', sum(total_loss) / len(total_loss))

                if epoch % 1 == 0:
                    best_metric = evaluate(model_student,  eval_dataloader, device)
                    print("Finish training, best metric: ")
                    print(best_metric)
        else:
            alpha = 0.8
            compute_kl =DistillKL(T=10).to(device)
            for epoch in range(num_train_epochs):
                model_student.train()
                total_loss = []
                for step, batch in enumerate(train_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model_student(**batch)
                    loss_pre = outputs.loss
                    logits_s = outputs.logits
                    with torch.no_grad():
                        outputs_teacher = model_teacher(**batch)
                        logits_t = outputs_teacher.logits

                    loss_kl = compute_kl(logits_s,logits_t)
                    loss = alpha * loss_pre + (1-alpha) * loss_kl
                    total_loss.append(float(loss))
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    completed_steps += 1

                    # if completed_steps >= max_train_steps:
                    #     break
                print('epoch = ', epoch, '   loss = ', sum(total_loss) / len(total_loss))

                if epoch % 1 == 0:
                    best_metric = evaluate(model_student,  eval_dataloader, device)
                    print("Finish training, best metric: ")
                    print(best_metric)

        os.makedirs(task_output_dir+f'\save_models',exist_ok=True)
        torch.save(model_student.state_dict(),task_output_dir+f'\save_models\model_teacher_{idx}.pth')

        sub_task_output_dir = task_output_dir + f'/Num_{idx}_{data_name}'
        os.makedirs(sub_task_output_dir, exist_ok=True)
        all_metric = {}
        for file in files_SmallParse:
            log_file = f'data_AD'
            metric = log_AD(tokenizer, model_student, device, log_file, max_length=256, dataset_name=file)
            print(metric)
            all_metric[file] = metric
        sorted_dict_by_key = dict(sorted(all_metric.items()))
        df = pd.DataFrame(sorted_dict_by_key)
        df.to_csv(f'{sub_task_output_dir}\\metrics.csv')

def hint_knowledge_distill(args):
    files_SmallParse = args.permutation
    task_output_dir = args.save+'/hint_knowledge_distill'
    os.makedirs(task_output_dir, exist_ok=True)
    with open(task_output_dir+'\order.txt','w') as files:
        files.write(str(files_SmallParse))

    # ori_data_path = 'D:\ZMJ\pythonProject\Log_continual_learning\datasets'
    model_path = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True, do_lower_case=False)

    data_collator = CustomDataCollator_AD(
        tokenizer, pad_to_multiple_of=None
    )

    for idx, data_name in enumerate(files_SmallParse):
        if idx == 0:
            model_student = RobertaForSequenceClassification.from_pretrained(model_path)
            model_student.to(device)
            tokenizer.model_max_length = model_student.config.max_position_embeddings - 2
        else:
            model_teacher = RobertaForSequenceClassification.from_pretrained(model_path)
            model_teacher.load_state_dict(torch.load(task_output_dir+f'/save_models/model_teacher_{idx-1}.pth'))
            for param in model_teacher.parameters():
                param.requires_grad = False
            model_student = RobertaForSequenceClassification.from_pretrained(model_path)

            model_teacher.to(device)
            model_student.to(device)
            tokenizer.model_max_length = model_student.config.max_position_embeddings - 2

        print(f'idx={idx} data_name={data_name}')
        logdata_train = multi_task_dataset_AD(ori_data_path, [data_name], tokenizer)
        train_dataloader = DataLoader(logdata_train, shuffle=True, collate_fn=data_collator, batch_size=16)

        logdata_eval = multi_task_dataset_AD(ori_data_path, [data_name], tokenizer, train_eval_test='eval')
        eval_dataloader = DataLoader(logdata_eval, shuffle=True, collate_fn=data_collator, batch_size=16)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model_student.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model_student.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=5e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        num_train_epochs = 3
        lr_scheduler = get_scheduler(
            name='polynomial',
            optimizer=optimizer,
            num_warmup_steps=num_train_epochs * len(train_dataloader)//10,
            num_training_steps=num_train_epochs * len(train_dataloader),
        )

        max_train_steps = 3200
        completed_steps = 0

        if idx == 0:
            for epoch in range(num_train_epochs):
                model_student.train()
                total_loss = []
                for step, batch in enumerate(train_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model_student(**batch,output_hidden_states=True)
                    loss = outputs.loss
                    total_loss.append(float(loss))
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    completed_steps += 1

                print('epoch = ', epoch, '   loss = ', sum(total_loss) / len(total_loss))

                if epoch % 1 == 0:
                    best_metric = evaluate(model_student,  eval_dataloader, device)
                    print("Finish training, best metric: ")
                    print(best_metric)
        else:
            alpha = 0.5
            compute_kl =HintLoss().to(device)
            for epoch in range(num_train_epochs):
                model_student.train()
                total_loss = []
                for step, batch in enumerate(train_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model_student(**batch,output_hidden_states=True)
                    loss_pre = outputs.loss
                    hidden_state_s = outputs.hidden_states
                    with torch.no_grad():
                        outputs_teacher = model_teacher(**batch,output_hidden_states=True)
                        hidden_state_t = outputs_teacher.hidden_states

                    loss_kl = 0
                    for i in range(len(hidden_state_s)):
                        bsz = hidden_state_s[i].shape[0]
                        f_s = hidden_state_s[i].view(bsz,-1)
                        f_t = hidden_state_t[i].view(bsz,-1)
                        loss_kl_layer = compute_kl(f_s,f_t)
                        loss_kl += loss_kl_layer
                    loss_kl = loss_kl/len(hidden_state_s)

                    loss = alpha * loss_pre + (1-alpha) * loss_kl
                    total_loss.append(float(loss))
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    completed_steps += 1

                print('epoch = ', epoch, '   loss = ', sum(total_loss) / len(total_loss))

                if epoch % 1 == 0:
                    best_metric = evaluate(model_student,  eval_dataloader, device)
                    print("Finish training, best metric: ")
                    print(best_metric)

        os.makedirs(task_output_dir+f'\save_models',exist_ok=True)
        torch.save(model_student.state_dict(),task_output_dir+f'\save_models\model_teacher_{idx}.pth')

        sub_task_output_dir = task_output_dir + f'/Num_{idx}_{data_name}'
        os.makedirs(sub_task_output_dir, exist_ok=True)
        all_metric = {}
        for file in files_SmallParse:
            log_file = f'data_AD'
            metric = log_AD(tokenizer, model_student, device, log_file, max_length=256, dataset_name=file)
            print(metric)
            all_metric[file] = metric
        sorted_dict_by_key = dict(sorted(all_metric.items()))
        df = pd.DataFrame(sorted_dict_by_key)
        df.to_csv(f'{sub_task_output_dir}\\metrics.csv')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="token")
    parser.add_argument("--permutation", types=list, default=[
    "BGL",
    "HDFS",
    "Spirit",
    "Thunderbird"
    ])
    parser.add_argument("--save", types=str, default='output')
    args = parser.parse_args()
    if args.task == "SFT":
        print('++++++++++++++++++++++++++++++++++++++++++\nfine_tuning_woCL()')
        fine_tuning_woCL(args)
    elif args.task == "SeqFT":
        print('++++++++++++++++++++++++++++++++++++++++++\nsequential_fine_tuning()')
        sequential_fine_tuning(args)
    elif args.task == "Inc Joint":
        print('++++++++++++++++++++++++++++++++++++++++++\ninc_joint()')
        inc_joint(args)
    elif args.task == "Multisys":
        print('++++++++++++++++++++++++++++++++++++++++++\ntrain_MultiTask()')
        train_MultiTask(args)
    elif args.task == "Frozen Cls":
        print('++++++++++++++++++++++++++++++++++++++++++\nsequential_keep_head()')
        sequential_keep_head(args)
    elif args.task == "Frozen Enc":
        print('++++++++++++++++++++++++++++++++++++++++++\nsequential_keep_body()')
        sequential_keep_body(args)
    elif args.task == "Frozen B9":
        print('++++++++++++++++++++++++++++++++++++++++++\nsequential_keep_body_wo_1_9()')
        sequential_keep_body_wo_1_9(args)
    elif args.task == "EWC":
        print('++++++++++++++++++++++++++++++++++++++++++\nsequential_ewc()')
        sequential_ewc(args)
    elif args.task == "ER":
        print('++++++++++++++++++++++++++++++++++++++++++\nsequential_er()')
        sequential_er(args)
    elif args.task == "KD-Logit":
        print('++++++++++++++++++++++++++++++++++++++++++\nsimple_knowledge_distill()')
        simple_knowledge_distill(args)
    elif args.task == "KD-Rep":
        print('++++++++++++++++++++++++++++++++++++++++++\nHint_knowledge_distill()')
        hint_knowledge_distill(args)
    else:
        print('Wrong task')