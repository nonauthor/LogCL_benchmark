# LogCL: A Benchmark for Cross-System Continual Log Analysis
This repository contains datasets and PyTorch implementation for baselines in LogCL: A Benchmark for Cross-System Continual Log Analysis.

Automated log analysis plays a pivotal role in software operations, with deep learning proving effective in individual system log analysis. However, the increasing number of systems has posed a challenge to traditional approaches, particularly in cross-system continual log analysis. Existing methods, whether based on multi-task learning or transfer learning, have shown limitations in adaptability and knowledge preservation. To tackle this issue, we introduce continual learning for cross-system continual log analysis, enabling models to learn log knowledge from new systems while preserving knowledge from previous ones.  Unfortunately, there is currently no benchmark available to evaluate the continual learning methods in cross-system continual log analysis. To fill this gap, we present LogCL, a novel benchmark consisting of log parsing and log anomaly detection tasks with logs from multiple systems. We also propose four essential abilities for continual learning: cross-system preservation, cross-system accumulation, cross-system generalization, and model utility. Our evaluation of various continual learning methods on LogCL demonstrates that continual learning methods can enhance cross-system log knowledge preservation and improve model utility but are less effective at accumulation and generalization. Additionally, we conduct an in-depth study on the robustness of continual learning methods in cross-system continuous log analysis and the factors that influence their robustness.

# Usage
## Requirements
numpy==1.23.5
torch==1.12.1
transformer==4.26.0
huggingface_hub=0.10.1
## Structure of Files
‘’‘
|-- dataset_cl
|    |-- datasets_AD
|    |-- datasets_LP
|    |-- logs_1k

|-- CL_models
|    |-- EWC.py
|    |-- er.py
|    |-- KD.py

|-- Log_Parsing
|    |-- benchmark_continual.py
|    |-- dataset_log.py
|    |-- evaluation_continual_learning.py
|    |-- log_parsing_zmj.py
|    |-- train_zmj.py
|    |-- evaluation

|-- Log_Parsing
|    |-- Log_anomaly_detection.py
|    |-- dataset_log_AD.py
|    |-- evaluation_continual_learning.py
|    |-- eval_AD.py
|    |-- read_results.py
|    |-- save_csv.py
|    |-- train_zmj_AD.py
’‘’


# Log parsing
## SFT
python train_zmj.py -task SFT
## SeqFT
python train_zmj.py -task SeqFT
## Inc Joint
python train_zmj.py -task Inc Joint
## Multisys
python train_zmj.py -task Multisys
## Frozen Cls
python train_zmj.py -task Frozen Cls
## Frozen Enc
python train_zmj.py -task Frozen Enc
## Frozen B9
python train_zmj.py -task Frozen B9
## EWC
python train_zmj.py -task EWC
## ER
python train_zmj.py -task ER
## KD-Logit
python train_zmj.py -task KD-Logit
## KD-Rep
python train_zmj.py -task KD-Rep

# Log anomaly detection
## SFT
python train_zmj_AD.py -task SFT
## SeqFT
python train_zmj_AD.py -task SeqFT
## Inc Joint
python train_zmj_AD.py -task Inc Joint
## Multisys
python train_zmj_AD.py -task Multisys
## Frozen Cls
python train_zmj_AD.py -task Frozen Cls
## Frozen Enc
python train_zmj_AD.py -task Frozen Enc
## Frozen B9
python train_zmj_AD.py -task Frozen B9
## EWC
python train_zmj_AD.py -task EWC
## ER
python train_zmj_AD.py -task ER
## KD-Logit
python train_zmj_AD.py -task KD-Logit
## KD-Rep
python train_zmj_AD.py -task KD-Rep
