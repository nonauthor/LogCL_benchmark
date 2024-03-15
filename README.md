# LogCL, cross-system log analysis includes log parsing and log anomaly detection tasks.
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
