# Residual Networks

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Training
```
# Start training with: 
python resnet.py

# You can manually assign parameters with: 
python resnet.py --lr=0.01

# To list all configurable parameters use: 
python resnet.py -h
```
## Parameters
| Description | DType       | Arguments  | default | 
| ----------- | ----------- | ---------- | ------- | 
| learning rate    | float  | lr         | 0.1 | 
| number of epochs | int    | e          | 2   | 
| optimizer        | string | o          | sgd | 

## Accuracy
| Parameter Setting | Acc.        |
| ----------------- | ----------- |
|   <test>          |  80.00%     |
