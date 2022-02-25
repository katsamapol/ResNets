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
| Arguments  | Description |
| -----------| ----------- |
| lr         | learning rate (as float, eg: "0.1")   |
| e          | number of epochs (as int)             |
| o          | optimizer (as string, eg: "sgd")      |

## Accuracy
| Parameter Setting | Acc.        |
| ----------------- | ----------- |
|   <test>          |  80.00%     |
