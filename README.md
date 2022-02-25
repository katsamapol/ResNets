# Residual Networks
ResNets PyTorch CIFAR10 Training

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
| Description | DType       | Arguments  | Default | 
| ----------- | ----------- | ---------- | ------- | 
| Learning rate                               | float  | lr         | 0.1 | 
| Data path                                   | string | path       | ./CIFAR10/  | 
| Number of epochs                            | int    | e          | 2   | 
| Number of data loader workers               | int    | w          | 12   | 
| Optimizer                                   | string | o          | sgd | 
| Number of residual layers                   | int    | n          | 4   | 
| Number of residual blocks for each of residual layer | int    | b           | 2 2 2 2 | 
| Number of channels in the first residual layer       | int    | c           | 64      | 
| Convolutional kernel sizes in each residual layer    | int    | covk        |         | 
| Skip connection kernel sizes in each residual layer  | int    | skipk       |         | 
| Average pool kernel sizes                            | int    | p           |         | 

## Accuracy
| Parameter Setting | Acc.        |
| ----------------- | ----------- |
|   <test>          |  80.00%     |
