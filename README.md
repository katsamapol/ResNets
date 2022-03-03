# Residual Networks
ResNets PyTorch CIFAR10 Training

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Training
```
# Start training with: 
python main.py

# You can manually assign parameters with: 
python main.py --lr=0.01

# To list all configurable parameters use: 
python main.py -h
```
## Parameters
| Description | DType       | Arguments  | Default | 
| ----------- | ----------- | ---------- | ------- | 
| Learning rate                          | float  | lr         | 0.1 | 
| Dataset directory                      | string | path       | ./CIFAR10/  | 
| # of epochs                            | int    | e          | 5   | 
| # of data loader workers               | int    | w          | 8   | 
| Optimizer                              | string | o          | sgd | 
| # of residual layers                   | int    | n          | 4   | 
| # of residual blocks for each of residual layer | int    | b           | 2 2 2 2 | 
| # of channels in the first residual layer       | int    | c           | 64      | 
| Convolutional kernel sizes    		 | int    | f        	 | 3       | 
| Skip connection kernel sizes 			 | int    | k     	  	 | 1       | 

![ResNet18 setting example](resnet18.jpg)

## Accuracy
| Parameter Setting | Acc.        |
| ----------------- | ----------- |
|   <test>          |  80.00%     |
