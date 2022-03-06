from time import perf_counter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

#from torchsummary import summary

import os
import argparse

def main():
    project1_model()

def project1_model():
    #argument parser
    parser = argparse.ArgumentParser(description='ResNet PyTorch CIFAR10 Trainer')
    parser.add_argument('--o', default="sgd", type=str, help='optimizer (as string, eg: "sgd")')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--m', type=float, help='momentum')
    parser.add_argument('--wd', type=float, help='weight decay')
    parser.add_argument('--path', default="./CIFAR10/", type=str, help='dataset directory')
    parser.add_argument('--e', default=5, type=int, help='# of epochs')
    parser.add_argument('--wk', default=2, type=int, help='# of data loader workers')
    parser.add_argument('--n', default=4, type=int, help='# of residual layers')
    parser.add_argument('--b', default=[2,1,1,1], type=int, nargs='+', help='number of residual blocks in each of the residual layer (e.g. -b 2 2 2 2)')
    parser.add_argument('--c', default=64, type=int, help='# of channels in the first residual layer')
    parser.add_argument('--f', default=3, type=int, help='Convolutional kernel sizes')
    parser.add_argument('--k', default=1, type=int, help='Skip connection kernel sizes')
    parser.add_argument('--p', default=[1, 1], type=int, nargs='+', help='# of padding at convolutional input layer and convolutional blocks inside residual layer (e.g. -p 1 1)')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    # Hyper-parameters
    datapath = args.path
    num_epochs = args.e
    num_workers = args.wk
    optimizer = args.o
    num_layers = args.n
    num_blocks = args.b
    num_channels = args.c
    conv_kernel = args.f
    skip_kernel = args.k
    padding = args.p

    if(len(num_blocks)!=num_layers):
        print(f"{num_blocks}, block arguments of {len(num_blocks)} and layers of {num_layers} mismatched.")
        quit()

    # for i in range(len(num_blocks)):
    #     try:
    #         num_blocks[i] = int(num_blocks[i])
    #     except:
    #         print(f"Cannot convert string of '{num_blocks[i]}' to integer")
    #         quit()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==> Running on : {device}")

    #print('==> Building model..')
    model = ResNet(BasicBlock, num_blocks, num_layers, num_channels, conv_kernel, skip_kernel, padding).to(device)

    epoch_at_best_acc = 0
    loss_at_best_acc =0
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/'+optimizer+'.pt')
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
        best_acc = 0

    print(f'==> Number of CPU: {os.cpu_count()}')
    print(f'==> Number of workers: {num_workers}')
    print(f'==> #Epochs: {num_epochs+start_epoch}, #Layers: {num_layers}, #Blocks: {num_blocks}, In channel: {num_channels}, Conv kernel: {conv_kernel}x{conv_kernel}, Skip kernel: {skip_kernel}x{skip_kernel}, #Padding: {padding}')

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # Optimizer selector
    criterion = nn.CrossEntropyLoss()

    if args.wd:
        weight_decay = args.wd
    else:
        weight_decay = 0.0005
    if(optimizer=="nesterov"):
        #SGD with nesterov
        #SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=True)
        if args.lr:
            learning_rate = args.lr
        else:
            learning_rate = 0.1
        if args.m:
            momentum = args.m
        else:
            momentum = 0.9
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True)
        print('==> Optimizer is SGD with Nesterov')
        print(f'==> Momentum: {momentum}')
    elif(optimizer=="adam"):
        #Adam
        #Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        if args.lr:
            learning_rate = args.lr
        else:
            learning_rate = 0.001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        print('==> Optimizer is Adam')
    elif(optimizer=="adagrad"):
        #Adagrad
        #Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
        if args.lr:
            learning_rate = args.lr
        else:
            learning_rate = 0.01
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        print('==> Optimizer is Adagrad')
    elif(optimizer=="adadelta"):
        #Adadelta
        #Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
        if args.lr:
            learning_rate = args.lr
        else:
            learning_rate = 1.0
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        print('==> Optimizer is Adadelta')
    elif(optimizer=="sgd"):
        #SGD
        #SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
        if args.lr:
            learning_rate = args.lr
        else:
            learning_rate = 0.1
        if args.m:
            momentum = args.m
        else:
            momentum = 0.9
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        print('==> Optimizer is SGD')
        print(f'==> Momentum: {momentum}')
    else:
        print(f"Invalid optimizer choice: sgd, nesterov, adam, adagrad, and adadelta")
        quit()
    print(f'==> Learning rate: {learning_rate}')
    print(f'==> Weight decay: {weight_decay}')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    other_params = 0
    trainable_params = 0
    for num in model.parameters():
        if num.requires_grad:
            trainable_params += num.numel()
        else:
            other_params += num.numel()

    print(f"Trainable Parameters: {trainable_params}")
    print(f"Other Parameters: {other_params}")
    #summary(model, (3,32,32))

    # Image preprocessing modules
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root=datapath,
                                                 train=True, 
                                                 transform=transform_train,
                                                 download=True)

    test_dataset = torchvision.datasets.CIFAR10(root=datapath,
                                                train=False, 
                                                transform=transform_test)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=128, 
                                               shuffle=True, 
                                               num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=100, 
                                              shuffle=False, 
                                              num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    total_train_data_loading_time_secs = 0
    total_train_training_time_secs = 0
    total_test_data_loading_time_secs = 0
    total_test_training_time_secs = 0
    total_data_loading_time_secs = 0
    total_training_time_secs = 0

    for epoch in range(start_epoch, num_epochs):
        # Training
        print(f'Train epoch: {(epoch+1)} | ', end="")
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        epoch_train_data_loading_time_secs = 0
        epoch_train_training_time_secs = 0
        # Start train data-loading time for each epoch
        data_loading_time = perf_counter()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # End train data-loading time for each epoch
            data_loading_time = perf_counter()-data_loading_time
            epoch_train_data_loading_time_secs += data_loading_time
            total_train_data_loading_time_secs += data_loading_time

            # Start train training time for each epoch
            pre_forward_time = perf_counter()
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # End train training time for each epoch
            post_backward_time = perf_counter()
            

            # Calculating for total running for each epoch
            duration_secs = (post_backward_time - pre_forward_time)
            epoch_train_training_time_secs += duration_secs
            total_train_training_time_secs += duration_secs

            # Calculating for train loss and accuracy
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            data_loading_time = perf_counter()
 
        # print Train loss, accuracy, data-loading time, and training time
        print('Loss: *%.3f* | Acc: *%.3f%%* (%d/%d) '
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        #print('Loss: %.3f | Acc: %.3f%% (%d/%d) | data-loading time: *%.16f* secs | training time: *%.16f* secs | total time: *%.16f* secs'
                #% (train_loss/(batch_idx+1), 100.*correct/total, correct, total,epoch_train_data_loading_time_secs,epoch_train_training_time_secs,(epoch_train_data_loading_time_secs+epoch_train_training_time_secs)))

        print(f'Test epoch: {(epoch+1)} | ', end="")
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        epoch_test_data_loading_time_secs = 0
        epoch_test_training_time_secs = 0
        with torch.no_grad():
            # Start test data-loading time for each epoch
            data_loading_time = perf_counter()
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                # End test data-loading time for each epoch
                data_loading_time = perf_counter()-data_loading_time
                epoch_test_data_loading_time_secs += data_loading_time
                total_test_data_loading_time_secs += data_loading_time

                # Start test training time for each epoch
                pre_forward_time = perf_counter()
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # End test training time for each epoch
                post_forward_time = perf_counter()

                # Calculating for total running for each epoch
                forward_duration_secs = (post_forward_time - pre_forward_time)
                epoch_test_training_time_secs += forward_duration_secs
                total_test_training_time_secs += forward_duration_secs

                # Calculating for test loss and accuracy
                test_loss += loss.item()
                _, predicted = outputs.max(1) #get top-1 accuracy
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                data_loading_time = perf_counter()

            # print Test loss, accuracy, data-loading time, and training time
            avg_test_loss = test_loss/(batch_idx+1)
            print('Loss: *%.3f* | Acc: *%.3f%%* (%d/%d)' 
                % (avg_test_loss, 100.*correct/total, correct, total))

            #print('Test Loss: %.3f | Acc: %.3f%% (%d/%d) | data loading time: *%.16f* secs | training time: *%.16f* secs | total time: *%.16f* secs' 
                #% (test_loss/(batch_idx+1), 100.*correct/total, correct, total, epoch_test_data_loading_time_secs, epoch_test_training_time_secs, (epoch_test_data_loading_time_secs+epoch_test_training_time_secs)))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            #print('==> Saving..')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/'+optimizer+'.pt')
            best_acc = acc
            epoch_at_best_acc = epoch
            loss_at_best_acc = avg_test_loss
        scheduler.step()

    # print execution time summary output
    print(f'Best Accuracy: *{best_acc:.3f}%*, loss: *{loss_at_best_acc}*, at epoch: *{epoch_at_best_acc}*.')
    print(f'==> Execution time summary of {num_epochs} epoch(s).')    
    print(f"Train data loading time: sum. *{total_train_data_loading_time_secs}* secs | avg. *{total_train_data_loading_time_secs/num_epochs}* secs per epoch.")
    print(f"Train training time: sum. *{total_train_training_time_secs}* secs | avg. *{total_train_training_time_secs/num_epochs}* secs per epoch.")
    print(f"Train total time: sum. *{total_train_data_loading_time_secs+total_train_training_time_secs}* secs | avg. *{(total_train_data_loading_time_secs+total_train_training_time_secs)/num_epochs}* secs per epoch.")
    
    print(f"Test data loading time: sum. *{total_test_data_loading_time_secs}* secs | avg. *{total_test_data_loading_time_secs/num_epochs}* secs per epoch.")
    print(f"Test training time: sum. *{total_test_training_time_secs}* secs | avg. *{total_test_training_time_secs/num_epochs}* secs per epoch.")
    print(f"Test total time: sum. *{total_test_data_loading_time_secs+total_test_training_time_secs}* secs | avg. *{(total_test_data_loading_time_secs+total_test_training_time_secs)/num_epochs}* secs per epoch.")
    
    total_data_loading_time_secs = total_train_data_loading_time_secs+total_test_data_loading_time_secs
    total_training_time_secs = total_train_training_time_secs+total_test_training_time_secs
    print(f"Total data loading time: sum. *{total_data_loading_time_secs}* secs | avg. *{total_data_loading_time_secs/num_epochs}* secs per epoch.")
    print(f"Total training time: sum. *{total_training_time_secs}* secs | avg. *{total_training_time_secs/num_epochs}* secs per epoch.")
    total_running_time_secs = total_data_loading_time_secs+total_training_time_secs
    print(f"Total running time: sum. *{total_running_time_secs}* secs | avg. *{total_running_time_secs/num_epochs}* secs per epoch.")



class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, conv_kernel, skip_kernel, padding, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=conv_kernel, stride=stride, padding=padding[1], bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=conv_kernel, stride=1, padding=padding[1], bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes, kernel_size=skip_kernel, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block_type, num_blocks, num_layers, num_channels, conv_kernel, skip_kernel, padding, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = num_channels
        self.num_layers = num_layers
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=conv_kernel,
                               stride=1, padding=padding[0], bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        if(1<=self.num_layers):
            self.layer1 = self._make_layer(block_type, num_channels, num_blocks[0], conv_kernel, skip_kernel, padding, stride=1)
        if(2<=self.num_layers):
            num_channels*=2
            self.layer2 = self._make_layer(block_type, num_channels, num_blocks[1], conv_kernel, skip_kernel, padding, stride=2)
        if(3<=self.num_layers):
            num_channels*=2
            self.layer3 = self._make_layer(block_type, num_channels, num_blocks[2], conv_kernel, skip_kernel, padding, stride=2)
        if(4<=self.num_layers):
            num_channels*=2
            self.layer4 = self._make_layer(block_type, num_channels, num_blocks[3], conv_kernel, skip_kernel, padding, stride=2)
        if(5<=self.num_layers):
            num_channels*=2
            self.layer5 = self._make_layer(block_type, num_channels, num_blocks[4], conv_kernel, skip_kernel, padding, stride=2)
        self.linear = nn.Linear(num_channels, num_classes)

    def _make_layer(self, block_type, planes, num_blocks, conv_kernel, skip_kernel, padding, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block_type(self.in_planes, planes, conv_kernel, skip_kernel, padding, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        if(1<=self.num_layers):
            out = self.layer1(out)
        if(2<=self.num_layers):
            out = self.layer2(out)
        if(3<=self.num_layers):
            out = self.layer3(out)
        if(4<=self.num_layers):
            out = self.layer4(out)
        if(5<=self.num_layers):
            out = self.layer5(out)
        out = F.avg_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1)
        
        out = self.linear(out)
        return out

if __name__ == "__main__":
    main();