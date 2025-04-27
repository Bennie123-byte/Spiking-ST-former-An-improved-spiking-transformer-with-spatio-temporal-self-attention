##code for paper:Spiking ST-former:An improved spiking transformer with spatio-temporal self-attention

## Requirements
timm==0.6.12; cupy==11.4.0; torch==1.12.1; spikingjelly==0.0.0.0.12; pyyaml; 

## Train
### Training  on ImageNet
Setting hyper-parameters in imagenet.yml

```
cd imagenet
python -m torch.distributed.launch --nproc_per_node=8 train.py
```

### Testing ImageNet Val data
```
cd imagenet
python test.py
```

### Training  on CIFAR10
Setting hyper-parameters in cifar10.yml
```
cd cifar10
python train.py
```

### Training  on CIFAR100
Setting hyper-parameters in cifar100.yml
```
cd cifar10
python train.py
```

### Training  on DVS128 Gesture
```
cd dvs128-gesture
python train.py
```

### Training  on CIFAR10-DVS
```
cd cifar10-dvs
python train.py
```

## Acknowledgement & Contact Information
[spikformer](https://github.com/ZK-Zhou/spikformer), [pytorch-image-models](https://github.com/huggingface/pytorch-image-models), [spikingjelly](https://github.com/fangwei123456/spikingjelly).

