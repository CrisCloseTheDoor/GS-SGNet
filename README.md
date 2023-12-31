## Introduction
  Code implement for Guided-Sampling Stepwise Goal based Net
  ![image](https://github.com/CrisCloseTheDoor/GS-SGNet/blob/main/GS-SGNet.png)
  ![image](https://github.com/CrisCloseTheDoor/GS-SGNet/blob/main/sampling_details.png)
## Dataset download links
A preprocessed datasets(trajectory coordinates) is available in `Trajectron-plus-plus/experiments/pedestrians/raw`. The train/validation/test splits are the same as those found in [Social GAN](https://github.com/agrimgupta92/sgan).
To further know the ETH/UCY original datasets, see [ETH Dataset](http://www.vision.ee.ethz.ch/en/datasets/) and [UCY Dataset](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data)

## Preprocess
```
cd Trajectron-plus-plus/experiments/pedestrians
python process_data.py
```
This process mainly do trajectory sample split and rotation, will generate the data format for model to use.

## Training

### Phase 1. Traning the SGNet backbone
```
python tools\ethucy\train_deterministic.py --gpu 0 --seed 1 --dataset ZARA1 --model SGNet --num_workers 0 --batch_size 128 --dropout 0.0 --lr 1e-4 --epochs 60
```
The model will be saved at `tools\ethucy\checkpoints`, we suggest you create a folder `checkpoints\deteriministic` and put the saved model of phase1 here.
- TO DO: Put Pretrained model here


### Phase 2. Training the distribution predictor
```
python tools\ethucy\train_gaussian.py --gpu 0 --seed 1 --pretrained_dir tools\ethucy\checkpoints\deterministic\[PATH_TO_Phase1_checkpoints] --checkpoint None --sample_method qmc --checkpoint_npsn None --dataset ZARA1 --model SGNet_Gaussian --num_workers 0 --batch_size 16 --dropout 0.0 --lr 1e-4 --epochs 60
```
The model will be saved at `tools\ethucy\checkpoints`, We suggest you create a folder `checkpoints\distr_pretrained` and put the saved model of phase2 here
- TO DO: Put Pretrained model here


### Phase 3. Training the NPSN module
```
python tools\ethucy\train_gaussian.py --gpu 0 --seed 1 --pretrained_dir tools\ethucy\checkpoints\distr_pretrained\[PATH_TO_Phase2_checkpoints] --checkpoint None --sample_method npsn --checkpoint_npsn None --dataset ZARA1 --model SGNet_Gaussian --num_workers 0 --batch_size 16 --dropout 0.0 --lr 1e-4 --epochs 60
```
The model will be saved at `tools\ethucy\checkpoints`.
- TO DO: Put Pretrained model here

## Evaluate code
```
python \tools\ethucy\eval_gaussian.py --checkpoint tools\ethucy\checkpoints\distr_pretrained\[PATH_TO_Phase2_checkpoints]\zara1.pth --checkpoint_npsn tools\ethucy\checkpoints\[PATH_TO_Phase2_checkpoints]\zara1.pth --gpu 0 --dataset ZARA1 --model SGNet_Gaussian --sample_method npsn --num_workers 0 --batch_size 16
```

## Hyper parameters you can refer to
Hyper params below may help to obtain a best possible training results:
Datasets | ETH | HOTEL | UNIV | ZARA1 | ZARA2
--- | --- | --- | --- | --- | --- 
dropout | 0.5 | 0.3 | 0.0 | 0.0 | 0.0
lr_rate | 5e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4
Initializing method | Default | Default | nn.init.normal | nn.init.normal | nn.init.normal

The seed we use are all `seed=1`
