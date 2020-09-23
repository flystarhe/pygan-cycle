# Jupyter
```
torch>=1.4.0
torchvision>=0.5.0
dominate>=2.4.0
visdom>=0.1.8.8
```

## Data
```
anomaly
├── train
│   ├── anomaly_large
│   │   ├── 1.png
│   │   └── 1.xml
│   ├── anomaly_small
│   │   ├── 2.png
│   │   └── 2.xml
│   ├── contamination
│   │   ├── 3.png
│   │   └── 3.xml
│   └── ok
│       ├── 4.png
│       ├── 4.xml
│       ├── 5.png
│       └── 5.xml
└── val
    ├── anomaly_large
    │   ├── 11.png
    │   └── 11.xml
    ├── anomaly_small
    │   ├── 12.png
    │   └── 12.xml
    ├── contamination
    │   ├── 13.png
    │   └── 13.xml
    └── ok
        ├── 14.png
        └── 14.xml
```

## Jupyter
```
%matplotlib inline
import os
import time
PROJ_HOME = "/data/sdv1/tmps/gits/pyhej-torch/pygan/module_cycle"
os.environ["PROJ_HOME"] = PROJ_HOME
os.chdir(PROJ_HOME)
!git log -1

# Train `--continue_train`
ARG_ROOT = ""
ARG_NAME = time.strftime("syrb_anomaly_%m%d_%H%M%S")
ARG_TAILS = "--gpu_ids 0,1,2,3 --model anomaly --input_nc 1 --output_nc 1"
ARG_TAILS += " --netD n_layers --ndf 64 --n_layers_D 3 --netG resnet_6blocks --ngf 64 --norm instance"
ARG_TAILS += " --dataset_mode anomaly --num_threads 8 --batch_size 8 --crop_size 256 --max_dataset_size 10000"
ARG_TAILS += " --display_freq 400 --update_html_freq 400 --save_latest_freq 5000 --save_epoch_freq 5"
ARG_TAILS += " --lr 0.0002 --gan_mode lsgan --pool_size 50 --lr_policy linear --lr_decay_iters 50"
ARG_TAILS += " --lambda_L1 10 --n_epochs 100 --n_epochs_decay 100 --beta1 0.5 --epoch latest"
ARG_TAILS += " --real_label 1.0 --fake_label 0.0 --cycle_size_G 5 --cycle_size_D 1"
ARGS = "--dataroot {} --name {} {}".format(ARG_ROOT, ARG_NAME, ARG_TAILS)
!PYTHONPATH={PROJ_HOME}:`pwd` python train.py {ARGS}

# Test
ARG_ROOT = ""
ARG_NAME = ""
ARG_TAILS = "--gpu_ids 0,1,2,3 --model anomaly --input_nc 1 --output_nc 1"
ARG_TAILS += " --netD n_layers --ndf 64 --n_layers_D 3 --netG resnet_6blocks --ngf 64 --norm instance"
ARG_TAILS += " --dataset_mode anomaly --num_threads 8 --batch_size 8 --crop_size 256 --max_dataset_size 10000"
ARG_TAILS += " --display_winsize 256 --results_dir ./results/ --phase test --num_test 50 --epoch latest"
ARGS = "--dataroot {} --name {} {}".format(ARG_ROOT, ARG_NAME, ARG_TAILS)
!PYTHONPATH={PROJ_HOME}:`pwd` python test_anomaly.py {ARGS}
```

* the type of GAN objective. `[vanilla| lsgan | wgangp]`
* learning rate policy. `[linear | step | plateau | cosine]`
* run `python -m visdom.server` and to view [http://localhost:8097/env/main](#)
* see more intermediate results, check out `./checkpoints/{ARG_NAME}/web/index.html`
* test results will be saved to a html file `./results/{ARG_NAME}/latest_test/index.html`
