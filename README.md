# wba-hack2018
## Directory Structure
- data
  - data extractor for beta VAE training
- oculomotor
  - main execution files

## Create Datasets
### OddOneOut (example)
See `oculomotor` for the hackathon architecture.

## 訓練データ作成方法
### OddOneOut
```bash
cd data
python oddoneoutgen.py --episode=<int> --length=<int> --scene=<int>
```
- create egocentric datasets
    - shape: (episode*length, height=128, width=128, channel=3)
    - save: data/images/OddOneOut/egocentric_images%Y%m%d.npy
- create allocentric datasets
    - shape: (scene, height=128, width=128, channel=3) 
    - save: data/images/OddOneOut/allocentric_images%Y%m%d.npy
