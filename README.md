# wba-hack2018
## Docker for GPU
### build docker container
```bash
cd wba-hack2018
docker build -t wbap/oculomotor ./oculomotor/
```
### Running in Interactive Mode
```bash
cd wba-hack2018
./oculomotor/helpers/gpu_interactive.sh
```

## 訓練データ作成方法
### OddOneOut
```bash
cd data
python oddoneoutgen.py --episode=<int> --length=<int> --scene=<int>
```
- エゴセントリック画像
    - shape: (episode*length, height=128, width=128, channel=3)
    - save: data/images/OddOneOut/egocentric_images%Y%m%d.npy
- アロセントリック画像
    - shape: (scene, height=128, width=128, channel=3) 
    - save: data/images/OddOneOut/allocentric_images%Y%m%d.npy
