# wba-hack2018
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
