# wba-hack2018
## 訓練データ作成方法
### OddOneOut
```bash
cd data
python oddoneoutgen.py --episode=<int> --length=<int> --scene=<int>
```
egocentric_images.shape: (episode*length, height=128, width=128, channel=3)  
allocentric_images.shape: (scene, height=128, width=128, channel=3)  
