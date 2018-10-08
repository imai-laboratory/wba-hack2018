from collections import OrderedDict

NUM_ACTIONS = 4

MODEL_PATHS = OrderedDict(
    PointToTarget='vae_models/pointtotarget/model.ckpt',
    ChangeDetection='vae_models/changedetection/model.ckpt',
    OddOneOut='vae_models/oddoneout/model.ckpt',
    VisualSearch='vae_models/visualsearch/model.ckpt',
    RandomDot='vae_models/randomdot/model.ckpt',
    MultipleObject='vae_models/multipleobjecttracking/model.ckpt'
)

PPO_MODEL_PATHS = OrderedDict(
    PointToTarget='ppo_models/pointtotarget/1007-1538951612.ckpt',
    ChangeDetection='ppo_models/changedetection/1007-1538935606.ckpt',
    OddOneOut='ppo_models/oddoneout/1007-1538935488.ckpt',
    VisualSearch='ppo_models/visualsearch/1007-1538933528.ckpt',
    RandomDot='ppo_models/randomdot/1007-1538937913.ckpt',
    MultipleObject='ppo_models/multipleobject/1007-1538933343.ckpt'
)
