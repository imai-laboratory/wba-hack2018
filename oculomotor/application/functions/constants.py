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
    PointToTarget=None,
    ChangeDetection=None,
    OddOneOut=None,
    VisualSearch=None,
    RandomDot=None,
    MultipleObject=None
)
