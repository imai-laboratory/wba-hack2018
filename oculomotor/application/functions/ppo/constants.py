STATE_WINDOW = 1
CONVS = [[8, 3, 1], [16, 3, 1]]
FCS = [64, 64]
PADDING = 'VALID'
LSTM_UNIT = 64
FINAL_STEP = 10 ** 6
RANDOM_SEED = 1
ACTORS = 1

STATE_SHAPE = [8, 8, 2]
LSTM = False
VALUE_FACTOR = 1.0
ENTROPY_FACTOR = 0.05
EPSILON = 0.2
LR = 3e-4
LR_DECAY = 'linear'
GRAD_CLIP = 0.5
TIME_HORIZON = 2048
BATCH_SIZE = 64
GAMMA = 0.99
LAM = 0.95
EPOCH = 10
