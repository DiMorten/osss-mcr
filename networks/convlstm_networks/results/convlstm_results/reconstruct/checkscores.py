import numpy as np
from icecream import ic
scores = np.load('scores_rebuilt_20180315_OpenPCS++_lm.npy')
ic(scores.shape)
ic(scores.dtype)
ic(np.average(scores), np.min(scores), np.max(scores))