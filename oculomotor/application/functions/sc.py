import numpy as np
import brica

EX_SCALE = 2
EY_SCALE = 2

class SC(object):
    """ 
    SC (superior colliculus) module.
    SC outputs action for saccade eye movement.
    """
    def __init__(self):
        self.timing = brica.Timing(6, 1, 0)

        self.last_fef_data = None

    def __call__(self, inputs):
        if 'from_fef' not in inputs:
            raise Exception('SC did not recieve from FEF')
        if 'from_bg' not in inputs:
            raise Exception('SC did not recieve from BG')

        # Likelihoods and eye movment params from accumulators in FEF module.
        fef_data = inputs['from_fef']
        # Likelihood thresolds from BG module.
        # bg input (4)
        bg_in = inputs['from_bg']
        bg_12 = np.tile(
            np.concatenate((np.tile([bg_in[0]], 4), np.tile([bg_in[1]], 4)), axis=0), 4
        )
        bg_34 = np.tile(
            np.concatenate((np.tile([bg_in[2]], 4), np.tile([bg_in[3]], 4)), axis=0), 4
        )
        bg_1234 = np.concatenate((bg_12, bg_34), axis=0)
        bg_data = np.tile(bg_1234, 3).tolist()

        action = self._decide_action(fef_data, bg_data)
        # Store FEF data for debug visualizer
        self.last_fef_data = fef_data
        return dict(to_environment=action)

    def _decide_action(self, fef_data, bg_data):
        sum_ex = 0.0
        sum_ey = 0.0

        assert(len(fef_data) == len(bg_data))

        count = 0
        EX_SCALE = 2


        # Calculate average eye ex, ey with has likelihoods over
        # the thresholds from BG.
        for i,data in enumerate(fef_data):
            likelihood = data[0]
            ex = data[1]
            ey = data[2]
            likelihood_threshold = bg_data[i]

            if likelihood > likelihood_threshold:
                nex = ex * EX_SCALE
                ney = ey * EY_SCALE
                sum_ex +=  min(1, nex)
                sum_ey += min(1, ney)
                count += 1

        # Action values should be within range [-1.0~1.0]
        if count != 0:
            action = [sum_ex / count, sum_ey / count]
        else:
            action = [0.0, 0.0]
        
        return np.array(action, dtype=np.float32)
