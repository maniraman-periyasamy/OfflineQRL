import gymnasium
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np

STATE_MEAN = [3.03736017e+02,  8.26348979e-04, -4.56448693e-02,  1.08840892e-01,
 -1.19461859e-03,  3.03835027e+02, -2.34984918e-04, -4.59271411e-02,
  9.90094757e-02, -1.06133354e-03,  3.03930111e+02, -1.14822620e-03,
 -4.52783120e-02,  9.50848521e-02, -9.13240992e-04,  3.04036938e+02,
 -1.88375528e-03, -4.54293472e-02,  1.06826585e-01, -7.35528914e-04,
  3.04149429e+02, -2.43071329e-03, -4.49196742e-02,  1.12491232e-01,
 -5.46957965e-04,  3.04286238e+02, -2.80094662e-03, -4.48336916e-02,
  1.36808467e-01, -3.70233311e-04,  3.04418807e+02, -2.97964592e-03,
 -4.42991289e-02,  1.32569313e-01, -1.78699259e-04,  3.04556278e+02,
 -2.96483726e-03, -4.28238489e-02,  1.37470894e-01,  1.48085807e-05,
  3.04697622e+02, -2.80390998e-03, -4.00871139e-02,  1.41344037e-01,
  1.60927051e-04,  3.04839727e+02, -2.48106830e-03, -4.02245729e-02,
  1.42105452e-01,  3.22841267e-04,]
STATE_STD = [1.54322903e+02, 6.18629888e-01, 7.07361842e-01, 7.27836362e+00,
 8.38921482e-02, 1.54665321e+02, 5.60310404e-01, 7.17135686e-01,
 7.35934005e+00, 7.54564492e-02, 1.55119965e+02, 5.15096300e-01,
 7.24676972e-01, 7.42637041e+00, 6.84085182e-02, 1.55670742e+02,
 4.83464998e-01, 7.28940656e-01, 7.48138581e+00, 6.26622397e-02,
 1.56303847e+02, 4.65733302e-01, 7.29457744e-01, 7.56876543e+00,
 5.86778850e-02, 1.56991925e+02, 4.62190363e-01, 7.28754587e-01,
 7.63806751e+00, 5.67528363e-02, 1.57724374e+02, 4.73037320e-01,
 7.26715823e-01, 7.70637775e+00, 5.70635184e-02, 1.58469268e+02,
 4.97932993e-01, 7.23159692e-01, 7.74133932e+00, 5.96524359e-02,
 1.59215995e+02, 5.36670479e-01, 7.16573326e-01, 7.75153670e+00,
 6.43153778e-02, 1.59967835e+02, 5.88688781e-01, 7.06242686e-01,
 7.74268083e+00, 7.05891822e-02,]
STATE_MIN = [32.65,        -1.4835299,   -1.,         -43.989998,    -0.26179934,
  32.99,        -1.3962635,   -1.,         -43.989998,    -0.24434614,
  33.33,        -1.308997,    -1.,         -43.989998,    -0.2268928,
  33.33,        -1.256637,    -1.,         -43.989998,    -0.2268928,
  33.33,        -1.2217306,   -1.,         -43.989998,    -0.2268928,
  33.33,        -1.2217306,   -1.,         -43.989998,    -0.20943952,
  33.33,        -1.2217306,   -1.,         -43.989998,    -0.1919862,
  33.33,        -1.256637,    -1.,         -43.989998,    -0.20943975,
  33.33,        -1.3264502,   -1.,         -43.989998,    -0.20943975,
   0.,          -1.3962635,   -1.,         -43.989998,    -0.22689301,]
STATE_MAX = [6.0481000e+02, 1.4835298e+00, 1.0000000e+00, 3.5740020e+01, 2.4434614e-01,
 6.0275000e+02, 1.3962634e+00, 1.0000000e+00, 3.5740020e+01, 2.2689295e-01,
 6.0275000e+02, 1.3264502e+00, 1.0000000e+00, 3.5740020e+01, 2.2689247e-01,
 6.0137000e+02, 1.2740904e+00, 1.0000000e+00, 3.5740020e+01, 2.0943928e-01,
 6.0069000e+02, 1.2391838e+00, 1.0000000e+00, 3.7119995e+01, 2.0943928e-01,
 6.0069000e+02, 1.2217305e+00, 1.0000000e+00, 3.7119995e+01, 1.9198656e-01,
 6.0378000e+02, 1.2391838e+00, 1.0000000e+00, 3.7119995e+01, 1.9198656e-01,
 6.0550000e+02, 1.2566371e+00, 1.0000000e+00, 3.7119995e+01, 1.9198632e-01,
 6.0825000e+02, 1.3089969e+00, 1.0000000e+00, 5.2920000e+01, 2.0943952e-01,
 6.1168000e+02, 1.3962634e+00, 1.0000000e+00, 5.2920000e+01, 2.0943958e-01,]

class QlindaCartPoleEnv(gymnasium.Env):
    
    def __init__(self, pos_model_path, theta_model_path):
        self.pos_delta_model = tf.keras.models.load_model(pos_model_path)
        self.theta_delta_model = tf.keras.models.load_model(theta_model_path)
        self.state = np.array(STATE_MEAN)
        
        self.worst_pos_reward = (np.maximum(STATE_MAX[-5] - STATE_MEAN[-5], STATE_MEAN[-5] - STATE_MIN[-5])/STATE_STD[-5])**2
        self.worst_theta_reward = (np.maximum(STATE_MAX[-4] - STATE_MEAN[-4], STATE_MEAN[-4] - STATE_MIN[-4])/STATE_STD[-4])**2
    
    def step(self, action):    
        self.state[-3] = action
        
        #new_pos_delta = self.pos_delta_model.predict(np.array([self.state]), verbose=0)[1][0,0]
        #new_theta_delta = self.theta_delta_model.predict(np.array([self.state]), verbose=0)[1][0,0]
        # model.predict() causes memory leak
        new_pos_delta = self.pos_delta_model(np.array([self.state]))[1][0,0]
        new_theta_delta = self.theta_delta_model(np.array([self.state]))[1][0,0]
        new_pos = self.state[-5] + new_pos_delta
        new_theta = self.state[-4] + new_theta_delta
        
        self.state = np.concatenate([self.state[5:], np.array([new_pos, new_theta, 0., new_pos_delta, new_theta_delta])])
        
        reward = (self.worst_pos_reward - ((self.state[-5] - STATE_MEAN[-5]) / STATE_STD[-5])**2) + (self.worst_theta_reward - ((self.state[-4] - STATE_MEAN[-4]) / STATE_STD[-4])**2)
        reward = np.maximum(reward, 0.)
        
        terminated = True
        if (STATE_MIN <= self.state).all() and (self.state <= STATE_MAX).all():
            terminated = False
        
        truncated = False
        info = {}
        done = terminated
        
        return self.state, reward, terminated, truncated, info, done

    def reset(self, pos_offset=0.):
        self.state = np.array(STATE_MEAN)
        self.state[0::5] = self.state[0::5] + pos_offset
        return self.state

    def render(self):
        raise NotImplementedError

    def close(self):
        pass