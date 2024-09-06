# coding=utf-8
from __future__ import division
import numpy as np
from industrial_benchmark_python.goldstone.environment import environment as GoldstoneEnvironment
from industrial_benchmark_python.EffectiveAction import EffectiveAction
from collections import OrderedDict

from industrial_benchmark_python.IDS import IDS
'''
The MIT License (MIT)

Copyright 2017 Siemens AG

Author: Stefan Depeweg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

class IDS(IDS):
    '''
    Lightweight python implementation of the industrial benchmark
    Uses the same standard settings as in src/main/ressources/simTest.properties 
    of java implementation
    '''

    def __init__(self,p=50,stationary_p=True, inital_seed=None):
        '''
        p sets the setpoint hyperparameter (between 1-100) which will
        affect the dynamics and stochasticity.

        stationary_p = False will make the setpoint vary over time. This
        will make the system more non-stationary.
        '''

        super().__init__(p, stationary_p, inital_seed)

        # observables
        self.observable_keys = ['v','g','f','c','reward']

    def step(self,delta):
        delta = np.concatenate([delta, [0.0]])
        super().step(delta)
    
    def updateOperationalCosts(self):
        eNewHidden = self.state['coc']
        operationalcosts = eNewHidden - np.random.randn()*(1+0.005*eNewHidden)
        self.state['c'] = operationalcosts