import torch
import numpy as np

class Actor:
    def __init__(self, args=None):
        self.learning_rate = 1e-4 if args is None else args['learning_rate']
        self.soft_update_tau = 2 ** -8 if args is None else args['soft_update_tau']  # 5e-3 ~= 2 ** -8
        self.model = None
        self.optimizer = None
        self.criterion = None