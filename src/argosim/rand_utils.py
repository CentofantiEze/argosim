""" Random utils.

A small module to hold a class for a temporary seed, 
so as to not mess with any global seed.

:Authors: Samuel Gullin <gullin@ia.forth.gr>

"""

import numpy as np

class local_seed:
    """
    Can be instantiated with no seed, in whihc case it will yield the default behaviour.
    """
    def __init__(self, seed=None):
        self.seed = seed
        self.old_state = None

    def __enter__(self):
        if self.seed:
            self.old_state = np.random.get_state()
            np.random.seed(self.seed)
    
    def __exit__(self, *_):
        if self.seed:
            np.random.set_state(self.old_state)
