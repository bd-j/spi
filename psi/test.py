import numpy as np
from models import MILESInterpolator

mlib = '/Users/bjohnson/Projects/psi/data/miles/miles_prugniel.h5'
fgk = {'teff': (3500.0, 7000.0)}

sps = MILESInterpolator(training_data=mlib)
sps.restrict_sample(bounds=fgk)
sps.construct_design_matrix()
sps.train()
