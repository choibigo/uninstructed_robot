import os
import sys
sys.path.append(r'/home/bluepot/dw_workspace/git/OmniGibson')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import cv2
import json
import pickle

from omnigibson.utils.transform_utils import quat_multiply

from sim_scripts.mapping_utils import *
from sim_scripts.simulation_utils import *
from sim_scripts.visualization_functions import *


q2 = [0.01965616, -0.01058712, -0.70298743, -0.7117372]
q1 = [0.455, 0.542, -0.542, -0.455]

sq_norm = q1[0]**2 + q1[1]**2 + q1[2]**2 + q1[3]**2
conj = [q1[0], -q1[1], -q1[2], -q1[3]]

prod = quat_multiply(q2, conj)

print([prod[0]/sq_norm, prod[1]/sq_norm, prod[2]/sq_norm, prod[3]/sq_norm])


q1 = [0.01965616, -0.01058712, -0.70298743, -0.7117372]
q2 = [0.455, 0.542, -0.542, -0.455]
q2_conj = [0.455, -0.542, 0.542, 0.455]

# sq_norm = q1[0]**2 + q1[1]**2 + q1[2]**2 + q1[3]**2
# conj = [q1[0], -q1[1], -q1[2], -q1[3]]

prod = quat_multiply(q2_conj, q1)

print(prod)
[ 0.07186053  0.7114574  -0.69978434  0.04249699]
# print([prod[0]/sq_norm, prod[1]/sq_norm, prod[2]/sq_norm, prod[3]/sq_norm])

# [-0.006697294986555347, 0.004104922246426539, -0.8883472846673519, 0.4591050347626258]
# [-0.006697295084878802, -0.004104922306691189, 0.8883472977092404, -0.45910504150277887]

# [-0.006914089444666794, 0.001854074324576066, 0.9642913294108821, 0.26262656491793435]
# [0.006921829679817122, 0.0018561499371890144, 0.9653708412833848, 0.2629205720153918]

[-0.0812448538800644, -0.05035204931180517, 0.706949407189794, -0.7005487894004933]
[0.0812706880479643, -0.050368060212605675, 0.7071742023467776, -0.700771549294605]

[-0.7005487894004933, 0.05035204931180517, -0.7103364607766254, 0.0424300358159561]
[0.7007715492946048, 0.05036806021260567, -0.7105623329459594, 0.04244352768180259]