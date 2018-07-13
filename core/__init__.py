# place all module imports here
from .sdf import *
from .fusion import Fusion
from .fusion_dm import FusionDM
import os

# place all global variables here (or in the main script)
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/depth')

