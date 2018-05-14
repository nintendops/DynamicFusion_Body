# place all module imports here
from .sdf import load_sdf
from .fusion import Fusion
import os

# place all global variables here (or in the main script)
DATA_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/data/'
