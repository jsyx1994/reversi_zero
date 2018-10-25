
import os
from multiprocessing import Lock

model_lock = Lock()
data_lock = Lock()
log_lock = Lock()

root_dir = os.path.dirname(__file__)
