
import os
from multiprocessing import Lock

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def memory_gpu(fraction):
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    set_session(tf.Session(config=config))

model_lock = Lock()
data_lock = Lock()
log_lock = Lock()

root_dir = os.path.dirname(__file__)
