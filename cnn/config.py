from conf import root_dir
from game.common import BOARD_SIZE
# Train
use_tensorboard = True
ln_rate = 1e-2
ln_momentum = .9
decay = 1e-2
# Model
model_challenger_path = root_dir + '/models/challenger.h5'
model_defender_path = root_dir + '/models/defender.h5'
features_path = root_dir + '/models/features.out'
labels_path = root_dir + '/models/labels.out'

cnn_filter_num = 32
cnn_filter_size = int(BOARD_SIZE / 2) - 1
drop_out_prob = 0.25
l2_reg = 1e-4
res_layer_num = 0

channel = 3