from conf import root_dir

# Train
use_tensorboard = True
ln_rate = 1e-4
ln_momentum = .9

# Model
model_challenger_path = root_dir + '/models/challenger.h5'
model_defender_path = root_dir + '/models/defender.h5'
features_path = root_dir + '/models/features.out'
labels_path = root_dir + '/models/labels.out'

cnn_filter_num = 32
cnn_filter_size = 3
drop_out_prob = 0.25
l2_reg = 1e-4
res_layer_num = 5
