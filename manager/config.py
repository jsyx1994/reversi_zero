
from conf import root_dir

features_path = root_dir + '/models/features.out'
label_path = root_dir + '/models/labels.out'
eval_log_path = root_dir + '/log/eval.log'
error_log = root_dir + '/log/error.log'
tensorboard_path = root_dir + '/log/tensorboard/'
epochpickle_path = root_dir + '/log/curr_epoch'
# eval
eval_rounds = 200
eval_timelimit = .6
# selfplay
self_pool = 16
selfplay_monitor = 0
selfplay_timelimit = 4

# opt
batch_size = 1024
opt_epochs = 100

opt_wait_date_time = 300
opt_data_threshold = 10000
opt_idle_time = .25 * opt_data_threshold * (selfplay_timelimit * .8) / self_pool

