
from conf import root_dir

features_path = root_dir + '/models/features.out'
label_path = root_dir + '/models/labels.out'
eval_log_path = root_dir + '/log/eval.log'
error_log = root_dir + '/log/error.log'
tensorboard_path = root_dir + '/log/tensorboard/'
epochpickle_path = root_dir + '/log/curr_epoch'
# eval
eval_rounds = 200
eval_timelimit = 0.3
# selfplay
self_pool = 2
selfplay_monitor = 0
selfplay_timelimit = 4

# opt
batch_size = 128
opt_epochs = 100
opt_wait_date_time = 60
opt_data_threshold = 2000
opt_idle_time = 60

