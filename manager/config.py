
from conf import root_dir

features_path = root_dir + '/models/features.out'
label_path = root_dir + '/models/labels.out'
eval_log_path = root_dir + '/log/eval.log'
error_log = root_dir + '/log/error.log'
tensorboard_path = root_dir + '/log/tensorboard/'
epochpickle_path = root_dir + '/log/curr_epoch'
# eval
eval_rounds = 60
eval_timelimit = .2
# opt
batch_size = 256
opt_epochs = 30
opt_idle_time = 100
opt_wait_date_time = 300
opt_data_threshold = 30

# slefplay

selfplay_monitor = 1
selfplay_timelimit = 1