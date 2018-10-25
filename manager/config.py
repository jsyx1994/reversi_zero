
from conf import root_dir

features_path = root_dir + '/models/features.out'
label_path = root_dir + '/models/labels.out'
eval_log_path = root_dir + '/log/eval.log'
error_log = root_dir + '/log/error.log'

# eval
eval_rounds = 20

# opt
batch_size = 256
opt_epochs = 20
opt_idle_time = 60
opt_wait_date_time = 300
opt_data_threshold = 10000

# slefplay

selfplay_monitor = 0
