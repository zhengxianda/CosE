import config
from  models import *
import json
import os 
os.environ['CUDA_VISIBLE_DEVICES']='1'
con1 = config.Config()
con1.set_in_path("./benchmarks/Sub-YAGO39K1/")
con1.set_work_threads(8)
con1.set_train_times(1000)
con1.set_nbatches(100)
con1.set_alpha(0.001)
con1.set_bern(0)
con1.set_dimension(200)
con1.set_margin(5.0)
con1.set_ent_neg_rate(1)
con1.set_rel_neg_rate(0)
con1.set_opt_method("SGD")
con1.set_save_steps(5000)
con1.set_valid_steps(5000)
con1.set_early_stopping_patience(10)
con1.set_checkpoint_dir("./checkpoint")
con1.set_result_dir("./result")
con1.set_test_link(True)
con1.set_test_triple(False)
con1.init()
con1.set_train_model(Sub)
con1.train()
