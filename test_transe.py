import config
from models import *
import json
import os 
os.environ['CUDA_VISIBLE_DEVICES']='1'
con = config.Config()
#Input training files from benchmarks/FB15K/ folder.
con.set_in_path("./benchmarks/Dis50-YAGO39K/")
#True: Input test files from the same folder.
con.set_work_threads(8)
con.set_dimension(100)
con.set_result_dir("./result")
con.set_test_link(True)
con.set_test_triple(False)
con.init()
con.set_test_model(TransE)
con.test()
