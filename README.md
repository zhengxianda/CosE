# **Cos Embedding**
Source code and datasets for OM2019 paper "Multi-view Embedding for Biomedical Ontology Matching "

The structure of the model is shown in the figure, as follow:

![aaaa](https://github.com/zhengxianda/CosE/master/img/framework.png)


# **Code**
* the model in our experiment are in the following scripts:
    
    CosE/models/Sub.py
    
    CosE/models/Dis.py
    
* To train and evalute the effective of these model, please run:  

```python
cd CosE
bash make.sh
python3 train_sub.py
``` 
```python
cd CosE
bash make.sh
python3 train_dis.py
```

##Dependencies
* Python 3  
* Pytorch  
* Numpy

##Training Parameters

* The training parameters are in the following scripts.
 
  train times:1000
  
  batchsize:100  
  
  learning rate:0.001  
  
  embedding size:200
  
  negative rate:1
  
  optimize method:"SGD"


# **Datasets**

In our experiments, we use YAGO39K and Ontology FMA.

Datasets are required in the folder benchmarks/ in the following format, containing five files:

* train.txt: training file, format (e1, e2, rel).

* valid.txt: validation file, same format as train.txt

* test.txt: test file, same format as train.txt.

* entity2id.txt: all entities and corresponding ids, one per line.

* relation2id.txt: all relations and corresponding ids, one per line.

You have to run "n-n.py" to finish data preprocessing

# **Citation**

If you use this model or code, please cite it as follows: 

Huan Gao, Xianda Zheng, Weizhuo Li, Guilin Qi, and Meng Wang.Cosine-based Embedding for Completing Schematic Knowledge.NlPCC 2019