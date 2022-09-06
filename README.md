# **Cos Embedding**
Source code and datasets of paper "Cosine-based Embedding for Completing LightWeight Schematic Knowledge" for submitting *applied sciences*, which is a substantial extension of the preliminary results published in Proceedings of the 9th International Conference on Natural Language Processing and Chinese Computing (NLPCC 2019)

The structure of the model is shown in the figure, as follow:

![picture](https://github.com/zhengxianda/CosE/raw/master/img/framework.jpg)


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

## Dependencies
* Python 3  
* Pytorch  
* Numpy

## Training Parameters

* The training parameters are in the following scripts.
 
  train times:1000
  
  batchsize:100  
  
  learning rate:0.001  
  
  embedding size:200
  
  negative rate:1
  
  optimize method:"SGD"


# **Datasets**

In our experiments, we use YAGO39K[1], Ontology FMA[2], FoodOn[3] and HeLiS[4].

Datasets are required in the folder benchmarks/ in the following format, containing five files:

* train.txt: training file, format (e1, e2, rel).

* valid.txt: validation file, same format as train.txt

* test.txt: test file, same format as train.txt.

* entity2id.txt: all entities and corresponding ids, one per line.

* relation2id.txt: all relations and corresponding ids, one per line.

You have to run "n-n.py" to finish data preprocessing

[1]Lv, X.; Hou, L.; Li, J.; Liu, Z. Differentiating Concepts and Instances for Knowledge Graph Embedding. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, 31 October-4 November 2018; pp. 1971–1979.

[2]Noy, N. F.; Musen, M. A.; Mejino Jr, J. L.; Rosse, C. Pushing the envelope: challenges in a frame-based representation of human anatomy. Data Knowl. Eng., 2004, 48(3), 335–359. 

[3]Dooley, D. M.; Griffiths, E. J.; Gosal, G. S.; Buttigieg, P. L.; Hoehndorf, R.; Lange, M. C.; Schriml L. M.; Brinkman F. S.; Hsiao, W. W. FoodOn: a harmonized food ontology to increase global food traceability, quality control and data integration. Science of Food, 2018, 2(1), 1–10. 

[4]Dragoni, M.; Bailoni, T.; Maimone, R.; Eccher, C. HeLiS: An Ontology for Supporting Healthy Lifestyles. In Proceedings of the 17th International Semantic Web Conference, Monterey, Monterey, California, USA, 8-12 October 2018; pp. 53–69.

<!-- # **Citation**

If you use this model or code, please cite it as follows: 

Huan Gao, Xianda Zheng, Weizhuo Li, Guilin Qi, and Meng Wang.Cosine-based Embedding for Completing Schematic Knowledge.NlPCC 2019.[[pdf]](https://github.com/zhengxianda/CosE/raw/master/img/paper.pdf) -->