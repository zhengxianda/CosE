# test
需要高博过目的代码：./models/Our.py

其中主要是"def _calc"函数：

#def _calc
def _calc(self, h, t, r)，sub三元组的r是0，dis三元组的r是1

r.to(torch.float32) 是{sub=0.0,dis=1.0}

(r.to(torch.float32)-0.5)*2 是{sub=-1.0,dis=1.0}

1.0 + (r.to(torch.float32)-0.5) * 2 * torch.cosine_similarity(h, t)
对应的是

sub：1-cos(h,r)

dis：1+cos(h,r)

其中h,t是TransD中经过投影矩阵相乘之后的h,t


