import os

d=os.getcwd()
d2 = "/".join(d.rsplit("/", 1)[:-1])
d2 = d2+'/data/nas-bench-macro_cifar10.json'
print(d2)
