import matplotlib.pyplot as plt

import pandas as pd 
import numpy as np

df = pd.read_csv("results_bench.csv")

nvidia_c = df["nvidia_gpu_0"]
cpu_c = df["package_0"]
ram_c = df["dram_0"]

#--------HISTOGRAMA GPU----------------------------
#Range das barras
num = min(nvidia_c)
hist_range=[]
k = 0
while(num<=max(nvidia_c)):
    hist_range.append(num)
    k=k+1
    num +=500
    
#Plot do histograma GPU 
plt.figure(figsize=(20, 10))
plt.title("Histograma consumo energético da GPU")
plt.hist(nvidia_c, bins=hist_range )
plt.ylabel("Numero de Arquiteturas")
plt.xlabel("Consumo energetico(uJ)")

plt.savefig("Histograma_GPU.png")
#--------------------------------------------------


#--------HISTOGRAMA CPU----------------------------
#Range das barras
num = min(cpu_c)
hist_range=[]
k = 0

while(num<=max(cpu_c)):
    hist_range.append(num)
    k=k+1
    num +=10000
 
plt.figure(figsize=(20, 10))
plt.title("Histograma consumo energético da CPU")
plt.hist(cpu_c, bins=hist_range )
plt.ylabel("Numero de Arquiteturas")
plt.xlabel("Consumo energetico(uJ)")
plt.savefig("Histograma_CPU.png")
#--------------------------------------------------

#--------HISTOGRAMA RAM----------------------------
num = min(ram_c)
hist_range=[]
k = 0

while(num<=max(ram_c)):
    hist_range.append(num)
    k=k+1
    num +=1000

 
plt.figure(figsize=(20, 10))
plt.title("Histograma consumo energético da RAM")
plt.hist(ram_c, bins=hist_range )
plt.ylabel("Numero de Arquiteturas")
plt.xlabel("Consumo energetico(uJ)")

plt.savefig("Histograma_RAM.png")
#--------------------------------------------------
