import csv 
import json
import pandas as pd
import os
from operator import itemgetter

#Obter a diretoria do ficheiro JSON
d=os.getcwd()
d2 = "/".join(d.rsplit("/", 1)[:-1])
d2 = d2 +'/data/nas-bench-macro_cifar10.json'

with open(d2) as json_file:              
    df = pd.read_csv("results_bench.csv")
    data = json.load(json_file)

    f1 = open("Rank_EnergyConsuptionGPU.txt", "w")
    f2 = open("Rank_EnergyConsuptionCPU.txt", "w")
    f3 = open("Rank_EnergyConsuptionRAM.txt", "w") 
                
#RANK ARQUITETURAS - Consumo energético GPU-----------------------------------------------
    df = df.sort_values('nvidia_gpu_0', ascending=False)
   #Remover o primeiro carater 
    gpu_c_sorted=(df["nvidia_gpu_0"]).tolist()
    arch_sorted_gpu=(df["arch_name"]).tolist()
    time_sorted_gpu=(df["time"]).tolist()
            
                
    f1.write("Posicao/Arquitetura/Consumo(uJ)/Tempo \n")  
    for i in range(len(arch_sorted_gpu)):
        f1.write(""+str(i+1)+" "+arch_sorted_gpu[i]+" "+str(gpu_c_sorted[i])+" "+str(time_sorted_gpu[i])+"\n")
                    
#-----------------------------------------------------------------------------------------

                
#RANK ARQUITETURAS - Consumo energético CPU-----------------------------------------------
    df = df.sort_values('package_0', ascending=False)
    #Remover o primeiro carater 
    cpu_c_sorted=(df["package_0"]).tolist()
    arch_sorted_cpu=(df["arch_name"]).tolist()
    time_sorted_cpu=(df["time"]).tolist()              
                
    f2.write("Posicao/Arquitetura/Consumo(uJ)/Tempo \n")  
    for i in range(len(arch_sorted_cpu)):
        f2.write(""+str(i+1)+" "+arch_sorted_cpu[i]+" "+str(cpu_c_sorted[i])+" "+str(time_sorted_cpu[i])+"\n")
#-----------------------------------------------------------------------------------------
                
                
#RANK ARQUITETURAS - Consumo energético RAM------------------------------------------------
    df = df.sort_values('dram_0', ascending=False)
    #Remover o primeiro carater 
    arch_sorted_ram=(df["arch_name"]).tolist()
    ram_c_sorted=(df["dram_0"]).tolist()
    time_sorted_ram = (df["time"]).tolist()             
                    
    f3.write("Posicao/Arquitetura/Consumo(uJ)/Tempo \n")  
    for i in range(len(arch_sorted_ram)):
        f3.write(""+str(i+1)+" "+arch_sorted_ram[i]+" "+str(ram_c_sorted[i])+" "+str(time_sorted_ram[i])+"\n")     
#----------------------------------------------------------------------------------------- 
                

