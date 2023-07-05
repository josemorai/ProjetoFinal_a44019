import csv 
import json
import pandas as pd
from operator import itemgetter
acc_dict = {'arch': str() , 'acc': float()}

with open('/home/josemorais/Desktop/NAS-Bench-Macro-master/data/nas-bench-macro_cifar10.json') as json_file:              
    df = pd.read_csv("results_bench.csv")
    data = json.load(json_file)
    #----------------------------------------SORT ACCURACY------------------------------------
    arch_csv=df["arch_name"]
    #Retirar o primeiro carater de cada arquitetura, para ficar apenas com valores numericos
    arch=list(data)
    for i in range(len(arch_csv)):
        arch_csv[i] = arch_csv[i][1:]
    print('Archs JSON \n')
    print(arch)
    
    print('Archs CSV \n')
    print(arch_csv)
    
    """
    #Array accuracies
    list_dict=[]
    f4.write("Arquitetura/Accuracy")
    #Ordenação das Accuracies do JSON
    for i in range(len(arch_csv)):
        for j in range(len(arch)):
            if(arch_csv[i] == arch[j]):
                acc_dict['arch']= arch[j]
                acc_dict['acc'] = data[arch[j]]["mean_acc"]
                list_dict.append(acc_dict)
          
   
    sorted_list_dict=[]
    for i in range(len(list_dict)):
        for j in range(len(list_dict)):
            max_arch=list_dict[i]
            if(list_dict['acc'][j]>list_dict['acc'][i]):
                max_arch=list_dict[j]
        sorted_list_dict.append(max_arch)  
    """     
    
    
    f1 = open("Rank_EnergyConsuptionGPU.txt", "w")
    f2 = open("Rank_EnergyConsuptionCPU.txt", "w")
    f3 = open("Rank_EnergyConsuptionRAM.txt", "w") 
    f4 = open("Rank_Accuracy.txt", "w") 
                
#RANK ARQUITETURAS - Consumo energético GPU-----------------------------------------------
    df = df.sort_values('nvidia_gpu_0', ascending=False)
   #Remover o primeiro carater 
    gpu_c_sorted=(df["nvidia_gpu_0"]).tolist()
    arch_sorted_gpu=(df["arch_name"]).tolist()
    time_sorted_gpu=(df["time"]).tolist()
        
    for i in range(len(arch_sorted_gpu)):
        arch_sorted_gpu[i]=arch_sorted_gpu[i][1:]    
                
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
    for i in range(len(arch_sorted_cpu)):
       arch_sorted_cpu[i]=arch_sorted_cpu[i][1:]    
                
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
    for i in range(len(arch_sorted_ram)):
        arch_sorted_ram[i]=arch_sorted_ram[i][1:]  
                    
    f3.write("Posicao/Arquitetura/Consumo(uJ)/Tempo \n")  
    for i in range(len(arch_sorted_ram)):
        f3.write(""+str(i+1)+" "+arch_sorted_ram[i]+" "+str(ram_c_sorted[i])+" "+str(time_sorted_ram[i])+"\n")     
#----------------------------------------------------------------------------------------- 
                

