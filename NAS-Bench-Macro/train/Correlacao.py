import json 
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as pt
from scipy.stats import kendalltau
#Leitura dos dados retirados do treino
#Arch_name retirado do csv com os resultados
#Para aceder á mean_acc => data[*arch_name*]['mean_acc']

#Obter a diretoria do ficheiro JSON
d=os.getcwd()
d2 = "/".join(d.rsplit("/", 1)[:-1])
d2 = d2 +'/data/nas-bench-macro_cifar10.json'

with open(d2) as json_file:
        data= json.load(json_file)
        df= pd.read_csv('results_bench.csv')
        #Nomes das arquiteturas do JSON
        arch=list(data)
        #Arquiteturas do CSV
        arch_csv=df["arch_name"]
        #Tempos de inferencia
        time_csv=df["time"]
        
        #Retirar o primeiro carater de cada arquitetura, para ficar apenas com valores numericos
        for i in range(len(arch_csv)):
            arch_csv[i] = arch_csv[i][1:]
                
        #Consumo energetico
        nvidia_c = df["nvidia_gpu_0"]
        cpu_c = df["package_0"]
        ram_c = df["dram_0"]
        
        #Array accuracies
        arr_acc=[]
        #Ordenação das Accuracies do JSON
        for i in range(len(arch_csv)):
            for j in range(len(arch)):
                if(arch_csv[i] == arch[j]):
                    arr_acc.append(data[arch[j]]["mean_acc"])
                    
#-------------CORRELACAO CONSUMO/ACCURACY---------------------------------------------        
        #Correlacao GPU/Accuracy
        corr_gpu_acc,_=kendalltau(nvidia_c, arr_acc)
        pt.figure(figsize=(20, 10))
        pt.title("Correlação do consumo energético da GPU com a Accuracy")
        pt.xlabel("Consumo Energético GPU (uJ)")
        pt.ylabel("Precisão média")
        pt.scatter(nvidia_c,arr_acc, color='blue')
        pt.savefig('CorrelacaoGPU_Acc.png')
        print('A correlação da GPU com a Accuracy é: '+str(corr_gpu_acc))
        
        #Correlação CPU/Accuracy
        corr_cpu_acc,_=kendalltau(cpu_c, arr_acc)
        pt.figure(figsize=(20, 10))
        pt.title("Correlação do consumo energético da CPU com a Accuracy")
        pt.xlabel("Consumo Energético CPU (uJ)")
        pt.ylabel("Precisão média")
        pt.scatter(cpu_c,arr_acc, color='blue')
        pt.savefig('CorrelacaoCPU_Acc.png')
        print('A correlação da CPU com a Accuracy é: '+str(corr_cpu_acc))
        
        #Correlação RAM/Accuracy
        corr_ram_acc,_=kendalltau(ram_c, arr_acc)
        pt.figure(figsize=(20, 10))
        pt.title("Correlação do consumo energético da RAM com a Accuracy")
        pt.xlabel("Consumo Energético RAM (uJ)")
        pt.ylabel("Precisão média")
        pt.scatter(ram_c,arr_acc, color='blue')
        pt.savefig('CorrelacaoRAM_Acc.png')
        print('A correlação da RAM com a Accuracy é: '+str(corr_ram_acc))
        
#-------------CORRELACAO CONSUMO/TEMPO DE INFERENCIA---------------------------------------------        
        #Correlacao GPU/Tempo de inferencia
        corr_gpu_time,_=kendalltau(time_csv, nvidia_c)
        pt.figure(figsize=(20, 10))
        pt.title("Correlação do consumo energético da GPU com o tempo de inferência")
        pt.ylabel("Consumo Energético GPU (uJ)")
        pt.xlabel("Tempo de inferencia(seg)")
        pt.scatter(time_csv, nvidia_c,color='blue')
        pt.savefig('CorrelacaoGPU_Inferencia.png')
        print('A correlação da GPU com o tempo de inferencia é: '+str(corr_gpu_time))
        
        #Correlação CPU/Tempo de inferencia
        corr_cpu_time,_=kendalltau(time_csv, cpu_c)
        pt.figure(figsize=(20, 10))
        pt.title("Correlação do consumo energético da CPU com o tempo de inferência")
        pt.ylabel("Consumo Energético CPU (uJ)")
        pt.xlabel("Tempo de inferencia(seg)")
        pt.scatter(time_csv, cpu_c, color='blue')
        pt.savefig('CorrelacaoCPU_Inferencia.png')
        print('A correlação da CPU com a Inferencia é: '+str(corr_cpu_time))
        
        #Correlação RAM/Accuracy
        corr_ram_time,_=kendalltau(time_csv, ram_c)
        pt.figure(figsize=(20, 10))
        pt.title("Correlação do consumo energético da RAM com o tempo de inferência")
        pt.ylabel("Consumo Energético RAM (uJ)")
        pt.xlabel("Tempo de inferencia(seg)")
        pt.scatter( time_csv,ram_c, color='blue')
        pt.savefig('CorrelacaoRAM_Inferencia.png')
        print('A correlação da RAM com a Inferencia é: '+str(corr_ram_time))
        
        
