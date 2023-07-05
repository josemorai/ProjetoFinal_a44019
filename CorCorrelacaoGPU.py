import json 
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as pt
from matplotlib.lines import Line2D
from scipy.stats import kendalltau
#Leitura dos dados retirados do treino
#Arch_name retirado do csv com os resultados
#Para aceder á mean_acc => data[*arch_name*]['mean_acc']

df= pd.read_csv('results_bench.csv')
#Arquiteturas do CSV
arch_csv=df["arch_name"]
#Tempos de inferencia
time_csv=df["time"]
#Consumo energetico
nvidia_c = df["nvidia_gpu_0"]


#Teste 3 - Se tiver mais blocos 0, vai ter uma cor, se tiver mais blocos 1 tem outra e o mesmo para o bloco 2
       
#Bloco 0
b0_t=[]
b0_c=[]
b0_a=[]
#Bloco 1
b1_t=[]
b1_c=[]
b1_a=[]
#Bloco 2
b2_t=[]
b2_c=[]
b2_a=[]

#Legenda
custom_lines = [Line2D([0], [0], color='red', lw=4),
                Line2D([0], [0], color='blue', lw=4),
                Line2D([0], [0], color='purple', lw=4)]
with open('/home/josemorais/Desktop/NAS-Bench-Macro-master/data/nas-bench-macro_cifar10.json') as json_file:           
    data= json.load(json_file)
    arch=list(data)    
    for i in range(len(arch_csv)):
            arch_csv[i] = arch_csv[i][1:]
    #Array accuracies
    arr_acc=[]
    #Ordenação das Accuracies do JSON
    for i in range(len(arch_csv)):
        for j in range(len(arch)):
            if(arch_csv[i] == arch[j]):
                arr_acc.append(data[arch[j]]["mean_acc"])                
                                
    for i in range(len(arch_csv)):
        a=0
        b=0 
        c=0
        #verificar quantos blocos '0', '1' e '2' há em cada arquitetura
        for j in range(len(arch_csv[i])):
            if arch_csv[i][j] == '0':
                a+=1
            if arch_csv[i][j] == '1':
                b+=1
            if arch_csv[i][j] == '2':
                c+=1
                        
        if(a>=b and a>=c):
            b0_c.append(nvidia_c[i])
            b0_t.append(time_csv[i])
            b0_a.append(arr_acc[i])
        if(b>=a and b>=c):
            b1_c.append(nvidia_c[i])
            b1_t.append(time_csv[i])
            b1_a.append(arr_acc[i])
        if(c>=a and c>=b):
            b2_c.append(nvidia_c[i])
            b2_t.append(time_csv[i])
            b2_a.append(arr_acc[i])



    pt.figure(figsize=(20, 10))
    pt.title("Correlacao do consumo energético da GPU com o tempo de inferencia")
    pt.ylabel("Consumo Energético GPU (uJ)")
    pt.xlabel("Tempo de inferencia(seg)")

    #BLOCO 0
    pt.scatter( b0_t,b0_c, color='red')
    #BLOCO 1
    pt.scatter( b1_t,b1_c, color='blue')
    #BLOCO 2
    pt.scatter( b2_t,b2_c, color='purple')
    pt.legend(custom_lines, ['Bloco 0', 'Bloco 1', 'Bloco 2'])
    pt.savefig('CorrelacaoGPU_Inferencia_Teste.png')


    #Correlação do consumo com a accuracy
    pt.figure(figsize=(20, 10))
    pt.title("Correlacao do consumo energético da GPU com Accuracy")
    pt.ylabel("Consumo Energético GPU (uJ)")
    pt.xlabel("Accuracy")

    #BLOCO 0
    pt.scatter(b0_a, b0_c, color='red')
    #BLOCO 1
    pt.scatter( b1_a,b1_c, color='blue')
    #BLOCO 2
    pt.scatter( b2_a,b2_c, color='purple')
    pt.legend(custom_lines, ['Bloco 0', 'Bloco 1', 'Bloco 2'])
    pt.savefig('CorrelacaoGPU_Accuracy_Teste.png')


        


    """
    #Testes para ver qual a razao dos tres focos na GPU
    #Teste2 - Ver quais as arquiteturas que estão entre os valores dos focos
    #Arq no bloco 0
    b0_a=[]
    #Arq no bloco 1
    b1_a=[]
    #Arq no bloco 2
    b2_a=[]

    for i in range(len(nvidia_c)):
        if(nvidia_c[i]<=7500):
            b0_a.append(arch_csv[i])
        if(nvidia_c[i]>=7500 and nvidia_c[i]<=15000):
            b1_a.append(arch_csv[i])    
        if(nvidia_c[i]>=15000):
            b2_a.append(arch_csv[i])

    print('FOCO 1')
    print(b0_a)
    print('')
    print('FOCO 2')
    print(b1_a)
    print('')
    print('FOCO 2')
    print(b2_a)
    print('')

    #Teste 1 - Verificar se tem a ver com os blocos que estamos a operar. 8 camadas e 3 blocos, ou seja, 00000000-> cada digito camada, o valor de cada digito diz se estamos a trabalhar na camada 0, 1 ou 2         
           
    #Bloco 0
    b0_t=[]
    b0_c=[]
    #Bloco 1
    b1_t=[]
    b1_c=[]
    #Bloco 2
    b2_t=[]
    b2_c=[]                            
    for i in range(len(arch_csv)):
        if(arch_csv[i][-1] == '0'):
            b0_c.append(nvidia_c[i])
            b0_t.append(time_csv[i])
        if(arch_csv[i][-1] == '1'):
            b1_c.append(nvidia_c[i])
            b1_t.append(time_csv[i])
        if(arch_csv[i][-1] == '2'):
            b2_c.append(nvidia_c[i])
            b2_t.append(time_csv[i])




    pt.figure(figsize=(20, 10))
    pt.title("Correlacao do consumo energético da CPU com o tempo de inferencia")
    pt.xlabel("Consumo Energético CPU (uJ)")
    pt.ylabel("Tempo de inferencia(seg)")

    #BLOCO 0
    pt.scatter(b0_c, b0_t, color='red')
    #BLOCO 1
    pt.scatter(b1_c, b1_t, color='blue')
    #BLOCO 2
    pt.scatter(b2_c, b2_t, color='green')
    pt.savefig('CorrelacaoGPU_Inferencia_Teste.png')
    """

            
