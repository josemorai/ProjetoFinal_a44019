import json 
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as pt
from matplotlib.lines import Line2D

#Leitura dos dados retirados do treino
#Arch_name retirado do csv com os resultados
#Para aceder á mean_acc => data[*arch_name*]['mean_acc']

#Obter a diretoria do ficheiro JSON
d=os.getcwd()
d2 = "/".join(d.rsplit("/", 1)[:-1])
d2 = d2 +'/data/nas-bench-macro_cifar10.json'

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

#Legenda dos gráficos
custom_lines = [Line2D([0], [0], color='red', lw=4),
                Line2D([0], [0], color='blue', lw=4),
                Line2D([0], [0], color='purple', lw=4)]
#Neste passo é necessário associar as accuracies presentes no ficheiro JSON com as arquiteturas que sao retiradas do CSV                
with open(d2) as json_file:           
    data= json.load(json_file)
    arch=list(data)    
    #Retirar o carater 'a' que foi usado para forçar a ler o nome da arquitetura como string 
    for i in range(len(arch_csv)):
            arch_csv[i] = arch_csv[i][1:]
    #Array accuracies
    arr_acc=[]
    #Ordenação das Accuracies do JSON
    for i in range(len(arch_csv)):
        for j in range(len(arch)):
            if(arch_csv[i] == arch[j]):
                arr_acc.append(data[arch[j]]["mean_acc"])                
    #Este passo serviu para agrupar as arquiteturas de acordo com a maneira de como estão codificadas.
    #É contado o numero de camadas que estão codificadas com '0', '1' e '2'. 
    #Depois os seus valores de medição energética, accuracy e tempo de inferencia são agrupados em listas.
    #A lista b0 corresponde ás arquiteturas que possuem a maioria das camadas codificadas com '0'
    #A lista b1 corresponde ás arquiteturas que possuem a maioria das camadas codificadas com '1'
    #A lista b2 corresponde ás arquiteturas que possuem a maioria das camadas codificadas com '2'                             
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


    #Atribui-se diferentes cores ás arquiteturas anteriormente agrupadas
    #Correlação do consumo energético como tempo de inferência
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
    pt.savefig('CorrelacaoGPU_Inferencia_Cor.png')


    #Correlação do consumo com a accuracy
    pt.figure(figsize=(20, 10))
    pt.title("Correlacao do consumo energético da GPU com Accuracy")
    pt.xlabel("Consumo Energético GPU (uJ)")
    pt.ylabel("Accuracy")
    
    #BLOCO 0
    pt.scatter( b0_c, b0_a, color='red')
    #BLOCO 1
    pt.scatter( b1_c, b1_a, color='blue')
    #BLOCO 2
    pt.scatter(b2_c, b2_a, color='purple')
    pt.legend(custom_lines, ['Bloco 0', 'Bloco 1', 'Bloco 2'])
    pt.savefig('CorrelacaoGPU_Accuracy_Cor.png')
          
