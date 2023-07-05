import itertools
import os
import sys
import time
import torch
import numpy as np

import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import Network
import utils

#Imports para medicao geral e importacao dos dados num fich csv
from pyJoules.energy_meter import measure_energy


#Imports para definicao do GPU a medir
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain, RaplCoreDomain
from pyJoules.device import DeviceFactory

#Import dos objetos de medicao
from pyJoules.energy_meter import EnergyMeter


#Biblioteca para ficheiro CSV
import csv


csvdata={"time": float(), "arch_name": str(), "nvidia_gpu_0": float(), "package_0":float(), "dram_0":float()}
field_names=['time', 'arch_name', 'nvidia_gpu_0', 'package_0', 'dram_0']

domains = [NvidiaGPUDomain(0), RaplCoreDomain(0), RaplDramDomain(0)]
devices = DeviceFactory.create_devices(domains)
meter = EnergyMeter(devices)

#comando para conseguir permissões para aceder aos componentes
os.system('sudo chmod -R a+r /sys/class/powercap/intel-rapl')
#tempo total
tempo_total=0

#Warmup para não introduzir medições erradas
def gpu_warmup():
    print("GPU is warming up")
    # Verifica se a GPU está disponível
    if torch.cuda.is_available():
        # Cria um tensor vazio na GPU
        x = torch.empty((1000, 1000)).cuda()
        # Realiza algumas operações na GPU
        y = torch.matmul(x, x)
        z = torch.sum(y)
        # Transfere o resultado de volta para a CPU (apenas para garantir que o benchmark não inclua a transferência de dados)
        z.cpu()

# Executa o warm-up da GPU
gpu_warmup()
def get_real_arch(arch, stages=[2, 3, 3]):
  arch = list(arch)
  result = ''
  for stage in stages:
    id_num = 0
    for idx in range(stage):
      op = arch.pop(0)
      if idx == 0:
        result += op
        continue
      if op != '0':
        result += op
      else:
        id_num += 1
    result += '0' * id_num
  return result

choices = ['0', '1', '2']
layers = 8

space_size = len(list(itertools.product(*[choices]*layers)))

with open('results_bench.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=field_names)
    writer.writeheader()
    evaluating_archs = {}
    for idx, arch in enumerate(itertools.product(*[choices]*layers)):
        meter.start(tag='')
        arch = ''.join(arch)
        #guardar nome da arquitetura - Foi preciso adicionar um chat para forçar o writer a intrepertar o valor numerico do nome da arquitetura como string 
        csvdata["arch_name"]='a'+str(arch)
        real_arch = get_real_arch(arch)
        if real_arch in evaluating_archs:
            print('Already evaluated.')
            continue
        evaluating_archs[real_arch] = 1
        if os.path.exists(os.path.join('bench-cifar10', '{}.txt'.format(real_arch))):
            print('Already evaluated.')
            continue
        print('Evaluating ({}/{}): {}'.format(idx, space_size, arch))
        #os.system('python3 train.py --arch {}'.format(real_arch))

        # Generate PyTorch arch
        arch = utils.decode_arch(real_arch)
        model = Network(arch, 10)
        model = model.cuda()
        
  
        # Create input to give to the architecture during inference
        # batch_size = 1; 3 channels, 32*32 pixels
        dummy_input = torch.randn(1,3,32,32, dtype=torch.float).cuda()

        # INIT LOGGERS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 100
        timings=np.zeros((repetitions,1))
        # GPU-WARM-UP
        # This removes bad measures that usually happen at the end of a training protocol.
        for _ in range(10):
            _ = model(dummy_input)
        # MEASURE PERFORMANCE
        # TODO: Add energy measurement.
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                # Forward the dummy input through the architecture
                _ = model(dummy_input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        
                
        #Guardar valor do tempo total     
        csvdata["time"]= curr_time
        meter.record(tag='')
        # Print mean inference time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print(mean_syn)
        meter.stop()
        
        #Metodos para guardar os valores no csv
        trace= meter.get_trace() 
        csvdata["nvidia_gpu_0"] = trace._samples[0].energy['nvidia_gpu_0']
        csvdata["package_0"] = trace._samples[0].energy['core_0']
        csvdata["dram_0"] = trace._samples[0].energy['dram_0']
        writer.writerow(csvdata)

print('Evaluation done.')

os.system("python3 Plots_Histograma.py")
os.system("python3 CorCorrelacaoGPU.py")
os.system("python3 SortEnergy.py")
os.system("python3 Correlacao.py")
