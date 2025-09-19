#-*- coding: utf-8 -*-
'''
Curso    : Inteligência Artificial: Algorítmos inteligentes de busca (udemy)
Nome     : VetorOrdenado.py
Descrição: VetorOrdenado
#Autor   : Leo       Fev/2018
'''

from __future__ import print_function  

import sys
sys.path.insert(0, '/media/sf_Documents/cursoBuscas/grafocidades/')

from Mapa import Mapa

class VetorOrdenado:
    
    def __init__(self, tamanho):
        self.numeroElementos = 0
        self.cidades = [None]*tamanho

    def inserir(self, cidade):
        #print('-------------------------------------')
        if self.numeroElementos == 0:
            self.cidades[0] = cidade
            self.numeroElementos = 1
            return
        posicao = 0 # Se retirar ocorre UnboundLocalError: local variable 'posicao' referenced before assignment
        i = 0       # idem acima
        while i < self.numeroElementos:
            if cidade.getDistanciaObjetivo() > self.cidades[posicao].getDistanciaObjetivo():
                 posicao += 1
            i += 1
        #print('numElem:{} posicao:{}'.format(self.numeroElementos, posicao))
        for k in range(self.numeroElementos, posicao, -1):
             #print('k: {}'.format(k))
             self.cidades[k] = self.cidades[k -1]
        self.cidades[posicao] = cidade
        self.numeroElementos += 1
        # for i in range(0, self.numeroElementos):
        #     print('i:{} Cidade:{}  DistanciaObj:{}'.format(i, self.cidades[i].getNome(), self.cidades[i].getDistanciaObjetivo()))

    def getPrimeiro(self):
        return self.cidades[0]

    def mostrar(self):
        for i in range(0, self.numeroElementos):
            print('Cidade:{}  DistanciaObj:{}'.format(self.cidades[i].getNome(), self.cidades[i].getDistanciaObjetivo()))
       
if __name__ == '__main__' :
    
    mapa = Mapa()
    vetor = VetorOrdenado(9)
    #'''
    vetor.inserir(mapa.getPortoUniao())   # 203
    vetor.inserir(mapa.getPauloFrontin()) # 172
    vetor.inserir(mapa.getBalsaNova())    #  41
    vetor.inserir(mapa.getPalmeira())     #  59
    '''
    vetor.inserir(mapa.getLapa())         #  74
    vetor.inserir(mapa.getTresBarras())   # 131
    vetor.inserir(mapa.getIrati())        # 139
    vetor.inserir(mapa.getPalmeira())     #  59
    '''

    vetor.mostrar()

'''
$ python VetorOrdenado.py
Cidade:Balsa Nova  DistanciaObj:41
Cidade:Palmeira  DistanciaObj:59
Cidade:Paulo Frontin  DistanciaObj:172
Cidade:Porto União  DistanciaObj:203
'''

'''
$ python VetorOrdenado.py
Cidade:Palmeira  DistanciaObj:59
Cidade:Lapa  DistanciaObj:74
Cidade:Três Barras  DistanciaObj:131
Cidade:Irati  DistanciaObj:139
'''
