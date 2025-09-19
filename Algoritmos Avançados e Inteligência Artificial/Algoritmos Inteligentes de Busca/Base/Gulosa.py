#-*- coding: utf-8 -*-
'''
Curso    : Inteligência Artificial: Algorítmos inteligentes de busca (udemy)
Nome     : Gulosa.py
Descrição: Gulosa
#Autor   : Leo       Fev/2018
'''

from __future__ import print_function  

import sys
sys.path.insert(0, '/media/sf_Documents/cursoBuscas/grafocidades/')
sys.path.insert(0, '/media/sf_Documents/cursoBuscas/estruturas/')

from Mapa import Mapa
from VetorOrdenado import VetorOrdenado

class Gulosa:

    def __init__(self, objetivo):
        self.objetivo = objetivo

        self.achou = False

    def buscar(self, atual):
        print('-----------------------------------')
        print('\nAtual: {}'.format(atual.getNome()))

        atual.setVisitado(True)

        if atual == self.objetivo:
            self.achou = True
        else:
            self.fronteira = VetorOrdenado(len(atual.getAdjacentes()))
            for a in atual.getAdjacentes():
                if not a.getCidade().isVisitado():
                    a.getCidade().setVisitado(True)
                    self.fronteira.inserir(a.getCidade())

            self.fronteira.mostrar()

            if self.fronteira.getPrimeiro() != None:
                Gulosa.buscar(self, self.fronteira.getPrimeiro())


if __name__ == '__main__' :
    
    mapa = Mapa()

    g = Gulosa(mapa.getCuritiba())

    g.buscar(mapa.getPortoUniao())


'''

$ python Gulosa.py
-----------------------------------

Atual: Porto União
Cidade:São Mateus do Sul  DistanciaObj:123
Cidade:Canoinhas  DistanciaObj:141
Cidade:Paulo Frontin  DistanciaObj:172
-----------------------------------

Atual: São Mateus do Sul
Cidade:Palmeira  DistanciaObj:59
Cidade:Lapa  DistanciaObj:74
Cidade:Três Barras  DistanciaObj:131
Cidade:Irati  DistanciaObj:139
-----------------------------------

Atual: Palmeira
Cidade:Campo Largo  DistanciaObj:27
-----------------------------------

Atual: Campo Largo
Cidade:Curitiba  DistanciaObj:0
Cidade:Balsa Nova  DistanciaObj:41
-----------------------------------

Atual: Curitiba

'''
