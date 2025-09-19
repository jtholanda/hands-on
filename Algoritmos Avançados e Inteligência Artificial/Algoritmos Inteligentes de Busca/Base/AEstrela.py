#-*- coding: utf-8 -*-
'''
Curso    : Inteligência Artificial: Algorítmos inteligentes de busca (udemy)
Nome     : AEstrela.py
Descrição: AEstrela
#Autor   : Leo       Fev/2018
'''

from __future__ import print_function  

import sys
sys.path.insert(0, '/media/sf_Documents/cursoBuscas/grafocidades/')
sys.path.insert(0, '/media/sf_Documents/cursoBuscas/estruturas/')

from Mapa import Mapa
from VetorOrdenadoAdjacente import VetorOrdenadoAdjacente

class AEstrela:

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
            self.fronteira = VetorOrdenadoAdjacente(len(atual.getAdjacentes()))
            for a in atual.getAdjacentes():
                if not a.getCidade().isVisitado():
                    a.getCidade().setVisitado(True)
                    self.fronteira.inserir(a)

            self.fronteira.mostrar()

            if self.fronteira.getPrimeiro() != None:
                AEstrela.buscar(self, self.fronteira.getPrimeiro())


if __name__ == '__main__' :
    
    mapa = Mapa()

    g = AEstrela(mapa.getCuritiba())

    g.buscar(mapa.getPortoUniao())


'''
$ python AEstrela.py 
-----------------------------------

Atual: Porto União
Adjacente:São Mateus do Sul DistanciaAEstrela:210
Adjacente:Paulo Frontin DistanciaAEstrela:218
Adjacente:Canoinhas DistanciaAEstrela:219
-----------------------------------

Atual: São Mateus do Sul
Adjacente:Lapa DistanciaAEstrela:134
Adjacente:Palmeira DistanciaAEstrela:136
Adjacente:Três Barras DistanciaAEstrela:174
Adjacente:Irati DistanciaAEstrela:196
-----------------------------------

Atual: Lapa
Adjacente:Contenda DistanciaAEstrela:65
Adjacente:Mafra DistanciaAEstrela:151
-----------------------------------

Atual: Contenda
Adjacente:Araucária DistanciaAEstrela:41
Adjacente:Balsa Nova DistanciaAEstrela:60
-----------------------------------

Atual: Araucária
Adjacente:Curitiba DistanciaAEstrela:37
-----------------------------------

Atual: Curitiba

'''
