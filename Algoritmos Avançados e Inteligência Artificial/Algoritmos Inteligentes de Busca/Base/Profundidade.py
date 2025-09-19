#-*- coding: utf-8 -*-
'''
Curso    : Inteligência Artificial: Algorítmos inteligentes de busca (udemy)
Nome     : Profundidade.py
Descrição: Profundidade
#Autor   : Leo       Fev/2018
'''

from __future__ import print_function  

import sys
sys.path.insert(0, '/media/sf_Documents/cursoBuscas/grafocidades/')
sys.path.insert(0, '/media/sf_Documents/cursoBuscas/estruturas/')

from Mapa import Mapa
from Pilha import Pilha

class Profundidade:

    def __init__(self, inicio, objetivo):
        self.inicio = inicio
        self.inicio.setVisitado(True)
        self.objetivo = objetivo

        self.fronteira = Pilha(20)
        self.fronteira.empilhar(inicio)

        self.achou = False

    def buscar(self):
        print('-----------------------------------')
        topo = self.fronteira.getTopo()
        print('Topo: {} Objetivo: {}'.format(topo.getNome(), self.objetivo.getNome()))
        if topo == self.objetivo:
            self.achou = True
            print('--- Achou = True ---')
        else:
            for a in topo.getAdjacentes():
                if not self.achou:
                    print('Verificando se já visitado: {}'.format(a.getCidade().getNome()))
                    if not a.getCidade().isVisitado():
                        a.getCidade().setVisitado(True)
                        self.fronteira.empilhar(a.getCidade())
                        Profundidade.buscar(self)
        print('Desempilhou: {}'.format(self.fronteira.desempilhar().getNome()))


if __name__ == '__main__' :
    
    mapa = Mapa()

    p = Profundidade(mapa.getPortoUniao(), mapa.getIrati())

    p.buscar()


'''

$ python Profundidade.py 
Topo:0 empilhou:Porto União
-----------------------------------
Topo: Porto União Objetivo: Irati
Verificando se já visitado: Paulo Frontin
Topo:1 empilhou:Paulo Frontin
-----------------------------------
Topo: Paulo Frontin Objetivo: Irati
Verificando se já visitado: Porto União
Verificando se já visitado: Irati
Topo:2 empilhou:Irati
-----------------------------------
Topo: Irati Objetivo: Irati
--- Achou = True ---
desempilhou:Irati topo:2
Desempilhou: Irati
desempilhou:Paulo Frontin topo:1
Desempilhou: Paulo Frontin
desempilhou:Porto União topo:0
Desempilhou: Porto União

'''
