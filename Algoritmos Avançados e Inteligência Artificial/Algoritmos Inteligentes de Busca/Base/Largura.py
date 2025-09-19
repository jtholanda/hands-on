#-*- coding: utf-8 -*-
'''
Curso    : Inteligência Artificial: Algorítmos inteligentes de busca (udemy)
Nome     : Largura.py
Descrição: Largura
#Autor   : Leo       Fev/2018
'''

from __future__ import print_function  

import sys
sys.path.insert(0, '/media/sf_Documents/cursoBuscas/grafocidades/')
sys.path.insert(0, '/media/sf_Documents/cursoBuscas/estruturas/')

from Mapa import Mapa
from Fila import Fila

class Largura:

    def __init__(self, inicio, objetivo):
        self.inicio = inicio
        self.inicio.setVisitado(True)
        self.objetivo = objetivo

        self.fronteira = Fila(20)
        self.fronteira.enfileirar(inicio)

        self.achou = False

    def buscar(self):
        print('-----------------------------------')
        primeiro = self.fronteira.getPrimeiro()
        print('Primeiro: {} Objetivo: {}'.format(primeiro.getNome(), self.objetivo.getNome()))
        if primeiro == self.objetivo:
            self.achou = True
            print('--- Achou = True ---')
        else:
            temp = self.fronteira.desenfileirar()
            print('Desenfileirou: {}'.format(temp.getNome()))
            for a in primeiro.getAdjacentes():
                print('Verificando se já visitado: {}'.format(a.getCidade().getNome()))
                if not a.getCidade().isVisitado():
                    a.getCidade().setVisitado(True)
                    self.fronteira.enfileirar(a.getCidade())
            if self.fronteira.getNumeroElementos() > 0:
                Largura.buscar(self)


if __name__ == '__main__' :
    
    mapa = Mapa()

    l = Largura(mapa.getPortoUniao(), mapa.getIrati())

    l.buscar()


'''

$ python Largura.py 
Enfileirar inicio:0 fim:0 numElem:1 elemento:Porto União
-----------------------------------
Primeiro: Porto União Objetivo: Irati
Desenfileirar inicio:1 fim:0 numElem:0 elemento: Porto União
Desenfileirou: Porto União
Verificando se já visitado: Paulo Frontin
Enfileirar inicio:1 fim:1 numElem:1 elemento:Paulo Frontin
Verificando se já visitado: Canoinhas
Enfileirar inicio:1 fim:2 numElem:2 elemento:Canoinhas
Verificando se já visitado: São Mateus do Sul
Enfileirar inicio:1 fim:3 numElem:3 elemento:São Mateus do Sul
-----------------------------------
Primeiro: Paulo Frontin Objetivo: Irati
Desenfileirar inicio:2 fim:3 numElem:2 elemento: Paulo Frontin
Desenfileirou: Paulo Frontin
Verificando se já visitado: Porto União
Verificando se já visitado: Irati
Enfileirar inicio:2 fim:4 numElem:3 elemento:Irati
-----------------------------------
Primeiro: Canoinhas Objetivo: Irati
Desenfileirar inicio:3 fim:4 numElem:2 elemento: Canoinhas
Desenfileirou: Canoinhas
Verificando se já visitado: Porto União
Verificando se já visitado: Três Barras
Enfileirar inicio:3 fim:5 numElem:3 elemento:Três Barras
Verificando se já visitado: Mafra
Enfileirar inicio:3 fim:6 numElem:4 elemento:Mafra
-----------------------------------
Primeiro: São Mateus do Sul Objetivo: Irati
Desenfileirar inicio:4 fim:6 numElem:3 elemento: São Mateus do Sul
Desenfileirou: São Mateus do Sul
Verificando se já visitado: Palmeira
Enfileirar inicio:4 fim:7 numElem:4 elemento:Palmeira
Verificando se já visitado: Irati
Verificando se já visitado: Lapa
Enfileirar inicio:4 fim:8 numElem:5 elemento:Lapa
Verificando se já visitado: Três Barras
Verificando se já visitado: Porto União
-----------------------------------
Primeiro: Irati Objetivo: Irati
--- Achou = True ---

'''
