#-*- coding: utf-8 -*-
'''
Curso    : Inteligência Artificial: Algorítmos inteligentes de busca (udemy)
Nome     : VetorOrdenadoAdjacente.py
Descrição: VetorOrdenadoAdjacente
#Autor   : Leo       Fev/2018
'''

from __future__ import print_function  

import sys
sys.path.insert(0, '/media/sf_Documents/cursoBuscas/grafocidades/')

class VetorOrdenadoAdjacente:
    
    def __init__(self, tamanho):
        self.numeroElementos = 0
        self.adjacentes = [None]*tamanho

    def inserir(self, adjacente):
        if self.numeroElementos == 0:
            self.adjacentes[0] = adjacente
            self.numeroElementos += 1
            return
        posicao = 0 # Se retirar ocorre UnboundLocalError: local variable 'posicao' referenced before assignment
        i = 0       # idem acima
        while i < self.numeroElementos:
            if adjacente.getDistanciaAEstrela() > self.adjacentes[posicao].getDistanciaAEstrela():
                 posicao += 1
            i += 1
        for k in range(self.numeroElementos, posicao, -1):
            self.adjacentes[k] = self.adjacentes[k -1]
        self.adjacentes[posicao] = adjacente
        self.numeroElementos += 1

    def getPrimeiro(self):
        return self.adjacentes[0].getCidade()

    def mostrar(self):
        for i in range(0, self.numeroElementos):
            print('Adjacente:{} DistanciaAEstrela:{}'.format(self.adjacentes[i].getCidade().getNome(), self.adjacentes[i].getDistanciaAEstrela()))
