#-*- coding: utf-8 -*-
'''
Curso    : Inteligência Artificial: Algorítmos inteligentes de busca (udemy)
Nome     : Cidade.py
Descrição: Classe Cidade
#Autor   : Leo       Fev/2018
'''

from __future__ import print_function  

class Cidade:
    
    def __init__(self, nome, distanciaObjetivo):
        self.nome = nome    
        self.visitado = False
        self.distanciaObjetivo = distanciaObjetivo
        self.adjacentes = []
        #print('Cidade: {} DistObj: {}'.format(self.nome, self.distanciaObjetivo))

    def addCidadeAdjacente(self, cidade):
        self.adjacentes.append(cidade)
        #print('\nCidade:{} CidadeAdjacente:{}\n adjacentes:{}'.format(self.nome, cidade, self.adjacentes))

    def getNome(self): 
        return self.nome
    
    def setNome(self, nome):
        self.nome = nome

    def isVisitado(self):
        return self.visitado

    def setVisitado(self, visitado):
        self.visitado = visitado

    def getAdjacentes(self):
        return self.adjacentes

    def getDistanciaObjetivo(self):
        return self.distanciaObjetivo

    def setDistanciaObjetivo(self, distanciaObjetivo):
        self.distanciaObjetivo = distanciaObjetivo

