#-*- coding: utf-8 -*-
'''
Curso    : Inteligência Artificial: Algorítmos inteligentes de busca (udemy)
Nome     : Adjacente.py
Descrição: Classe Adjacente
#Autor   : Leo       Fev/2018
'''

from __future__ import print_function  
from Cidade import Cidade

class Adjacente():
      
    def __init__(self, cidade, distancia):
        self.cidade = cidade
        self.distancia = distancia
        # Distância em linha reta + distância pela estrada
        self.distanciaAEstrela = self.cidade.getDistanciaObjetivo() + self.distancia 
        # print('\nAdjacente:{} DistObj:{} DistEstr:{} DistAEstr:{}'.format(self.cidade.getNome(), self.cidade.getDistanciaObjetivo(),
        #                                                                   self.distancia, self.distanciaAEstrela))

    def getCidade(self):
        return self.cidade

    def setCidade(self, cidade):
        self.cidade = cidade  

    def getDistancia(self):
        return self.distancia

    def setDistancia(self, distancia):
        self.distancia = distancia

    def getDistanciaAEstrela(self):
        return self.distanciaAEstrela


