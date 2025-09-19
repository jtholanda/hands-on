#-*- coding: utf-8 -*-
'''
Curso    : Inteligência Artificial: Algorítmos inteligentes de busca (udemy)
Nome     : Fila.py
Descrição: Classe Fila
#Autor   : Leo       Fev/2018
'''

from __future__ import print_function  

import sys
sys.path.insert(0, '/media/sf_Documents/cursoBuscas/grafocidades/')

from Mapa import Mapa

class Fila:

    def __init__(self, tamanho):
        self.tamanho = tamanho
        self.cidades = [None]*self.tamanho
        self.inicio = 0
        self.fim = -1
        self.numeroElementos = 0

    def enfileirar(self, cidade):
        if not Fila.filaCheia(self):
            if self.fim == self.tamanho -1:
                self.fim = -1
            self.fim += 1
            self.cidades[self.fim] = cidade 
            self.numeroElementos += 1
            print('Enfileirar inicio:{} fim:{} numElem:{} elemento:{}'.format(self.inicio, self.fim,
                                                                              self.numeroElementos, cidade.getNome()))
        else:
            print("A fila já está cheia")        

    def desenfileirar(self):
        if not Fila.filaVazia(self):
            temp = self.cidades[self.inicio]
            self.inicio += 1
            if self.inicio == self.tamanho:
                self.inicio = 0
            self.numeroElementos -= 1
            print('Desenfileirar inicio:{} fim:{} numElem:{} elemento: {}'.format(self.inicio, self.fim, 
                                                                                  self.numeroElementos, temp.getNome()))
            return temp
        else:
            print('A fila já está vazia')
            return None

    def getPrimeiro(self):
        return self.cidades[self.inicio]

    def filaVazia(self):
        return self.numeroElementos == 0

    def filaCheia(self):
        return self.numeroElementos == self.tamanho

    def getNumeroElementos(self):
        return self.numeroElementos


if __name__ == '__main__' :

    fila = Fila(5)
    mapa = Mapa()

    fila.enfileirar(mapa.getAraucaria())
    fila.enfileirar(mapa.getBalsaNova())
    fila.enfileirar(mapa.getContenda())
    
    print('Primeiro da fila: {}'.format(fila.getPrimeiro().getNome()))
    
    fila.desenfileirar()
    fila.desenfileirar()
    
    fila.enfileirar(mapa.getCanoinhas())
    fila.enfileirar(mapa.getIrati())
    fila.enfileirar(mapa.getPalmeira())
    fila.enfileirar(mapa.getPortoUniao())
  
    print('Primeiro da fila: {}'.format(fila.getPrimeiro().getNome()))

'''

$ python Fila.py
Enfileirar inicio:0 fim:0 numElem:1 elemento:Araucária
Enfileirar inicio:0 fim:1 numElem:2 elemento:Balsa Nova
Enfileirar inicio:0 fim:2 numElem:3 elemento:Contenda
Primeiro da fila: Araucária
Desenfileirar inicio:1 fim:2 numElem:2 elemento: Araucária
Desenfileirar inicio:2 fim:2 numElem:1 elemento: Balsa Nova
Enfileirar inicio:2 fim:3 numElem:2 elemento:Canoinhas
Enfileirar inicio:2 fim:4 numElem:3 elemento:Irati
Enfileirar inicio:2 fim:0 numElem:4 elemento:Palmeira
Enfileirar inicio:2 fim:1 numElem:5 elemento:Porto União
Primeiro da fila: Contenda


'''