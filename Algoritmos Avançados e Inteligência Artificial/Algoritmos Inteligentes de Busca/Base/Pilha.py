#-*- coding: utf-8 -*-
'''
Curso    : Inteligência Artificial: Algorítmos inteligentes de busca (udemy)
Nome     : Pilha.py
Descrição: Classe Pilha
#Autor   : Leo       Fev/2018
'''

import sys
sys.path.insert(0, '/media/sf_Documents/cursoBuscas/grafocidades/')
#print(sys.path)

from Mapa import Mapa

class Pilha:
    
    def __init__(self, tamanho):
        self.tamanho = tamanho
        self.cidades = [None]*self.tamanho
        self.topo = -1

    def empilhar(self, cidade):
        if not Pilha.pilhaCheia(self):
            self.topo += 1
            self.cidades[self.topo] = cidade
            print('Topo:{} empilhou:{}'.format(self.topo, cidade.getNome()))
        else:
            print("A pilha já está cheia")

    def desempilhar(self):
        if not Pilha.pilhaVazia(self):
            print('desempilhou:{} topo:{}'.format(self.cidades[self.topo].getNome(), self.topo))
            temp = self.cidades[self.topo]
            self.topo -= 1
            return temp
        else:
            print('A pilha já está vazia')
            return None
      
    def getTopo(self):
        return self.cidades[self.topo]
   
    def pilhaVazia(self):
        return (self.topo == -1)

    def pilhaCheia(self):
        return (self.topo == self.tamanho - 1)


if __name__ == '__main__' :
    pilha = Pilha(5)
    mapa = Mapa()
   
    pilha.empilhar(mapa.getPortoUniao())
    pilha.empilhar(mapa.getCampoLargo())
    pilha.empilhar(mapa.getCanoinhas())
    pilha.empilhar(mapa.getCuritiba())
    pilha.empilhar(mapa.getContenda())
    #pilha.empilhar(mapa.getLapa())

    print("Topo da pilha: {}".format(pilha.getTopo().getNome()))
   
    pilha.desempilhar()
    pilha.desempilhar()
    pilha.desempilhar()
    #pilha.desempilhar()
    #pilha.desempilhar()

    if pilha.desempilhar() == None:
        print("Erro pilha vazia")

    print("Topo da pilha: {}".format(pilha.getTopo().getNome()))

'''
$ python Pilha.py
Topo:0 empilhou:Porto União
Topo:1 empilhou:Campo Largo
Topo:2 empilhou:Canoinhas
Topo:3 empilhou:Curitiba
Topo:4 empilhou:Contenda
Topo da pilha: Contenda
desempilhou:Contenda topo:4
desempilhou:Curitiba topo:3
desempilhou:Canoinhas topo:2
desempilhou:Campo Largo topo:1
Topo da pilha: Porto União
'''