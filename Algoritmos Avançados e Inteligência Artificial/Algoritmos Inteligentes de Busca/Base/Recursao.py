#-*- coding: utf-8 -*-
'''
Curso    : Inteligência Artificial: Algorítmos inteligentes de busca (udemy)
Nome     : Recursão.py
Descrição: Teste Recursão
#Autor   : Leo       Fev/2018
'''

class Recursao:
    
    def funcao(self, i):
        print('Executando ' + str(i))
        
        if i < 5:
            i += 1
            self.funcao(i)


if __name__ == '__main__' :

    i = 0

    r = Recursao()
    r.funcao(i)
    
