from VetorOrdenadoAdjacente import VetorOrdenadoAdjacente

class AEstrela:
    def __init__(self, objetivo):
        self.objetivo = objetivo
        self.achou = False

    def buscar(self, atual):
        print('-----------------------------------')
        print('\nAtual: {}'.format(atual.nome))
        atual.visitado = True

        if atual == self.objetivo:
            self.achou = True
        else:
            self.fronteira = VetorOrdenadoAdjacente(len(atual.adjacentes))
            for a in atual.adjacentes:
                if a.cidade.visitado == False:
                    a.cidade.visitado = True
                    self.fronteira.inserir(a)
            self.fronteira.mostrar()
            if self.fronteira.getPrimeiro() != None:
                AEstrela.buscar(self, self.fronteira.getPrimeiro())
                
from Mapa import Mapa
mapa = Mapa()
aestrela = AEstrela(mapa.curitiba)
aestrela.buscar(mapa.portoUniao)