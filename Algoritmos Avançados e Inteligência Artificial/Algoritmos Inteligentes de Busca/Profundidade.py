from Mapa import Mapa
from Pilha import Pilha

class Profundidade:

    def __init__(self, inicio, objetivo):
        self.inicio = inicio
        self.inicio.visitado = 's'
        self.objetivo = objetivo

        self.fronteira = Pilha(20)
        self.fronteira.empilhar(inicio)

        # depois
        self.achou = 'n'

    def buscar(self):
        print('-----------------------------------')
        topo = self.fronteira.getTopo()
        print('Topo: {}'.format(topo.nome))
        
        # depois
        if topo == self.objetivo:
            self.achou = 's'
        else:
            for a in topo.adjacentes:
                # depois
                if self.achou == 'n':
                    print('Verificando se já visitado: {}'.format(a.cidade.nome))
                    if a.cidade.visitado == 'n':
                        a.cidade.visitado = 's'
                        self.fronteira.empilhar(a.cidade)
                        Profundidade.buscar(self)
        print('Desempilhou: {}'.format(self.fronteira.desempilhar().nome))
   
mapa = Mapa()
profundidade = Profundidade(mapa.portoUniao, mapa.curitiba)
profundidade.buscar()
