class Cidade:
    def __init__(self, nome, distanciaObjetivo):
        self.visitado = False
        self.distanciaObjetivo = distanciaObjetivo
        self.nome = nome    
        self.adjacentes = []
        
    def addCidadeAdjacente(self, cidade):
        self.adjacentes.append(cidade)
    
'''c = Cidade("teste", 10)
c.nome
c.visitado
c.distanciaObjetivo
c.adjacentes'''