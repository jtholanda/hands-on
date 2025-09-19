from Adjacente import Adjacente
from Cidade import Cidade

class Mapa:
    
    portoUniao = Cidade("Porto União", 203)
    pauloFrontin = Cidade("Paulo Frontin", 172)
    canoinhas = Cidade("Canoinhas", 141)
    irati = Cidade("Irati", 139)
    palmeira = Cidade("Palmeira", 59)
    campoLargo = Cidade("Campo Largo", 27)
    curitiba = Cidade("Curitiba", 0)
    balsaNova = Cidade("Balsa Nova", 41)
    araucaria = Cidade("Araucária", 23)
    saoJose = Cidade("São José dos Pinhais", 13)
    contenda = Cidade("Contenda", 39)
    mafra = Cidade("Mafra", 94)
    tijucas = Cidade("Tijucas do Sul", 56)
    lapa = Cidade("Lapa", 74)
    saoMateus = Cidade("São Mateus do Sul", 123)
    tresBarras = Cidade("Três Barras", 131)

    portoUniao.addCidadeAdjacente(Adjacente(pauloFrontin, 46))  
    portoUniao.addCidadeAdjacente(Adjacente(canoinhas, 78))
    portoUniao.addCidadeAdjacente(Adjacente(saoMateus, 87))
       
    pauloFrontin.addCidadeAdjacente(Adjacente(portoUniao, 46))
    pauloFrontin.addCidadeAdjacente(Adjacente(irati, 75))
    
    canoinhas.addCidadeAdjacente(Adjacente(portoUniao, 78))
    canoinhas.addCidadeAdjacente(Adjacente(tresBarras, 12))
    canoinhas.addCidadeAdjacente(Adjacente(mafra, 66))
    
    irati.addCidadeAdjacente(Adjacente(pauloFrontin, 75))
    irati.addCidadeAdjacente(Adjacente(palmeira, 75))
    irati.addCidadeAdjacente(Adjacente(saoMateus, 57))
    
    palmeira.addCidadeAdjacente(Adjacente(irati, 75))
    palmeira.addCidadeAdjacente(Adjacente(saoMateus, 77))
    palmeira.addCidadeAdjacente(Adjacente(campoLargo, 55))
    
    campoLargo.addCidadeAdjacente(Adjacente(palmeira, 55))
    campoLargo.addCidadeAdjacente(Adjacente(balsaNova, 22))
    campoLargo.addCidadeAdjacente(Adjacente(curitiba, 29))
    
    curitiba.addCidadeAdjacente(Adjacente(campoLargo, 29))
    curitiba.addCidadeAdjacente(Adjacente(balsaNova, 51))
    curitiba.addCidadeAdjacente(Adjacente(araucaria, 37))
    curitiba.addCidadeAdjacente(Adjacente(saoJose, 15))
    
    balsaNova.addCidadeAdjacente(Adjacente(curitiba, 51))
    balsaNova.addCidadeAdjacente(Adjacente(campoLargo, 22))
    balsaNova.addCidadeAdjacente(Adjacente(contenda, 19))
    
    araucaria.addCidadeAdjacente(Adjacente(curitiba, 37))
    araucaria.addCidadeAdjacente(Adjacente(contenda, 18))
    
    saoJose.addCidadeAdjacente(Adjacente(curitiba, 15))
    saoJose.addCidadeAdjacente(Adjacente(tijucas, 49))
    
    contenda.addCidadeAdjacente(Adjacente(balsaNova, 19))
    contenda.addCidadeAdjacente(Adjacente(araucaria, 18))
    contenda.addCidadeAdjacente(Adjacente(lapa, 26))

    mafra.addCidadeAdjacente(Adjacente(tijucas, 99))
    mafra.addCidadeAdjacente(Adjacente(lapa, 57))
    mafra.addCidadeAdjacente(Adjacente(canoinhas, 66))
    
    tijucas.addCidadeAdjacente(Adjacente(mafra, 99))
    tijucas.addCidadeAdjacente(Adjacente(saoJose, 49))

    lapa.addCidadeAdjacente(Adjacente(contenda, 26))
    lapa.addCidadeAdjacente(Adjacente(saoMateus, 60))
    lapa.addCidadeAdjacente(Adjacente(mafra, 57))
    
    saoMateus.addCidadeAdjacente(Adjacente(palmeira, 77))
    saoMateus.addCidadeAdjacente(Adjacente(irati, 57))
    saoMateus.addCidadeAdjacente(Adjacente(lapa, 60))
    saoMateus.addCidadeAdjacente(Adjacente(tresBarras, 43))
    saoMateus.addCidadeAdjacente(Adjacente(portoUniao, 87))
    
    tresBarras.addCidadeAdjacente(Adjacente(saoMateus, 43))
    tresBarras.addCidadeAdjacente(Adjacente(canoinhas, 12))
    
'''mapa = Mapa()
mapa.portoUniao.nome
mapa.portoUniao.visitado
mapa.portoUniao.distanciaObjetivo
mapa.portoUniao.adjacentes

for i in range(len(mapa.portoUniao.adjacentes)):
    print(mapa.portoUniao.adjacentes[i].cidade.nome)
    print(mapa.portoUniao.adjacentes[i].distancia)'''
