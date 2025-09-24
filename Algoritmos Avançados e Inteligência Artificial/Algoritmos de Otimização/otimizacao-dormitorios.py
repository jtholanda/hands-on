import otimizacao_voos  # Importa o módulo com algoritmos de otimização (randomica, subida, tempera, genetico)

# Lista de dormitórios disponíveis
dormitorios = ['São Paulo', 'Flamengo', 'Coritiba', 'Cruzeiro', 'Fortaleza']

# Lista de preferências de cada pessoa
# Cada tupla contém o nome e uma tupla com duas opções de dormitório (primeira e segunda preferência)
preferencias=[('Amanda', ('Cruzeiro', 'Coritiba')),
              ('Pedro', ('São Paulo', 'Fortaleza')),
              ('Marcos', ('Flamengo', 'São Paulo')),
              ('Priscila', ('São Paulo', 'Fortaleza')),
              ('Jessica', ('Flamengo', 'Cruzeiro')), 
              ('Paulo', ('Coritiba', 'Fortaleza')), 
              ('Fred', ('Fortaleza', 'Flamengo')), 
              ('Suzana', ('Cruzeiro', 'Coritiba')), 
              ('Laura', ('Cruzeiro', 'Coritiba')), 
              ('Ricardo', ('Coritiba', 'Flamengo'))]

# Define o domínio de cada posição da solução
# Cada posição pode variar de 0 até o número de vagas restantes (2 vagas por dormitório)
dominio = [(0, (len(dormitorios) * 2) - i - 1) for i in range(0, len(dormitorios) * 2)] 

# Função para imprimir a alocação das pessoas nos dormitórios
def imprimir_solucao(solucao):
    solucao = [6,1,2,1,2,0,2,2,0,0]  # Aqui sobrescreve a solução para teste/debug
    vagas = []  # Lista que representa cada vaga disponível
    for i in range(len(dormitorios)):
        vagas += [i, i]  # Cada dormitório tem 2 vagas, então adiciona duas vezes
    
    for i in range(len(solucao)):
        atual = solucao[i]  # Índice da vaga alocada para a pessoa i
        dormitorio = dormitorios[vagas[atual]]  # Converte índice da vaga em nome do dormitório
        print(preferencias[i][0], dormitorio)  # Imprime nome da pessoa e dormitório alocado
        del vagas[atual]  # Remove a vaga usada para não ser reutilizada

# Teste de impressão com solução fixa
imprimir_solucao([6,1,2,1,2,0,2,2,0,0])

# Função de custo: avalia o "desconforto" de uma solução
def funcao_custo(solucao):
    custo = 0  # Inicializa custo
    vagas = [0,0,1,1,2,2,3,3,4,4]  # Lista de todas as vagas disponíveis (2 por dormitório)
    
    for i in range(len(solucao)):
        atual = solucao[i]  # Índice da vaga alocada
        dormitorio = dormitorios[vagas[atual]]  # Nome do dormitório correspondente
        preferencia = preferencias[i][1]  # Tupla com as duas preferências da pessoa
        
        if preferencia[0] == dormitorio:  # Primeira preferência
            custo += 0
        elif preferencia[1] == dormitorio:  # Segunda preferência
            custo += 1
        else:  # Nenhuma das preferências
            custo += 3
        
        del vagas[atual]  # Remove a vaga usada
    
    return custo  # Retorna o custo total da solução

# Calcula custo de uma solução de teste
funcao_custo([6,1,2,1,2,0,2,2,0,0])

# Executa busca aleatória (randomica)
solucao_randomica = otimizacao_voos.pesquisa_randomica(dominio, funcao_custo)
custo_randomica = funcao_custo(solucao_randomica)  # Calcula custo da solução
imprimir_solucao(solucao_randomica)  # Imprime solução

# Executa subida de encosta (hill climbing)
solucao_subida_encosta = otimizacao_voos.subida_encosta(dominio, funcao_custo)
custo_subida_encosta = funcao_custo(solucao_subida_encosta)
imprimir_solucao(solucao_subida_encosta)

# Executa tempera simulada (simulated annealing)
solucao_tempera = otimizacao_voos.tempera_simulada(dominio, funcao_custo)
custo_tempera = funcao_custo(solucao_tempera)
imprimir_solucao(solucao_tempera) 

# Executa algoritmo genético
solucao_genetico = otimizacao_voos.genetico(dominio, funcao_custo)
custo_genetico = funcao_custo(solucao_genetico)
imprimir_solucao(solucao_genetico)
