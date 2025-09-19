def funcao(i):
    
    print('Executando ')
    if i < 5:
        i += 1
        funcao(i)
        
i = 1
funcao(i)


