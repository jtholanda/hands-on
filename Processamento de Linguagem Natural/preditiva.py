import pandas as pd
import numpy as np
from math import sqrt
from IPython.display import display
import pandas_profiling 
import datetime

import sweetviz as sv
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf

# Métricas de Desempenho
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import plot_confusion_matrix
from scipy.stats import ks_2samp

import matplotlib
matplotlib.use('module://ipykernel.pylab.backend_inline')

def log_message(message):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S - "), message)


def gera_relatorios_aed(df, target_feat=None, 
                        html_pp='base_aed_pp.html', 
                        html_sv='base_aed_sv.html'):
    '''
    '''
    # Gera relatório usando Pandas Profiling
    perfil_pp = df.profile_report()
    perfil_pp.to_file(output_file=html_pp)
    
    # Gera relatório usando SweetViz
    perfil_sv = sv.analyze(df, target_feat=target_feat)
    perfil_sv.show_html(html_sv)
    
    return perfil_pp, perfil_sv
    

class analise_iv:
        
    # função private
    def __get_tab_bivariada(self, var_escolhida):
     
        # Cria a contagem de Target_1 e Target_0
        df_aux = self.df.copy() 
        df_aux['target2'] = self.df[self.target]
        df2 = df_aux.pivot_table(values='target2',
                                 index=var_escolhida,
                                 columns=self.target,
                                 aggfunc='count')
        
        df2 = df2.rename(columns={0:'#Target_0',
                                  1:'#Target_1'})
        df2.fillna(0, inplace=True)

        # Cria as demais colunas da tabela bivariada
        df2['Total'] = (df2['#Target_0'] + df2['#Target_1'])
        df2['%Freq'] = (df2['Total'] / (df2['Total'].sum()) * 100).round(decimals=2)
        df2['%Target_1'] = (df2['#Target_1'] / (df2['#Target_1'].sum()) * 100).round(decimals=2)
        df2['%Target_0'] = (df2['#Target_0'] / (df2['#Target_0'].sum()) * 100).round(decimals=2)
        df2['%Target_0'] = df2['%Target_0'].apply(lambda x: 0.01 if x == 0 else x) #corrige problema do log indeterminado
        df2['%Taxa_de_Target_1'] = (df2['#Target_1'] / df2['Total'] * 100).round(decimals=2)
        df2['Odds'] = (df2['%Target_1'] / df2['%Target_0']).round(decimals=2)
        df2['Odds'] = df2.Odds.apply(lambda x: 0.01 if x == 0 else x) #corrige problema do log indeterminado
        df2['LN(Odds)'] = np.log(df2['Odds']).round(decimals=2)
        df2['IV'] = (((df2['%Target_1'] / 100 - df2['%Target_0'] / 100) * df2['LN(Odds)'])).round(decimals=2)
        df2['IV'] = np.where(df2['Odds'] == 0.01, 0 , df2['IV']) 

        df2 = df2.reset_index()
        df2['Variavel'] = var_escolhida
        df2 = df2.rename(columns={var_escolhida: 'Var_Range'})
        df2 = df2[['Variavel','Var_Range', '#Target_1','#Target_0', 'Total', '%Freq', '%Target_1', '%Target_0',
       '%Taxa_de_Target_1', 'Odds', 'LN(Odds)', 'IV']]
        
        # Guarda uma cópia da tabela no histórico
        self.df_tabs_iv = pd.concat([self.df_tabs_iv, df2], axis = 0)
        
        return df2
        
    def get_bivariada(self, var_escolhida='all_vars'):
        
        if var_escolhida == 'all_vars':
                       
            #vars = self.df.drop(self.target,axis = 1).columns
            vars = self.get_lista_iv().index
            for var in vars:
                tabela = self.df_tabs_iv[self.df_tabs_iv['Variavel'] == var]
                print('==> "{}" tem IV de {}'.format(var,self.df_tabs_iv[self.df_tabs_iv['Variavel'] == var]['IV'].sum().round(decimals=2)))
                # printa a tabela no Jupyter
                display(tabela)
            
            return
        
        else:
            print('==> "{}" tem IV de {}'.format(var_escolhida,self.df_tabs_iv[self.df_tabs_iv['Variavel'] == var_escolhida]['IV'].sum().round(decimals=2)))
            return self.df_tabs_iv[self.df_tabs_iv['Variavel'] == var_escolhida]
                   
            
    def get_lista_iv(self):
        
    
        # agrupa a lista de IV's em ordem descrescente
        lista = (self.df_tabs_iv.groupby('Variavel').agg({'IV':'sum'})).sort_values(by=['IV'],ascending=False)
            
        return lista
    
    

    def __init__(self, df, target, nbins=10):

        self.df = df.copy()
        self.target = target

        #lista de variaveis numericas
        df_num = self.df.loc[:,((self.df.dtypes == 'int32') | 
                                (self.df.dtypes == 'int64') | 
                                (self.df.dtypes == 'float64')
                               )
                            ]

        vars = df_num.drop(target,axis = 1).columns

        for var in vars:
            nome_var = 'fx_' + var 
            df_num[nome_var] = pd.qcut(df_num[var], 
                                       q=nbins, 
                                       precision=2,
                                       duplicates='drop')
            df_num = df_num.drop(var, axis = 1)
            df_num = df_num.rename(columns={nome_var: var})

        #lista de variaveis qualitativas
        df_str = self.df.loc[:,((self.df.dtypes == 'object') | 
                                (self.df.dtypes == 'category') |
                                (self.df.dtypes == 'bool'))]


        self.df = pd.concat([df_num,df_str],axis = 1)


         # inicializa tab historica
        self.df_tabs_iv = pd.DataFrame()

        vars = self.df.drop(self.target,axis = 1).columns
        for var in vars:
            self.__get_tab_bivariada(var);

        # remove tabs de iv duplicadas
        self.df_tabs_iv = self.df_tabs_iv.drop_duplicates(subset=['Variavel','Var_Range'], keep='last')
        
        
# Função para cálculo do KS
def ks_stat(y, y_pred):
    return ks_2samp(y_pred[y==1], y_pred[y!=1]).statistic

# Função para cálculo do desempenho de modelos
def calcula_desempenho(modelo, x_train, y_train, x_test, y_test):
    
    if isinstance(modelo, tf.keras.Model):
        ypred_train = (modelo.predict(x_train) > 0.5).astype("int32")
        ypred_proba_train = modelo.predict(x_train)[:,0]
        
        ypred_test = (modelo.predict(x_test) > 0.5).astype("int32")
        ypred_proba_test = modelo.predict(x_test)[:,0]
        
    else:
        try:
            # Cálculo dos valores preditos
            ypred_train = modelo.predict(x_train)
            ypred_proba_train = modelo.predict_proba(x_train)[:,1]

            ypred_test = modelo.predict(x_test)
            ypred_proba_test = modelo.predict_proba(x_test)[:,1]
            
        except Exception as e:
            print(f'Modelo não suportado: {e}')
            return None

    # Métricas de Desempenho
    acc_train = accuracy_score(y_train, ypred_train)
    acc_test = accuracy_score(y_test, ypred_test)
    
    roc_train = roc_auc_score(y_train, ypred_proba_train)
    roc_test = roc_auc_score(y_test, ypred_proba_test)
    
    ks_train = ks_stat(y_train, ypred_proba_train)
    ks_test = ks_stat(y_test, ypred_proba_test)
    
    prec_train = precision_score(y_train, ypred_train, zero_division=0)
    prec_test = precision_score(y_test, ypred_test, zero_division=0)
    
    recl_train = recall_score(y_train, ypred_train)
    recl_test = recall_score(y_test, ypred_test)
    
    f1_train = f1_score(y_train, ypred_train)
    f1_test = f1_score(y_test, ypred_test)

    df_desemp = pd.DataFrame({'Treino':[acc_train, roc_train, ks_train, 
                                        prec_train, recl_train, f1_train],
                              'Teste':[acc_test, roc_test, ks_test,
                                       prec_test, recl_test, f1_test]},
                            index=['Acurácia','AUROC','KS',
                                   'Precision','Recall','F1'])
    
    df_desemp['Variação'] = round(df_desemp['Teste'] / df_desemp['Treino'] - 1, 2)
    
    return df_desemp


def matriz_confusao(modelo, X_train, y_train, X_test, y_test):
    
    plot_confusion_matrix(modelo, X_train, y_train)
    plt.title('Treino')    
    plot_confusion_matrix(modelo, X_test, y_test)
    plt.title('Teste')
    
    

# Função para cálculo do desempenho de modelos
def calcula_desempenho_reg(modelo, x_train, y_train, x_test, y_test, y_scaler, historico=None):

    # Calcula valores preditos pelo modelo
    if historico is not None:
        rmse_hist = historico.history[list(historico.history.keys())[1]]
        val_rmse_hist = historico.history[list(historico.history.keys())[3]]
        
    else:
        rmse_hist = modelo.history_[list(modelo.history_.keys())[1]]
        val_rmse_hist = modelo.history_[list(modelo.history_.keys())[3]]

    y_train_pred = y_scaler.inverse_transform(modelo.predict(x_train))
    y_test_pred  = y_scaler.inverse_transform(modelo.predict(x_test))
    
    # Métricas de Desempenho
    r2_train = r2_score(y_train, y_train_pred)
    r2_test =  r2_score(y_test, y_test_pred)

    rmse_train = sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test  = sqrt(mean_squared_error(y_test,  y_test_pred))
    
    df_desemp = pd.DataFrame({'Treino':[r2_train, rmse_train],
                              'Teste':[r2_test, rmse_test]
                             },
                            index=['R²','RMSE'])
    
    df_desemp['Treino'] = df_desemp['Treino'].round(2)
    df_desemp['Teste'] = df_desemp['Teste'].round(2)
    df_desemp['Variação'] = round(df_desemp['Teste'] / df_desemp['Treino'] - 1, 2)
    
    
    # Gráfico com a evolução do treinamento
    matplotlib.use('module://ipykernel.pylab.backend_inline')

    fig, ax = plt.subplots(figsize=(8, 8))    
    plt.plot(rmse_hist)
    plt.plot(val_rmse_hist)
    plt.title('RMSE do Modelo')
    plt.ylabel('RMSE')
    plt.xlabel('Épocas')
    plt.legend(['Treino', 'Validação'], loc='upper left')
    plt.show()
    
    
    # Dispersão dos valores Observados vs. Preditos
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), sharey=True)
    fig.suptitle('Valores Observados vs. Predições', fontsize = 16)
    min_val = min([np.min(y_train.min()),y_train_pred.min()])*0.5
    max_val = max([np.max(y_train.max()),y_train_pred.max()])*1.1

    axs[0].plot(y_train, y_train_pred, 'ro');
    axs[0].plot([y_train.min()[0], y_train_pred.max()], [y_train.min()[0], y_train_pred.max()], 'k--', lw=1)
    axs[0].set_xlim([min_val, max_val])
    axs[0].set_ylim([min_val, max_val])

    axs[1].plot(y_test, y_test_pred, 'ro');
    axs[1].plot([y_test.min()[0], y_test_pred.max()], [y_test.min()[0], y_test_pred.max()], 'k--', lw=1)
    axs[1].set_xlim([min_val, max_val])
    axs[1].set_ylim([min_val, max_val])

    for ax in axs.flat:
        ax.set(xlabel='Valores Observados', ylabel='Valore Preditos')
    plt.show()
    
    return df_desemp


def dispersao_modelo(y_obs, y_pred):
    matplotlib.use('module://ipykernel.pylab.backend_inline')

    plt.style.use('ggplot')
    plt.rc('xtick', labelsize=10) 
    plt.rc('ytick', labelsize=10) 
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    min_val = min([np.min(y_obs.min()),y_pred.min()])*0.5
    max_val = max([np.max(y_obs.max()),y_pred.max()])*1.1
    
    plt.plot(y_obs, y_pred, 'ro')
    plt.xlabel('Observados', fontsize = 10)
    plt.ylabel('Preditos', fontsize = 10)    
    plt.title('Predições vs. Observados', fontsize = 10)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    plt.show()
    
    
def cria_grafico_var_qualitativa(tab):

    # Aumenta o tamanho do gráfico (largura 8 e altura 4)
    fig = plt.figure(figsize=(8,4))

    # Cria um gráfico de barras usando o indice da tabela como rótulos do eixo X
    cor = np.random.rand(3)
    plt.bar(tab.index,tab['Freq_Relativa'],width = 0.7, tick_label=tab.index,color=cor,alpha=0.6)

    plt.ylim(0,tab['Freq_Relativa'].max()+0.2)
    plt.title("Frequência Relativa de {}".format(list(tab.columns)[0]))

    # cria um conjunto de pares de rótulos e frequencias relativas
    for x,y in zip(tab.index,tab['Freq_Relativa']):

        # formata o rotulo do percentual
        rotulo = "{:.4f}".format(y)

        # coloca o rotulo na posição (x,y), alinhado ao centro e com distância 0,5 do ponto (x,y)
        plt.annotate(rotulo,(x,y),ha='center',textcoords="offset points",xytext=(0,5))
        
        
def tabela_bivariada(data, var):
    
    df = pd.DataFrame(data[var].value_counts()).sort_values(by=var,ascending=False)
    total = df[var].sum()
    df['Freq_Relativa'] = (df[var]/total).round(decimals=4)
    df['Freq_Acumulada'] = df['Freq_Relativa'].cumsum().round(decimals=4)
    return df