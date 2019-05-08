import pandas as pd #importando o pandas
from sklearn.tree import DecisionTreeClassifier #importando o algoritmi DecisionTreeClassifier

###IMPORTANDO DATASETS DE TREINO E TESTE

train = pd.read_csv('train.csv') #carregando os dados de treinamento
test = pd.read_csv('test.csv')   #carregando os dados de teste



###FASE DE VARREDURA DOS DADOS, DEIXA-LOS PRONTO PARA O TREINAMENTO

train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True) #tirando colunas desnecessarias para a predicao
test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)  #deve tirar da tabela de test tbm para nao dar erro
#inplace = True apenas diz que vc quer alter o dataset que estausando a funcao

new_data_train = pd.get_dummies(train) #transforma todos dados string em dados numericos
new_data_test = pd.get_dummies(test)   #para treinar tem que ser numero, em outras palavras ele 
                                       #pega um coluna se sexo que tem masculino e fenino e divide em duas
                                       #onde uma sera masculino e outra feminio e os passageiros recebem
                                       #1 ou 0 se sao ou nao dql sexo

#print(new_data_train.isnull().sum().sort_values(ascending=False).head(10))
'''
new_data_train indica que nós utilizaremos o DataFrame;
.isnull() é uma função que retorna todos os valores nulos encontrados;
.sum() irá somar todas as ocorrências e agrupá-las;
.sort_values(ascending=False) ordenará os dados. Ao passar o argumento 
ascending=False eu indico querer ordenar do maior para o menor.
a resposta que ele da e que tem 177 valores NaN na coluna Age
'''

new_data_train['Age'].fillna(new_data_train['Age'].mean(), inplace=True)
new_data_test['Age'].fillna(new_data_test['Age'].mean(), inplace=True)
'''
fillna e uma funcao que troca todos os valores NaN para o valor passado,
nesse caso e a media das idades

obs: Cada caso de nulo e um caso e vc deve analisar a melhor forma de trata-lo
'''

#fazer a verificacao de nulos nos dados de teste agora = 1 nulo na coluna Fare
#print(new_data_test.isnull().sum().sort_values(ascending=False).head(10))

#tirando o valor nulo do preco da passagem(Fare)
new_data_test['Fare'].fillna(new_data_test['Fare'].mean(), inplace=True)



###FASE DE TREINAMENTO

#Separar entradas e a saida X e Y, do data_train
X = new_data_train.drop('Survived', axis=1)
Y = new_data_train['Survived']

#Fazendo a instaciacao do algoritmo, nesse caso uma arvore de precisao
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
'''
max_depth e um numero de camadas da arvore ou perguntas nesse caso que a arvore ira fazer, 
tres foi o melhor desempenho
random_state e a semente gerada pelo gerador de numeros aleatorios
'''

tree.fit(X,Y)
'''
Fazendo o treinamento do algoritmo, passando a entradas e a saida esperada
'''


'''
X_test recebe o data_set de teste, so para padronizar, note que nao precisa dar drop nesse
Y_test le o arquivo que tem as saidas do teste e drop o numero da passagem
'''
#print(tree.score(X,Y)) # printa o score usando o conjunto de train



###ARRUMANDO ARQUIVOS PARA FAZER A SUBMISSAO DOS DADOS DA FORMA QUE E PEDIDO

submission = pd.DataFrame() #cria um dataframe
submission['PassengerId'] = new_data_test['PassengerId'] #uma coluna recebendo os id
submission['Survived'] = tree.predict(new_data_test)  #uma coluna com a predicao do conjunto teste


#submission.to_csv('submission.csv', index=False) #salvando em um arquivo csv e sem index ou seja, sem 


