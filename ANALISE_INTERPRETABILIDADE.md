# Etapa 3 — Interpretabilidade dos modelos

## O que foi analisado

O alvo é `cognitive_decay_tier`. Os três modelos foram treinados com as mesmas 11
variáveis originais, em 7.200 registros, e avaliados nos mesmos 1.800 registros
de teste. As categorias `gender` e `primary_platform` recebem *one-hot encoding*
em cada pipeline; o KNN também padroniza as variáveis numéricas. O identificador
`participant_id` e as medidas cognitivas não são entradas dos modelos.

Usamos SHAP por permutação sobre `predict_proba` dos três pipelines. A referência
contém 40 registros sorteados **somente do treino**. As explicações foram
calculadas para 18 registros do teste: quatro por classe e os dois casos
comparados abaixo. Para cada variável e modelo, a importância global é a média
de `|SHAP|` sobre esses registros e as quatro classes. Essa amostra pequena e
deliberadamente equilibrada serve para exploração; a ordem das variáveis pode
mudar com outra amostra ou referência. Os valores estão na escala de
**probabilidade**, e não são os mesmos números da importância por impureza da
árvore.

Os [gráficos comparativos](resultados/shap/comparacao_importancias.png), os
[valores calculados](resultados/shap/comparacao_importancias.csv) e as
[explicações individuais](resultados/shap/explicacoes.txt) são gerados pelo código.
Os gráficos individuais mostram valores positivos e negativos para a **classe
prevista por aquele modelo**. Um SHAP positivo aumenta a probabilidade dessa
classe em relação à referência escolhida; não é evidência de causa e efeito.

## Árvore de Decisão

A árvore classifica um registro seguindo condições sucessivas até uma folha. A
primeira condição da árvore treinada é `daily_short_form_hours <= 2,55`.
Sua estrutura completa tem profundidade 21 e 791 folhas. O
[SVG da árvore](resultados/arvore/arvore_completa.svg) e as
[regras em texto](resultados/arvore/regras_completas.txt) mostram todas as divisões.

Pela importância por redução de impureza calculada durante o treino, as três
variáveis principais foram tempo diário de vídeos curtos (48,9%), swipes por
minuto (19,8%) e duração média dos vídeos (8,5%). A análise SHAP de 18
registros também destacou o tempo diário de vídeos (média de 0,203 na escala
de probabilidade), seguido da duração média dos vídeos (0,090) e dos swipes
por minuto (0,061). Os valores dos dois métodos têm definições diferentes;
comparamos a **ordem das variáveis**, não seus percentuais diretamente.

No registro 6811, a classe real era *Severe Focus Fragmentation*, e a árvore
acertou. O registro tinha 4,5 horas diárias de vídeos curtos, 10,2 swipes por
minuto e duração média de 5,9 segundos por vídeo. Ele seguiu, entre outras,
as condições `daily_short_form_hours > 2,55`, `<= 5,85`,
`swipes_per_minute <= 10,6` e `avg_clip_duration_sec <= 8,55`. Chegou a uma
folha que continha nove registros de treino e previu *Severe* com 100% de
probabilidade. Para essa classe, o SHAP atribuiu contribuições positivas de
aproximadamente **33,8 pontos percentuais** ao tempo diário, **25,0** aos
swipes e **18,8** à duração média. Essa probabilidade de 100% descreve a
folha; não garante acerto em futuros registros. A grande quantidade de folhas
também dificulta acompanhar a árvore inteira, apesar de cada caminho ser
formado por regras legíveis.

## Naïve Bayes

O `GaussianNB` estima, para cada classe, sua frequência inicial no treino
(*prior*) e uma distribuição normal para cada variável transformada. Para
classificar, ele combina essas informações assumindo que as variáveis são
independentes quando a classe é conhecida. As médias e variâncias aprendidas
permitem investigar por que um valor parece mais compatível com uma classe do
que com outra; a classificação resulta da **combinação de todas as variáveis**.

No treino, os *priors* de *Mild* e *Severe* eram, respectivamente, **40,9%** e
**15,1%**. No registro 6811, o Naïve Bayes errou: previu *Mild* com
**91,25%**, embora a classe real fosse *Severe*. A tabela mostra três valores
desse registro e as médias que o modelo aprendeu para as duas classes:

| Variável | Registro 6811 | Média Mild | Média Severe |
| --- | ---: | ---: | ---: |
| Horas diárias de vídeos curtos | 4,5 | 3,65 | 5,64 |
| Tempo total de tela, em horas | 7,6 | 7,39 | 9,44 |
| Swipes por minuto | 10,2 | 6,44 | 9,77 |

O tempo total de tela de 7,6 horas está próximo da média de *Mild*; os 10,2
swipes por minuto estão próximos da média de *Severe*. Considerando também
as variâncias, os valores, as outras variáveis e o *prior*, o cálculo do
modelo favoreceu *Mild*. Em termos de logaritmo da razão entre *Mild* e
*Severe*, o *prior* contribuiu cerca de **+1,00**, o tempo total de tela
**+0,78** e as horas diárias de vídeos **+0,38**. A duração média dos vídeos
contribuiu **−0,65**, em favor de *Severe*. A soma de todos os termos foi
aproximadamente **+2,34**, explicando por que *Mild* recebeu probabilidade
maior. Comparar somente as médias seria insuficiente: o modelo usa também
as variâncias.

No SHAP desse mesmo registro, as maiores contribuições positivas para a
probabilidade de *Mild* foram horas diárias de vídeos (**+12,4 pontos
percentuais**) e tempo total de tela (**+10,2**). O SHAP compara a previsão
com uma **referência de treino**; já a razão de verossimilhanças acima compara
diretamente **Mild com Severe**. Por isso, o sinal ou o tamanho de uma variável
não precisa coincidir nos dois cálculos.

Na amostra SHAP, as variáveis com maior impacto médio no Naïve Bayes foram
sessões de rolagem por dia (0,081), tempo total de tela (0,076) e tempo de
rolagem antes de dormir (0,071). Há uma limitação importante: horas diárias
de vídeos, sessões por dia e tempo total de tela são correlacionados no CSV
(coeficientes entre 0,89 e 0,95 para esses pares). A associação também aparece
dentro das classes: em *Mild*, a correlação entre horas diárias e sessões por
dia é 0,88; em *Severe*, é 0,94. Isso contraria a hipótese de independência
condicional usada pelo Naïve Bayes. Além disso, o
`GaussianNB` aplica sua hipótese gaussiana também às colunas binárias criadas
pelo *one-hot encoding*. Essas hipóteses podem tornar as probabilidades mal
calibradas; a confiança de 91,25% no caso errado é um motivo para não tratar
esse número como certeza.

## KNN

O KNN usado no trabalho busca os cinco registros de treino mais próximos depois
do pré-processamento. A classe prevista é a mais frequente entre eles. No
registro 6811, **três vizinhos eram Severe e dois eram Mild**. Por isso, o KNN
acertou *Severe* e atribuiu probabilidade de **60%** a essa classe (três votos
em cinco). Esse resultado pode ser explicado diretamente pela vizinhança,
mas não revela sozinho quais variáveis determinaram as distâncias.

O SHAP complementa a votação: para a probabilidade de *Severe* nesse registro,
a duração média dos vídeos contribuiu cerca de **+17,3 pontos percentuais**,
o escore de multitarefa **+14,3** e os swipes por minuto **+6,5**, em relação
à referência. Na amostra SHAP, as maiores importâncias médias do KNN foram
horas diárias de vídeos (0,086) e swipes por minuto (0,083).

O KNN não aprende uma lista simples de regras ou coeficientes globais: a
explicação direta depende dos vizinhos de **cada registro**. A escolha de
cinco vizinhos, da escala e da medida de distância afeta o resultado. Com
apenas 105 registros da classe *Acute* no conjunto inteiro, casos dessa classe
podem ter poucos vizinhos semelhantes; no teste, o KNN identificou 12 dos 21
casos *Acute*. O SHAP fornece uma explicação local aproximada, condicionada à
referência selecionada, sem transformar o KNN em um modelo de regras.

## Comparação e limites da interpretação

Os três modelos deram importância ao tempo diário de vídeos curtos, mas não
concordaram totalmente na ordem das variáveis. Árvore e KNN destacaram os
swipes por minuto; o Naïve Bayes deu mais destaque às sessões de rolagem e ao
tempo total de tela. O caso 6811 mostra uma consequência concreta dessa
diferença: KNN e árvore acertaram *Severe*, enquanto o Naïve Bayes previu
*Mild* com alta confiança.

Essas análises explicam **como os modelos usaram os dados disponíveis**. Elas
não mostram que vídeos curtos causam perda de atenção. Antes da discussão
final do trabalho, ainda é necessário conferir na descrição da fonte como
`cognitive_decay_tier` foi construído e se alguma variável de entrada
participou diretamente da definição do rótulo. Isso é especialmente relevante
para interpretar a força das associações encontradas.

## Referências metodológicas

- [Documentação do SHAP](https://shap.readthedocs.io/en/latest/): valores SHAP, explicações locais e comparação com uma referência.
- [Documentação do `GaussianNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html): médias, variâncias e probabilidades das classes.
- [Estrutura de uma árvore de decisão no scikit-learn](https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html): condições e caminho de uma previsão.
