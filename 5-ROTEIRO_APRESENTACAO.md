# Roteiro da apresentação em vídeo

**Duração planejada:** aproximadamente 12 minutos e 30 segundos, dentro da
faixa de 10 a 15 minutos pedida no enunciado. O roteiro cobre dataset,
metodologia, avaliação, interpretabilidade dos três modelos, comparação,
importância da interpretabilidade e conclusões. Os tempos são sugestões;
ajustem à quantidade de integrantes e ao ritmo da fala.

| Parte | Tempo sugerido | Ideia central |
| --- | --- | --- |
| 1. Abertura e pergunta do trabalho | 0:00–0:45 | Prever uma classe é apenas parte do problema; queremos entender a decisão. |
| 2. Dataset e contexto | 0:45–2:15 | Fonte, alvo, exemplos de variáveis e distribuição das classes. |
| 3. Preparação dos dados | 2:15–3:45 | Seleção das colunas, codificação, padronização e divisão 80/20. |
| 4. Modelos e método de avaliação | 3:45–4:30 | KNN, Naïve Bayes e árvore; mesmas divisões e métricas. |
| 5. Resultados preditivos | 4:30–6:00 | Acurácia, F1 macro e dificuldade da classe rara. |
| 6. Árvore de Decisão | 6:00–7:00 | Regras, importância e contribuição SHAP. |
| 7. Naïve Bayes | 7:00–8:00 | Probabilidades condicionais, SHAP e um erro confiante. |
| 8. KNN | 8:00–9:00 | Voto dos vizinhos, dificuldade de interpretação direta e SHAP. |
| 9. Comparação entre modelos | 9:00–11:00 | Concordâncias, divergências e caso 6811. |
| 10. Importância e conclusão | 11:00–12:30 | O que as explicações permitem compreender e seus limites. |

## 1. Abertura: qual é a pergunta?

**Na tela:** título do trabalho, integrantes e a pergunta: “Por que o modelo
chegou a essa classificação?”.

**Pontos para a fala:** o grupo treinou três algoritmos para prever categorias
de `cognitive_decay_tier` e investigou tanto o desempenho quanto as razões
das previsões. Apresentem a proposta como um exercício de aprendizado de
máquina e interpretabilidade. Ainda não há base para afirmar efeito causal de
vídeos curtos sobre atenção.

## 2. Dataset e contexto

**Na tela:** fonte
[Kaggle — Brain Rot: Short Form Video & Attention Span](https://www.kaggle.com/datasets/samartalwar/brain-rot-short-form-video-and-attention-span),
9.000 registros, quatro classes e exemplos de colunas numéricas e categóricas.
Uma pequena tabela com as frequências das classes ajuda mais que listar as
19 colunas.

**Pontos para a fala:** o alvo é categórico, o que torna a tarefa uma
classificação. Há 3.863 registros Normal, 3.677 Mild, 1.355 Severe e apenas
105 Acute. Comentem que essa diferença de frequência influencia a leitura
das métricas. O documento
[1-DATASET_E_PRE_PROCESSAMENTO.md](1-DATASET_E_PRE_PROCESSAMENTO.md) reúne os
detalhes. Antes da gravação, verifiquem na página da fonte como o alvo foi
criado; se isso não estiver documentado, digam claramente que a regra não
foi confirmada.

## 3. Metodologia e pré-processamento

**Na tela:** fluxo simples: `CSV → 11 características + alvo → 80% treino /
20% teste → transformações → três modelos`.

**Pontos para a fala:** removemos o identificador; deixamos de fora seis
medidas cognitivas por possível proximidade com o alvo, sem afirmar que há
vazamento confirmado. Não havia valores ausentes nem linhas duplicadas. As
duas variáveis categóricas foram codificadas por one-hot. O KNN recebeu
variáveis numéricas padronizadas, pois depende de distâncias; a árvore e o
Naïve Bayes receberam as numéricas nas unidades originais. A divisão 80/20
foi estratificada. Todas as transformações foram ajustadas apenas com o treino.

## 4. Modelos e avaliação

**Na tela:** três caixas: `KNN (5 vizinhos)`, `GaussianNB`, `Árvore de Decisão`.
Ao lado, as métricas: acurácia, precisão, recall e F1 por classe, F1 macro e
matriz de confusão.

**Pontos para a fala:** os três receberam os mesmos exemplos de treino e
foram testados nos mesmos 1.800 registros. Expliquem F1 macro em uma frase:
ele dá o mesmo peso a cada classe, inclusive à rara. Isso justifica olhar
além da acurácia.

## 5. Resultados preditivos

**Na tela:** tabela resumida de
[4-COMPARACAO_MODELOS.md](4-COMPARACAO_MODELOS.md):

| Modelo | Acurácia | F1 macro |
| --- | ---: | ---: |
| KNN | 83,22% | 78,05% |
| Naïve Bayes | 79,11% | 69,15% |
| Árvore de Decisão | 81,28% | 79,09% |

**Pontos para a fala:** o KNN teve maior acurácia; a árvore, maior F1 macro.
Dos 21 registros Acute no teste, o KNN identificou 12; os outros dois modelos
identificaram 15. O Naïve Bayes, porém, também confundiu 28 registros Severe
com Acute. Essa combinação de acertos e falsos positivos mostra por que
nenhuma métrica isolada resume o resultado.

## 6. Interpretabilidade da árvore

**Na tela:** usem os
[primeiros níveis da árvore](resultados/arvore/primeiros_niveis.png), porque a
árvore inteira tem 791 folhas e não caberia de forma legível no vídeo. Mostrem
também o
[gráfico comparativo de SHAP](resultados/shap/comparacao_importancias.png).

**Pontos para a fala:** cada previsão segue uma sequência de condições. A
primeira divisão usa horas diárias de vídeos curtos em torno de 2,55 horas.
Tempo diário, duração média dos vídeos e swipes tiveram destaque na amostra
SHAP da árvore. Um caminho pode ser lido diretamente, mas a árvore completa
é grande; folhas pequenas podem gerar confiança excessiva.

## 7. Interpretabilidade do Naïve Bayes

**Na tela:** uma tabela simples com o registro 6811: classe real Severe,
previsão Mild, probabilidade de 91,25%. Mostrem também o
[gráfico SHAP desse caso](resultados/shap/Naive_Bayes_erro_em_algum_modelo.png).

**Pontos para a fala:** o modelo combina a frequência de cada classe com as
distribuições aprendidas para as características. No exemplo, tempo total de
tela e horas diárias de vídeos favoreceram Mild na combinação; outros dados
favoreceram Severe. A decisão final foi Mild e estava errada. Comentem a
hipótese de independência do Naïve Bayes: algumas variáveis de uso são
correlacionadas mesmo dentro das classes. As probabilidades do modelo não
devem ser tratadas como certeza.

## 8. Interpretabilidade do KNN

**Na tela:** “Registro 6811: 3 vizinhos Severe + 2 vizinhos Mild → previsão
Severe”. Mostrem o
[gráfico SHAP do KNN nesse caso](resultados/shap/KNN_erro_em_algum_modelo.png).

**Pontos para a fala:** é possível inspecionar os vizinhos para explicar o
voto, mas o KNN não fornece regras globais simples. O SHAP ajuda a estimar
quais características pesaram na previsão local. No caso mostrado, duração
média dos vídeos, escore de multitarefa e swipes aumentaram a probabilidade
de Severe em relação à referência usada.

## 9. Comparação e análise crítica

**Na tela:** o
[gráfico de importâncias SHAP dos três modelos](resultados/shap/comparacao_importancias.png)
e uma linha para o caso 6811: `real = Severe; KNN = Severe; NB = Mild;
árvore = Severe`.

**Pontos para a fala:** os três dão alguma importância ao tempo diário de
vídeos, mas a ordem das variáveis difere. KNN e árvore destacam swipes; o
Naïve Bayes dá mais peso a sessões e tempo total de tela. No caso 6811, o
Naïve Bayes errou com alta probabilidade, enquanto os outros acertaram. Isso
mostra que métricas agregadas semelhantes não significam decisões iguais.
Ressaltem que o SHAP foi calculado em apenas 18 registros e depende da
referência escolhida.

## 10. Por que interpretabilidade importa? Conclusão

**Na tela:** três frases curtas: “Desempenho”, “Motivos das previsões”,
“Limites e erros”.

**Pontos para a fala:** a interpretabilidade permitiu identificar quais
informações influenciaram os modelos, explicar um erro concreto e perceber
que uma probabilidade alta pode acompanhar uma previsão errada. As regras da
árvore, as distribuições do Naïve Bayes, os vizinhos do KNN e o SHAP
responderam a partes diferentes da pergunta inicial. A conclusão é que
boas métricas são necessárias, mas uma decisão deve ser usada com mais
cuidado quando não entendemos suas condições e limitações. Nenhuma das
análises demonstra que o uso de vídeos curtos **causa** perda de atenção.

## Antes de gravar e entregar

- Confirmar o que a fonte diz sobre origem dos registros e definição de
  `cognitive_decay_tier`; ajustar a fala se houver informação nova.
- Ensaiar para manter o vídeo entre 10 e 15 minutos. Deixar tabelas e gráficos
  legíveis e explicar as unidades dos números SHAP como mudanças na
  probabilidade.
- Disponibilizar o código no GitHub ou em pasta compactada, conforme o
  enunciado. Publicar o vídeo como **não listado** no YouTube ou em outra
  plataforma de compartilhamento e incluir o link em um arquivo `.txt`.
- Revisar os documentos do projeto: [dataset e pré-processamento](1-DATASET_E_PRE_PROCESSAMENTO.md),
  [interpretabilidade](3-ANALISE_INTERPRETABILIDADE.md) e
  [comparação](4-COMPARACAO_MODELOS.md).

O prazo indicado no enunciado é **15/10/2026**.
