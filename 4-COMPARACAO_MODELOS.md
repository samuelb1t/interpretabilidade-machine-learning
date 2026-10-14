# Etapa 4 — Comparação e análise dos resultados

## Como foi feita a comparação

Os modelos KNN, Naïve Bayes e Árvore de Decisão foram treinados com os mesmos
7.200 registros e avaliados nos mesmos 1.800 registros de teste. A divisão foi
estratificada, com 80% dos dados para treino e 20% para teste. O alvo foi
`cognitive_decay_tier`, com quatro classes. Foram usadas 11 variáveis de
entrada, incluindo hábitos de uso de vídeos curtos, sono, idade e plataforma.
As decisões sobre seleção de variáveis e pré-processamento estão no código;
as explicações individuais e os detalhes dos métodos de interpretabilidade
estão em [3-ANALISE_INTERPRETABILIDADE.md](3-ANALISE_INTERPRETABILIDADE.md).

| Modelo | Acurácia no teste | F1 macro | F1 ponderado |
| --- | ---: | ---: | ---: |
| KNN | 83,22% | 78,05% | 83,19% |
| Naïve Bayes | 79,11% | 69,15% | 79,21% |
| Árvore de Decisão | 81,28% | 79,09% | 81,33% |

O KNN obteve a maior **acurácia**, acertando a maior fração dos registros de
teste. A árvore teve o maior **F1 macro**, que calcula o F1 de cada classe e
depois dá o mesmo peso às quatro. Essa diferença importa porque as classes
são desbalanceadas: havia apenas **21 exemplos Acute** no teste, contra 773
da classe Normal. Uma acurácia alta pode conviver com desempenho fraco em uma
classe rara. O F1 ponderado, por dar mais peso às classes frequentes, fica
próximo da acurácia nos três modelos.

O desempenho na classe rara mostra esse cuidado na prática:

| Modelo | Acute identificados | Recall Acute | Precisão Acute | Recall Severe |
| --- | ---: | ---: | ---: | ---: |
| KNN | 12 de 21 | 57% | 75% | 76% |
| Naïve Bayes | 15 de 21 | 71% | 35% | 64% |
| Árvore de Decisão | 15 de 21 | 71% | 75% | 80% |

O Naïve Bayes encontrou mais casos Acute do que o KNN, mas chamou de Acute
**28 registros que eram Severe**; por isso, sua precisão para Acute foi de
apenas 35%. A árvore combinou recall de 71% para Acute com precisão de 75%
e teve o maior recall para Severe. Como só há 21 casos Acute no teste, uma
diferença de poucos acertos muda bastante esses percentuais. As conclusões
sobre essa classe devem ser apresentadas com essa limitação.

## Os resultados fazem sentido para o problema?

Dentro deste conjunto de dados, sim: as variáveis ligadas à intensidade e ao
modo de uso de vídeos curtos aparecem entre as mais utilizadas pelos modelos.
Por exemplo, tempo diário de vídeos curtos e swipes por minuto tiveram
destaque no KNN e na árvore. Isso é coerente com a tarefa de classificar os
níveis de `cognitive_decay_tier` a partir dos hábitos registrados.

Esse resultado descreve **padrões aprendidos neste dataset**. Ele não mostra
que o uso de vídeos curtos causou perda de atenção, nem garante desempenho
semelhante em pessoas fora do conjunto de dados. A interpretação também
depende de confirmar na fonte como `cognitive_decay_tier` foi produzido.
Se o rótulo tiver sido calculado diretamente a partir de alguma variável
usada como entrada, parte do desempenho pode refletir a recuperação dessa
regra. Essa verificação ainda está pendente.

## Os modelos concordam sobre as variáveis mais relevantes?

**Concordam parcialmente.** Para comparar os três pelo mesmo procedimento,
aplicamos SHAP por permutação às probabilidades previstas. Usamos 40 registros
de treino como referência e uma amostra de 18 registros de teste, com
representação das quatro classes. A tabela apresenta as três maiores médias
de `|SHAP|` de cada modelo nessa amostra; os valores são mudanças médias
absolutas na escala de probabilidade, e não porcentagens de importância:

| Modelo | 1ª variável | 2ª variável | 3ª variável |
| --- | --- | --- | --- |
| KNN | Horas diárias de vídeos (0,086) | Swipes por minuto (0,083) | Rolagem antes de dormir (0,055) |
| Naïve Bayes | Sessões de rolagem por dia (0,081) | Tempo total de tela (0,076) | Rolagem antes de dormir (0,071) |
| Árvore de Decisão | Horas diárias de vídeos (0,203) | Duração média dos vídeos (0,090) | Swipes por minuto (0,061) |

As horas diárias de vídeos aparecem com peso relevante nos três, embora não
estejam entre as três primeiras do Naïve Bayes. Árvore e KNN também destacam
os swipes por minuto. O Naïve Bayes dá mais destaque a sessões de rolagem e
tempo total de tela. O
[gráfico de comparação SHAP](resultados/shap/comparacao_importancias.png)
mostra todas as 11 variáveis. A ordem é **exploratória**, pois vem de apenas
18 registros e de valores SHAP aproximados; não deve ser generalizada para
todo o conjunto de teste sem uma análise maior.

## O que as ferramentas explicaram?

Na **Árvore de Decisão**, a estrutura e o caminho até uma folha mostram as
condições exatas aplicadas a um registro. A importância por redução de
impureza resume quais variáveis ajudaram mais nas divisões durante o treino.
O SHAP acrescenta uma explicação local em termos de contribuição à
probabilidade de cada classe. O tempo diário de vídeos foi o principal
destaque tanto na importância da árvore quanto na amostra SHAP.

No **Naïve Bayes**, as frequências iniciais das classes, médias e variâncias
condicionais permitem examinar como o modelo combina as evidências. O SHAP
mostra como cada variável deslocou a probabilidade de uma classe em relação
à referência escolhida. Essas duas análises respondem perguntas diferentes:
a comparação das distribuições confronta classes; o SHAP compara uma previsão
com uma referência.

No **KNN**, os vizinhos mostram diretamente de onde veio o voto. O SHAP ajuda
a investigar quais características influenciaram a probabilidade de uma
previsão específica. Ele não cria uma regra global simples para o KNN.

O registro de teste **6811** ilustra a diferença entre os modelos. Sua classe
real era *Severe Focus Fragmentation*. O KNN acertou porque três dos cinco
vizinhos eram Severe; a árvore também acertou, seguindo regras até uma folha
com registros dessa classe. O Naïve Bayes previu *Mild Attentional Strain*
com **91,25%** de probabilidade. Nesse caso, o SHAP indicou que horas diárias
de vídeos e tempo total de tela contribuíram para aumentar a probabilidade
de Mild no Naïve Bayes; para os outros modelos, outras combinações de
variáveis sustentaram Severe. O caso mostra que modelos podem usar os mesmos
dados de maneiras diferentes e que uma previsão com alta confiança ainda
pode estar errada.

## Limitações de interpretabilidade

| Modelo | Limitação principal neste experimento |
| --- | --- |
| KNN | A explicação direta depende dos cinco vizinhos de cada caso. Distância, escala e quantidade de vizinhos afetam o resultado; não há uma lista de regras aprendidas que resuma todas as previsões. |
| Naïve Bayes | Supõe independência entre variáveis dada a classe e distribuições gaussianas também para colunas binárias do one-hot encoding. Várias variáveis de uso são correlacionadas mesmo dentro das classes, o que dificulta interpretar as probabilidades como confiança bem calibrada. |
| Árvore de Decisão | Cada caminho é legível, mas a árvore completa tem profundidade 21 e 791 folhas. Folhas com poucos exemplos podem produzir probabilidades extremas e regras pouco estáveis. |

O SHAP também tem limites comuns aos três: suas atribuições dependem da
referência escolhida e, neste trabalho, foram estimadas por permutação para
uma amostra pequena. Variáveis correlacionadas podem dividir ou deslocar
atribuições entre si. Nenhuma dessas explicações estabelece causalidade.

## Síntese para a apresentação

Os três modelos tiveram desempenho suficiente para comparar suas decisões,
mas não existe um vencedor único para todos os critérios. O KNN alcançou a
maior acurácia; a árvore obteve o maior F1 macro e o melhor recall da classe
Severe; o Naïve Bayes identificou mais casos Acute que o KNN, porém produziu
mais falsos positivos nessa classe. As explicações revelaram concordância
parcial sobre as variáveis relevantes e mostraram, no caso 6811, que modelos
com resultados agregados próximos podem chegar a decisões diferentes para
uma pessoa. Por isso, avaliar apenas a acurácia não basta: examinar erros,
classes raras e razões das previsões ajuda a compreender quando confiar
menos em uma decisão do modelo.
