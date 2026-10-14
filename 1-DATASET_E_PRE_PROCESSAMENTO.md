# Dataset e pré-processamento

## Problema e fonte

O projeto investiga um problema de **classificação**: dadas características
registradas sobre uso de vídeos curtos, tempo de tela, sono e perfil, prever a
categoria `cognitive_decay_tier`. As quatro categorias presentes no arquivo
são *Normal / Intact Focus*, *Mild Attentional Strain*, *Severe Focus
Fragmentation* e *Acute Brain Rot / Executive Burnout*.

O arquivo local é [`dataset.csv`](dataset.csv), associado ao conjunto público
[Brain Rot: Short Form Video & Attention Span, no Kaggle](https://www.kaggle.com/datasets/samartalwar/brain-rot-short-form-video-and-attention-span).
Os números abaixo foram verificados no **CSV usado no projeto**. A origem dos
registros e a regra usada para definir `cognitive_decay_tier` ainda não foram
confirmadas na documentação da fonte. Por isso,
tratamos o arquivo como um conjunto para estudar métodos de classificação e
interpretabilidade; ainda não podemos afirmar que os resultados representam
pessoas reais ou uma relação causal.

## Estrutura do arquivo

O CSV contém **9.000 linhas e 19 colunas**: um identificador, 17 possíveis
variáveis explicativas e uma coluna alvo. Satisfaz os requisitos de tamanho,
classificação, variáveis numéricas e variáveis categóricas do enunciado.
As categorias textuais são `gender` (Female, Male e Non-Binary) e
`primary_platform` (TikTok, Instagram Reels, YouTube Shorts e Snapchat
Spotlight). Há medidas numéricas, como idade, horas diárias de vídeos curtos,
swipes por minuto e horas de sono. No arquivo, a idade vai de 16 a 55 anos,
as horas diárias de vídeos de 0,3 a 10,5 e as horas de sono de 3,5 a 9,5.

O alvo é desbalanceado:

| Categoria | Registros | Parcela aproximada |
| --- | ---: | ---: |
| Normal / Intact Focus | 3.863 | 42,92% |
| Mild Attentional Strain | 3.677 | 40,86% |
| Severe Focus Fragmentation | 1.355 | 15,06% |
| Acute Brain Rot / Executive Burnout | 105 | 1,17% |

Existem **zero valores ausentes**. Não há linhas inteiramente duplicadas,
inclusive quando o identificador é ignorado. O código chama `drop_duplicates()`
antes de selecionar as variáveis; nessa versão do CSV, a chamada não elimina
nenhum registro. Não foi necessário preencher valores ausentes. Também não
foram removidos valores extremos automaticamente: o projeto não dispõe de
informação suficiente para tratá-los como erros de medição.

## Seleção das variáveis

Usamos **11 características** como entrada (`X`):

| Tipo | Colunas selecionadas |
| --- | --- |
| Categóricas | `gender`, `primary_platform` |
| Numéricas | `age`, `daily_short_form_hours`, `avg_clip_duration_sec`, `swipes_per_minute`, `scroll_sessions_day`, `bedtime_scroll_min`, `total_sleep_hours`, `multitasking_score`, `total_screen_time_hours` |

A coluna `participant_id` foi excluída porque é um identificador único, sem
significado previsto para a classificação. A coluna `cognitive_decay_tier`
foi separada como variável alvo (`y`). As seis medidas cognitivas
`sustained_attention_span_sec`, `working_memory_score`,
`task_switch_latency_sec`, `brain_fog_score`, `dopamine_burnout_index` e
`attention_retention_pct` ficaram fora de `X` nesta primeira versão. Elas
descrevem resultados cognitivos próximos daquilo que o alvo pretende resumir.
Sem conhecer a regra de construção do rótulo, incluí-las poderia permitir que
o modelo apenas reconstruísse a própria categoria. **Isso é uma precaução
contra possível vazamento de informação, não uma confirmação de que essas
colunas foram usadas para criar o alvo.**

Essa seleção define o alcance do experimento: estamos avaliando a previsão
das categorias a partir das 11 características escolhidas. Outros conjuntos
de variáveis poderiam produzir desempenhos e explicações diferentes.

## Divisão e transformações

O conjunto foi dividido com `train_test_split(test_size=0.2,
random_state=42, stratify=y)`: **7.200 registros para treino** e **1.800 para
teste**. A estratificação preserva aproximadamente a proporção das quatro
classes nas duas partes; no teste ficaram 773 casos Normal, 735 Mild, 271
Severe e 21 Acute. A semente fixa permite reproduzir a mesma divisão.

As variáveis categóricas são transformadas com `OneHotEncoder`. Cada categoria
vira uma coluna indicadora, sem impor uma ordem artificial às plataformas ou
aos gêneros. O codificador foi configurado com `handle_unknown="ignore"` para
que uma categoria inédita no teste não impeça a previsão. Nas variáveis
numéricas, a decisão depende do modelo:

| Modelo | Tratamento das numéricas | Motivo |
| --- | --- | --- |
| KNN | `StandardScaler` | O KNN usa distâncias; variáveis medidas em escalas maiores poderiam dominar a comparação. |
| Naïve Bayes (`GaussianNB`) | Valores originais | A padronização não é necessária para o cálculo de médias e variâncias gaussianas por classe. |
| Árvore de Decisão | Valores originais | A árvore decide por limiares; manter as unidades facilita ler as regras. |

O `StandardScaler` usado no KNN subtrai a média e divide pelo desvio-padrão
estimados **somente no treino**. Ele é uma padronização, não uma transformação
para o intervalo de 0 a 1. Cada modelo é um `Pipeline` que reúne transformações
e classificador. Assim, ao avaliar os 1.800 registros de teste, o código
aplica as transformações aprendidas no treino sem reajustá-las com dados do
teste. O alvo permanece com seus rótulos textuais; não é uma entrada para
o pré-processamento das características.

## Escolhas de modelagem e pontos a revisar

Como comparação inicial, usamos `KNeighborsClassifier(n_neighbors=5)`,
`GaussianNB()` e `DecisionTreeClassifier(random_state=42)` sem busca de
hiperparâmetros. A árvore não recebeu limite de profundidade nesta versão.
O `GaussianNB` também recebe colunas binárias do one-hot encoding, embora
sua hipótese de distribuição normal seja uma aproximação inadequada para
elas. Essas escolhas precisam ser lembradas ao discutir desempenho e
interpretação.

A classe Acute tem somente 105 registros no arquivo completo. Por isso, além
da acurácia, usamos precisão, recall e F1 por classe, F1 macro, F1 ponderado
e matrizes de confusão. Os resultados estão discutidos em
[4-COMPARACAO_MODELOS.md](4-COMPARACAO_MODELOS.md); a análise SHAP e os exemplos
individuais estão em
[3-ANALISE_INTERPRETABILIDADE.md](3-ANALISE_INTERPRETABILIDADE.md).

Antes de finalizar o trabalho, é importante consultar a documentação da fonte
para verificar **origem dos registros, significado das medidas e construção
de `cognitive_decay_tier`**. Até essa verificação, as conclusões dizem respeito
ao comportamento dos modelos neste CSV.
