"""Compara explicações SHAP dos três classificadores nas variáveis originais."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PASTA_RESULTADOS = Path(__file__).parent / "resultados" / "shap"
COLUNAS_CATEGORICAS = ("gender", "primary_platform")
AMOSTRAS_POR_CLASSE = 4
TAMANHO_REFERENCIA = 40
PERMUTACOES = 2


def criar_codificacao_temporaria(X_train):
    """Converte texto em números só para o masker do SHAP.

    Antes de chamar predict_proba, cada código volta à categoria original.
    Assim, o pipeline continua fazendo seu próprio one-hot encoding e escala.
    """
    colunas = list(X_train.columns)
    categorias = {
        coluna: sorted(X_train[coluna].unique())
        for coluna in COLUNAS_CATEGORICAS
    }

    def codificar(dados):
        convertido = dados[colunas].copy()
        for coluna, opcoes in categorias.items():
            convertido[coluna] = pd.Categorical(
                convertido[coluna], categories=opcoes
            ).codes.astype(float)
        return convertido

    def decodificar(matriz):
        convertido = pd.DataFrame(matriz, columns=colunas)
        for coluna, opcoes in categorias.items():
            convertido[coluna] = [
                opcoes[int(codigo)] if 0 <= int(codigo) < len(opcoes)
                else "__categoria_desconhecida__"
                for codigo in convertido[coluna]
            ]
        return convertido

    return codificar, decodificar


def selecionar_exemplos(modelos, X_test, y_test):
    """Escolhe casos comparáveis e uma pequena amostra de todas as classes."""
    previsoes = {nome: modelo.predict(X_test) for nome, modelo in modelos.items()}
    reais = y_test.to_numpy()
    todos_acertaram = np.logical_and.reduce([
        previsto == reais for previsto in previsoes.values()
    ])
    houve_erro = np.logical_or.reduce([
        previsto != reais for previsto in previsoes.values()
    ])
    pos_acerto = np.flatnonzero(todos_acertaram)
    pos_erro = np.flatnonzero(houve_erro)
    if not len(pos_acerto) or not len(pos_erro):
        raise ValueError("É necessário haver um acerto conjunto e pelo menos um erro.")

    casos = {
        "acerto_dos_tres": X_test.index[pos_acerto[0]],
        "erro_em_algum_modelo": X_test.index[pos_erro[0]],
    }
    # A amostra equilibrada permite observar também a classe mais rara.
    indices = set(y_test.groupby(y_test).sample(
        n=AMOSTRAS_POR_CLASSE, random_state=42
    ).index)
    indices.update(casos.values())
    X_amostra = X_test.loc[sorted(indices)]
    y_amostra = y_test.loc[X_amostra.index]
    return X_amostra, y_amostra, casos


def salvar_grafico_local(nome, caso, classe, valores, exemplo, base, previsto):
    """Plota os valores SHAP da classe prevista, em pontos percentuais."""
    ordem = np.argsort(np.abs(valores))
    rotulos = [f"{coluna} = {exemplo[coluna]}" for coluna in exemplo.index]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(
        np.array(rotulos)[ordem],
        np.array(valores)[ordem] * 100,
        color=["#c14953" if valores[i] > 0 else "#3279a8" for i in ordem],
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribuição SHAP para a probabilidade (pontos percentuais)")
    ax.set_title(
        f"{nome.replace('_', ' ')} — {caso.replace('_', ' ')}\n"
        f"Classe prevista: {classe}"
    )
    ax.text(
        0.01, -0.18,
        f"Referência: {base:.1%}  +  contribuições: {sum(valores):+.1%}"
        f"  =  previsão: {previsto:.1%}",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / f"{nome}_{caso}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def interpretar_modelos(modelos, X_train, X_test, y_test):
    """Calcula SHAP da probabilidade das quatro classes para os três modelos."""
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    X_amostra, y_amostra, casos = selecionar_exemplos(modelos, X_test, y_test)

    referencia = X_train.sample(n=TAMANHO_REFERENCIA, random_state=42)
    codificar, decodificar = criar_codificacao_temporaria(X_train)
    referencia_codificada = codificar(referencia)
    amostra_codificada = codificar(X_amostra)
    max_evals = 2 * X_train.shape[1] * PERMUTACOES + 1

    amostra_exportada = X_amostra.copy()
    amostra_exportada.insert(0, "classe_real", y_amostra)
    importancias = {}
    linhas = [
        "INTERPRETABILIDADE COM SHAP",
        "Método: SHAP por permutação aplicado a predict_proba de cada pipeline.",
        f"Referência: {len(referencia)} registros sorteados somente do treino (seed 42).",
        f"Amostra de teste: {len(X_amostra)} registros; "
        f"{AMOSTRAS_POR_CLASSE} por classe mais os dois casos comparados.",
        f"Orçamento de avaliações SHAP por registro: {max_evals} "
        f"({PERMUTACOES} permutações completas).",
        "Categorias são códigos temporários no SHAP e são decodificadas antes",
        "da previsão. O pré-processamento de cada pipeline permanece o mesmo.",
        "Valores SHAP e importâncias estão na escala de probabilidade.",
        "Importância global = média do valor SHAP absoluto sobre os registros",
        "amostrados e as quatro classes. A amostra é pequena e equilibrada:",
        "serve à interpretação exploratória, não representa a frequência real das classes.",
        "SHAP explica previsões do modelo em relação à referência; não demonstra causalidade.",
    ]

    for nome, modelo in modelos.items():
        print(f"Calculando SHAP para {nome} ({len(X_amostra)} registros)...", flush=True)

        def predizer_probabilidades(matriz):
            return modelo.predict_proba(decodificar(matriz))

        explainer = shap.Explainer(
            predizer_probabilidades,
            referencia_codificada,
            algorithm="permutation",
            output_names=list(modelo.classes_),
            seed=42,
        )
        explicacoes = explainer(
            amostra_codificada,
            max_evals=max_evals,
            silent=True,
        )
        probabilidades = modelo.predict_proba(X_amostra)
        reconstrucao = explicacoes.base_values + explicacoes.values.sum(axis=1)
        if not np.allclose(reconstrucao, probabilidades, atol=1e-6):
            raise RuntimeError(f"Valores SHAP não reproduzem predict_proba de {nome}.")

        importancia = np.abs(explicacoes.values).mean(axis=(0, 2))
        importancias[nome] = pd.Series(importancia, index=X_train.columns)
        amostra_exportada[f"previsao_{nome}"] = modelo.predict(X_amostra)

        linhas.extend([f"\n--- {nome} ---", "Importâncias globais:"])
        linhas.append(importancias[nome].sort_values(ascending=False).to_string())
        for caso, indice in casos.items():
            posicao = X_amostra.index.get_loc(indice)
            classe = modelo.predict(X_amostra.loc[[indice]])[0]
            coluna_classe = list(modelo.classes_).index(classe)
            base = float(explicacoes.base_values[posicao, coluna_classe])
            valores = explicacoes.values[posicao, :, coluna_classe]
            previsto = float(probabilidades[posicao, coluna_classe])
            exemplo = X_amostra.loc[indice]

            salvar_grafico_local(
                nome, caso, classe, valores, exemplo, base, previsto
            )
            contribuicoes = pd.Series(valores, index=X_train.columns)
            linhas.extend([
                f"\n{caso}: índice {indice}; classe real: {y_amostra.loc[indice]}",
                f"Classe prevista: {classe}; probabilidade: {previsto:.2%}",
                f"Referência: {base:.2%}; soma SHAP: {valores.sum():+.2%}",
                "Contribuições à probabilidade da classe prevista:",
                contribuicoes.reindex(
                    contribuicoes.abs().sort_values(ascending=False).index
                ).to_string(),
            ])

    tabela = pd.DataFrame(importancias)
    tabela.index.name = "variavel"
    tabela.to_csv(PASTA_RESULTADOS / "comparacao_importancias.csv")
    amostra_exportada.to_csv(PASTA_RESULTADOS / "amostra_avaliada.csv", index_label="indice_csv")
    (PASTA_RESULTADOS / "explicacoes.txt").write_text(
        "\n".join(linhas) + "\n", encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    tabela_para_grafico = tabela.loc[tabela.mean(axis=1).sort_values().index]
    tabela_para_grafico = tabela_para_grafico.rename(columns=lambda nome: nome.replace("_", " "))
    tabela_para_grafico.plot.barh(ax=ax)
    ax.set_xlabel("Média de |SHAP| nas probabilidades das quatro classes")
    ax.set_ylabel("")
    ax.set_title("Comparação das variáveis usadas pelos três modelos")
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / "comparacao_importancias.png", dpi=160)
    plt.close(fig)

    print("\nImportância SHAP média na amostra:")
    print(tabela.round(4).to_string())
    print(f"\nGráficos, amostra e relatório salvos em: {PASTA_RESULTADOS}")
