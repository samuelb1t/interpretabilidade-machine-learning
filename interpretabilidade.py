from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Salva os gráficos sem abrir janelas.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import export_text, plot_tree


def explicar_previsao_arvore(modelo, X_test, y_test, posicao):
    """Explica um registro pela sua posição no conjunto de teste."""
    exemplo = X_test.iloc[[posicao]]
    preprocessamento = modelo.named_steps["preprocessamento"]
    classificador = modelo.named_steps["classificador"]
    # A árvore usa float32 internamente para comparar valores com os limiares.
    valores = np.asarray(preprocessamento.transform(exemplo), dtype=np.float32)
    nomes = preprocessamento.get_feature_names_out()
    estrutura = classificador.tree_
    folha = classificador.apply(valores)[0]
    caminho = classificador.decision_path(valores).indices
    prevista = modelo.predict(exemplo)[0]
    probabilidades = modelo.predict_proba(exemplo)[0]

    linhas = [
        f"Registro de índice {exemplo.index[0]} no CSV carregado",
        f"Classe real: {y_test.iloc[posicao]}",
        f"Classe prevista: {prevista}",
        "\nCaracterísticas originais:",
        exemplo.iloc[0].to_string(),
        "\nCondições percorridas (da raiz até a folha):",
    ]
    for no in caminho:
        if no == folha:
            continue
        coluna = estrutura.feature[no]
        valor = valores[0, coluna]
        limite = estrutura.threshold[no]
        operador = "<=" if valor <= limite else ">"
        linhas.append(
            f"Nó {no}: {nomes[coluna]} = {valor:.6g} {operador} {limite:.6g}"
        )

    linhas.append(f"Folha {folha}: {estrutura.n_node_samples[folha]} registros de treino.")
    linhas.append("Probabilidades estimadas pela proporção de classes na folha:")
    for classe, probabilidade in zip(classificador.classes_, probabilidades):
        linhas.append(f"  {classe}: {probabilidade:.2%}")
    return "\n".join(linhas)


def interpretar_arvore(modelo, X_test, y_test):
    """Gera importância global, gráfico da árvore e dois exemplos locais."""
    pasta = Path(__file__).parent / "resultados" / "arvore"
    pasta.mkdir(parents=True, exist_ok=True)
    preprocessamento = modelo.named_steps["preprocessamento"]
    classificador = modelo.named_steps["classificador"]
    nomes = preprocessamento.get_feature_names_out()

    # Soma as importâncias das colunas one-hot para cada variável original.
    colunas_numericas = list(preprocessamento.transformers_[0][2])
    colunas_categoricas = list(preprocessamento.transformers_[1][2])
    encoder = preprocessamento.named_transformers_["categoricas"]
    origens = colunas_numericas.copy()
    for coluna, categorias in zip(colunas_categoricas, encoder.categories_):
        origens.extend([coluna] * len(categorias))
    importancias = pd.Series(classificador.feature_importances_, index=origens)
    importancias = importancias.groupby(level=0).sum().sort_values(ascending=False)
    importancias.rename("importancia").to_csv(pasta / "importancias.csv", index_label="variavel")

    fig, ax = plt.subplots(figsize=(10, 6))
    importancias.sort_values().plot.barh(ax=ax, color="#2878a5")
    ax.set_title("Árvore de Decisão: importância das variáveis no treino")
    ax.set_xlabel("Importância por redução de impureza (soma = 1)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(pasta / "importancias.png", dpi=160)
    plt.close(fig)

    # Limita apenas a visualização; o modelo treinado permanece completo.
    classes_curtas = {
        "Acute Brain Rot / Executive Burnout": "Acute",
        "Mild Attentional Strain": "Mild",
        "Normal / Intact Focus": "Normal",
        "Severe Focus Fragmentation": "Severe",
    }
    fig, ax = plt.subplots(figsize=(24, 10))
    plot_tree(
        classificador,
        feature_names=[nome.split("__", 1)[1] for nome in nomes],
        class_names=[classes_curtas.get(classe, classe) for classe in classificador.classes_],
        max_depth=2,
        filled=True,
        rounded=True,
        node_ids=True,
        precision=3,
        fontsize=10,
        ax=ax,
    )
    ax.set_title(
        "Primeiros níveis da árvore (ramos com (...) continuam)\n"
        "value: contagens de treino na ordem Acute, Mild, Normal, Severe",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(pasta / "primeiros_niveis.png", dpi=160)
    plt.close(fig)

    # SVG mantém o texto nítido ao ampliar. O dpi baixo evita criar um grande
    # buffer de pixels em memória; o arquivo salvo continua sendo vetorial.
    with plt.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(
            figsize=(classificador.get_n_leaves() * 3, (classificador.get_depth() + 1) * 2),
            dpi=12,
        )
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plot_tree(
            classificador,
            feature_names=[nome.split("__", 1)[1] for nome in nomes],
            class_names=[classes_curtas.get(classe, classe) for classe in classificador.classes_],
            max_depth=None,
            filled=True,
            rounded=True,
            node_ids=True,
            precision=3,
            fontsize=10,
            ax=ax,
        )
        fig.savefig(pasta / "arvore_completa.svg", format="svg")
        plt.close(fig)

    regras = export_text(
        classificador,
        feature_names=[nome.split("__", 1)[1] for nome in nomes],
        max_depth=classificador.get_depth(),
        decimals=6,
    )
    (pasta / "regras_completas.txt").write_text(regras, encoding="utf-8")

    previsoes = modelo.predict(X_test)
    acertos = previsoes == y_test.to_numpy()
    linhas = [
        "INTERPRETAÇÃO DA ÁRVORE DE DECISÃO",
        f"Profundidade completa: {classificador.get_depth()}",
        f"Número de folhas: {classificador.get_n_leaves()}",
        "\nImportância global (redução de impureza no treino):",
        importancias.to_string(),
        "\nImportância não indica causa nem direção do efeito. Pode favorecer variáveis",
        "com muitos pontos de corte. As importâncias de cada categoria foram somadas.",
        "primeiros_niveis.png mostra só os primeiros níveis; arvore_completa.svg",
        "mostra todos os nós. Os caminhos abaixo também são completos.",
        "Nas condições, colunas categóricas são indicadores: 1 = presente; 0 = ausente.",
        "Probabilidades de 100% em uma folha não garantem acerto em novos registros.",
        "Os exemplos são o primeiro acerto e o primeiro erro na ordem do teste,",
        "selecionados para ilustração; não representam todos os casos.",
    ]
    for titulo, mascara in [("EXEMPLO CORRETO", acertos), ("EXEMPLO INCORRETO", ~acertos)]:
        posicoes = np.flatnonzero(mascara)
        linhas.append(f"\n--- {titulo} ---")
        if len(posicoes):
            linhas.append(explicar_previsao_arvore(modelo, X_test, y_test, int(posicoes[0])))
        else:
            linhas.append("Não há exemplos desse tipo no conjunto de teste.")

    relatorio = "\n".join(linhas)
    (pasta / "explicacoes.txt").write_text(relatorio, encoding="utf-8")
    print("\n" + relatorio)
    print(f"\nGráficos e relatório salvos em: {pasta}")
