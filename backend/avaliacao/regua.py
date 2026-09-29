# -*- coding: utf-8 -*-
"""A régua determinística da suíte.

Funções puras que comparam o que a execução produziu com o gabarito declarado no
caso. Nenhuma delas chama modelo: tudo aqui é medido sobre dado que o próprio
sistema registrou — o caminho no grafo, a classificação da triagem, os trechos
recuperados, as ferramentas executadas, os contadores. O resultado é
reprodutível e não custa token.

Cada critério existe porque uma decisão de arquitetura deixou o valor à mão: o
caminho existe porque cada node escreve o próprio nome; a classificação existe
porque a triagem devolve um `Literal`, e não uma frase; as ferramentas existem
porque a execução é do nosso código, e não do modelo.

O que **não** é critério:

- `groundedness`: similaridade cosseno entre a resposta e o trecho mais próximo.
  Mede proximidade de assunto, não o que a resposta afirma — "pode" e "não pode"
  ficam no mesmo ponto — e dá nota baixa à recusa correta, que não se parece com
  documento nenhum. É sempre reportado, nunca decide.
- `fontes` como "fontes citadas": neste sistema a lista é o que a busca
  **recuperou**, na ordem do ranking, e não o que a resposta usou. O critério
  que existe é o honesto: o documento esperado está entre os recuperados.
"""
from __future__ import annotations

from typing import Any, NamedTuple

from app import dados, regras
from app.grafo import TEXTO_FORA_DE_ESCOPO, TEXTO_SEM_HISTORICO
from avaliacao.isolamento import fila_isolada


class Veredito(NamedTuple):
    """O resultado de um critério aplicado a uma execução.

    `avaliado=False` marca o critério que não pôde ser medido. Ele aparece na
    saída, mas não entra na conta: um critério não medido não é um critério
    reprovado — e também não é um critério aprovado.
    """

    criterio: str
    ok: bool
    detalhe: str
    avaliado: bool = True


# Os textos que o próprio sistema escreve sem passar pelo modelo. Comparáveis
# por igualdade exata, e por isso o critério mais barato que existe.
TEXTOS_FIXOS = {
    "fora_de_escopo": TEXTO_FORA_DE_ESCOPO,
    "sem_historico": TEXTO_SEM_HISTORICO,
}


def _resposta(estado: dict):
    atendimento = estado.get("atendimento")
    return atendimento.resposta if atendimento is not None else None


def _e_subsequencia(esperados: list[str], observados: list[str]) -> bool:
    """`esperados` aparece em `observados` nessa ordem, sem exigir adjacência."""
    restante = iter(observados)
    return all(node in restante for node in esperados)


# --- Critérios ---------------------------------------------------------------


def rota(estado: dict, sonda, esperado: list[str]) -> Veredito:
    """Os nodes esperados foram percorridos, nessa ordem relativa.

    Comparação por **subsequência**, nunca por lista igual. O caminho traz todos
    os nodes do turno, e a ordem em que os dois ramos paralelos terminam é
    decisão interna do LangGraph: estável na prática, mas não contrato. Amarrar
    o critério nela fabricaria vermelho sem o sistema ter mudado.
    """
    observada = estado.get("trajetoria") or []
    if _e_subsequencia(esperado, observada):
        return Veredito("rota", True, " → ".join(esperado))
    return Veredito("rota", False, f"esperado {' → '.join(esperado)}; observado {' → '.join(observada)}")


def nao_passa_por(estado: dict, sonda, esperado: list[str]) -> Veredito:
    """Nenhum destes nodes foi visitado."""
    visitados = [node for node in esperado if node in (estado.get("trajetoria") or [])]
    if not visitados:
        return Veredito("nao_passa_por", True, ", ".join(esperado))
    return Veredito("nao_passa_por", False, f"passou por {', '.join(visitados)}")


def escopo(estado: dict, sonda, esperado: str) -> Veredito:
    """A triagem classificou a mensagem como o esperado.

    Mede a primeira decisão do grafo isoladamente: a rota inteira depende dela,
    e quando a rota falha este critério diz se o problema nasceu aqui.
    """
    observado = estado.get("escopo")
    if observado == esperado:
        return Veredito("escopo", True, observado)
    return Veredito("escopo", False, f"esperado {esperado}; observado {observado}")


def fontes_incluem(estado: dict, sonda, esperado: list[str]) -> Veredito:
    """Os documentos esperados estão entre os que a busca recuperou.

    Inclusão, e não igualdade: a busca devolve sempre `k` trechos, e os demais
    documentos da lista são vizinhos de ranking, não erro.
    """
    recuperados = estado.get("fontes") or []
    faltando = [arquivo for arquivo in esperado if arquivo not in recuperados]
    if not faltando:
        return Veredito("fontes", True, ", ".join(esperado))
    return Veredito("fontes", False, f"não recuperou {', '.join(faltando)}")


def recusa(estado: dict, sonda, esperado: bool) -> Veredito:
    """A resposta admitiu não ter a informação — com a base respondendo.

    O sinal é `precisa_de_humano`, o mesmo campo que o próprio sistema usa para
    decidir se amplia a busca. Ele sozinho não basta: falha da base também liga
    esse campo, e uma recusa por banco fora do ar passaria por recusa correta.
    Por isso a segunda condição — houve trecho recuperado. A recusa que conta é
    a que leu a base e não achou.
    """
    resposta = _resposta(estado)
    if resposta is None:
        return Veredito("recusa", False, "execução sem resposta")
    houve_trecho = bool(estado.get("trechos"))
    recusou = resposta.precisa_de_humano and houve_trecho
    if recusou == esperado:
        return Veredito("recusa", True, "recusou e encaminhou" if recusou else "respondeu sem encaminhar")
    if esperado and resposta.precisa_de_humano and not houve_trecho:
        return Veredito("recusa", False, "encaminhou sem trecho recuperado: é falha da base, não recusa")
    if esperado:
        return Veredito("recusa", False, "esperava recusa; respondeu sem encaminhar")
    return Veredito("recusa", False, "esperava resposta; recusou e encaminhou")


def texto_fixo(estado: dict, sonda, esperado: str) -> Veredito:
    """A resposta é, letra por letra, o texto fixo que o sistema escreve sozinho."""
    resposta = _resposta(estado)
    alvo = TEXTOS_FIXOS[esperado]
    if resposta is not None and resposta.resposta == alvo:
        return Veredito("texto_fixo", True, esperado)
    return Veredito("texto_fixo", False, f"esperado o texto fixo {esperado}")


def ferramenta(estado: dict, sonda, esperado: dict | None) -> Veredito:
    """A ferramenta esperada foi executada, com os argumentos esperados.

    Lê a sonda, e não o estado: ela vê a execução de qualquer node, inclusive a
    do ramo paralelo, que não deixa rastro no estado. Os argumentos são
    comparados por **subconjunto** — um argumento de texto livre, como o motivo
    de uma troca, é redação do modelo, e o caso só declara o que tem gabarito.

    `null` no caso declara o contrário: nenhuma ferramenta pode ter rodado.
    """
    executadas = sonda.ferramentas
    if esperado is None:
        if not executadas:
            return Veredito("ferramenta", True, "nenhuma executada")
        nomes = ", ".join(f["nome"] for f in executadas)
        return Veredito("ferramenta", False, f"esperava nenhuma; executou {nomes}")
    nome = esperado["nome"]
    argumentos = esperado.get("argumentos", {})
    for f in executadas:
        if f["nome"] == nome and all(f["argumentos"].get(k) == v for k, v in argumentos.items()):
            return Veredito("ferramenta", True, f"{nome}({_formatar(argumentos)})")
    observadas = "; ".join(f"{f['nome']}({_formatar(f['argumentos'])})" for f in executadas) or "nenhuma"
    return Veredito("ferramenta", False, f"esperado {nome}({_formatar(argumentos)}); executou {observadas}")


def contem(estado: dict, sonda, esperado: list[str]) -> Veredito:
    """O texto da resposta contém o que uma resposta correta precisa afirmar.

    Existe porque os outros critérios medem o **caminho** — a rota, o documento
    recuperado, a ferramenta — e não o **conteúdo**. Sem este, uma resposta que
    recuperou a política certa e afirmou o prazo errado passaria em todos eles:
    documento certo é indício de resposta certa, não prova.

    Os trechos esperados vêm do gabarito — o que o documento ou a ferramenta de
    fato dizem —, nunca da resposta observada. Sem diferenciar maiúsculas: a caixa
    do texto não é o que se afirma.
    """
    resposta = _resposta(estado)
    texto = resposta.resposta.lower() if resposta is not None else ""
    faltando = [trecho for trecho in esperado if trecho.lower() not in texto]
    if not faltando:
        return Veredito("contem", True, ", ".join(esperado))
    return Veredito("contem", False, f"não afirma {', '.join(faltando)}")


def tipo(estado: dict, sonda, esperado: str) -> Veredito:
    """O assunto que a resposta declarou é o esperado.

    Só é critério porque o campo é um conjunto **fechado** no schema
    (`TipoDeAtendimento`): o modelo escolhe um valor da lista e não tem como
    devolver uma variação dele. Se fosse texto livre com sugestões na
    descrição, um acento ou um espaço reprovaria o caso sem o sistema ter
    piorado — e um critério que falha sozinho ensina o time a ignorar o
    vermelho de todos os outros.
    """
    resposta = _resposta(estado)
    observado = resposta.tipo.value if resposta is not None else None
    if observado == esperado:
        return Veredito("tipo", True, observado)
    return Veredito("tipo", False, f"esperado {esperado}; observado {observado}")


def tentativas(estado: dict, sonda, esperado: int) -> Veredito:
    """O ciclo de ampliação da busca rodou o número esperado de vezes."""
    observado = estado.get("tentativas", 0)
    if observado == esperado:
        return Veredito("tentativas", True, str(observado))
    return Veredito("tentativas", False, f"esperado {esperado}; observado {observado}")


def ramos(estado: dict, sonda, esperado: bool) -> Veredito:
    """Os dois ramos paralelos entregaram resposta parcial à junção.

    O estado zera esses campos com texto vazio, e não com `None`, a cada turno:
    presença aqui é texto não vazio.
    """
    presentes = {
        "pedido": bool(estado.get("resposta_do_pedido")),
        "regra": bool(estado.get("resposta_da_regra")),
    }
    ambos = all(presentes.values())
    if ambos == esperado:
        return Veredito("ramos", True, "pedido e regra presentes" if ambos else "sem ramos")
    ausentes = [nome for nome, ok in presentes.items() if not ok]
    return Veredito("ramos", False, f"ramo ausente: {', '.join(ausentes)}")


def bifurcacao(estado: dict, sonda, esperado: str) -> Veredito:
    """A pergunta composta foi separada do jeito que o schema pede.

    O número do pedido (`esperado`) vai na metade do pedido e fica **fora** da
    metade da regra — é essa metade que vira consulta na busca vetorial, e um
    número de pedido não descreve assunto nenhum.
    """
    do_pedido = estado.get("parte_do_pedido") or ""
    da_regra = estado.get("parte_da_regra") or ""
    problemas = []
    if esperado not in do_pedido:
        problemas.append("número ausente da parte do pedido")
    if esperado in da_regra:
        problemas.append("número presente na parte da regra")
    if not problemas:
        return Veredito("bifurcacao", True, f"regra sem {esperado}")
    return Veredito("bifurcacao", False, "; ".join(problemas))


def regra(estado: dict, sonda, esperado: str) -> Veredito:
    """A regra de troca barrou o pedido, e a recusa chegou ao modelo.

    A regra é código (`regras.impedimento_para_troca`), e não instrução ao
    modelo — então o critério compara com a saída da **própria função**, e não
    com uma frase escrita à mão no caso. Se a regra mudar, o gabarito muda junto;
    o que o critério afirma é que o fluxo a acionou e devolveu o que ela disse.

    A regra é calculada com a fila vazia, como a execução a encontrou.
    """
    pedido = dados.buscar_pedido(esperado)
    if pedido is None:
        return Veredito("regra", False, f"pedido {esperado} não existe na fixture")
    with fila_isolada():
        impedimento = regras.impedimento_para_troca(pedido)
    if impedimento is None:
        return Veredito("regra", False, f"o pedido {esperado} não tem impedimento — o caso não exercita a regra")
    saidas = " ".join(f["saida"] or "" for f in sonda.ferramentas)
    if impedimento in saidas:
        return Veredito("regra", True, impedimento)
    return Veredito("regra", False, f"a regra não chegou ao fluxo para o pedido {esperado}")


def _formatar(argumentos: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in argumentos.items())


# Os critérios que a régua sabe medir. Um critério declarado num caso e ausente
# daqui aparece como não avaliado — nunca é contado como aprovado no silêncio.
CRITERIOS = {
    "rota": rota,
    "nao_passa_por": nao_passa_por,
    "escopo": escopo,
    "fontes": fontes_incluem,
    "contem": contem,
    "tipo": tipo,
    "recusa": recusa,
    "texto_fixo": texto_fixo,
    "ferramenta": ferramenta,
    "tentativas": tentativas,
    "ramos": ramos,
    "bifurcacao": bifurcacao,
    "regra": regra,
}


def avaliar(estado: dict, sonda, criterios: dict[str, Any]) -> list[Veredito]:
    """Aplica ao que a execução produziu os critérios do caso que a régua mede."""
    return [
        CRITERIOS[nome](estado, sonda, esperado)
        for nome, esperado in criterios.items()
        if nome in CRITERIOS
    ]
