# -*- coding: utf-8 -*-
"""O fluxo do atendimento, declarado como grafo.

Até aqui o fluxo existia, mas só implicitamente: era o corpo de `responder()` —
um `if` de modo, um `for` de ferramenta e um `return` no fim. Funcionava, e o
problema não era o funcionamento: era não haver onde *ler* o fluxo. Para saber
o que acontece com uma pergunta era preciso executar a função na cabeça.

Aqui o mesmo fluxo é declarado. Cada etapa é um **node** — uma função que
recebe o estado e devolve o que mudou nele. Cada decisão é uma **edge
condicional** — uma função que só escolhe o próximo node, e não faz trabalho
nenhum. O que era variável local de `responder()` passa a ser campo do estado,
e o estado é o único canal entre os nodes.

O que se ganha não é desempenho: é topologia. O grafo se desenha
(`GRAFO.get_graph().draw_mermaid()`), e capacidade nova entra como node novo em
vez de `if` mais fundo dentro de uma função que já era grande.

O trabalho em si **não** mudou de lugar. As funções que montam o contexto,
executam ferramenta, contam token e traduzem erro do Bedrock continuam em
`assistente.py`, e são chamadas daqui. Este arquivo é a topologia; aquele é o
trabalho.

    START → rota_por_modo (condicional)
              com_busca → recuperar → conversar
              sem_busca →             conversar
    conversar → rota_da_ferramenta (condicional)
              ferramenta → executar_ferramentas → conversar   ← ciclo
              formalizar → formalizar → END
"""
import logging
from typing import TypedDict

from botocore.exceptions import ClientError
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph

# As funções de trabalho continuam em `assistente.py`; o que este arquivo faz é
# ligá-las numa ordem legível. O import é nominal de propósito: a lista abaixo é
# o inventário exato do que a topologia precisa para funcionar.
from app.assistente import (
    FERRAMENTAS,
    MAXIMO_DE_PASSOS,
    _executar_ferramenta,
    _falha_da_base,
    _fontes,
    _mensagem_de_contexto,
    _tokens_de_entrada,
    _trechos_do_modo,
    montar_modelo,
    prompt,
    traduzir_erro_do_bedrock,
)
from app.config import MODELOS, PERFIS_DE_ATENDIMENTO
from app.schemas import Atendimento, RespostaAtendimento, TipoDeAtendimento

logger = logging.getLogger(__name__)

# O único modo que não precisa montar contexto nenhum. Todos os outros passam
# pelo node `recuperar` — inclusive o stuffing, que não faz busca mas carrega a
# base do disco. É o mesmo node porque é a mesma pergunta ("o que entra no
# contexto?"), respondida por `_trechos_do_modo`.
MODO_SEM_CONTEXTO = "sem_conhecimento"


class EstadoAtendimento(TypedDict, total=False):
    """O que flui entre os nodes.

    `total=False` porque cada node devolve um **dicionário parcial**: só as
    chaves que ele mudou. O LangGraph sobrescreve essas chaves no estado e passa
    adiante — não há reducer aqui, e por isso não há acúmulo automático.

    Os campos são, um a um, as variáveis locais que `responder()` tinha. A
    diferença é que agora elas têm nome no contrato, e qualquer node pode ler o
    que outro escreveu sem ninguém passar parâmetro para ninguém.
    """

    # Entrada — o que a interface escolheu
    pergunta: str
    perfil: str
    modelo: str
    temperatura: float
    modo: str
    # Intermediários — o que era variável local do fluxo
    mensagens: list
    trechos: list[tuple[str, str]]
    fontes: list[str]
    tokens_de_entrada: int
    passos: int
    top_k: int
    # Saída — o que `responder()` devolve
    atendimento: Atendimento


# --- Auxiliares dos nodes ---------------------------------------------------


def _abrir_conversa(state: EstadoAtendimento) -> list:
    """Monta as mensagens iniciais: sistema com o tom, contexto e a pergunta.

    A mensagem de contexto entra depois do `format_messages`, e é aí que ela
    escapa da formatação de template — um `{prazo}` escrito dentro de um
    documento seria lido como variável e a chamada quebraria. Vai antes da
    mensagem do cliente para o modelo ler a base antes de ler a pergunta.
    """
    mensagens = prompt.format_messages(
        instrucao_de_tom=PERFIS_DE_ATENDIMENTO[state["perfil"]]["instrucao_de_tom"],
        pergunta=state["pergunta"],
    )
    trechos = state.get("trechos")
    if trechos:
        mensagens.insert(-1, _mensagem_de_contexto(trechos))
    return mensagens


def _falha_do_bedrock(state: EstadoAtendimento, erro: ClientError) -> dict:
    """A falha do Bedrock sai no formato do schema, e não como exceção.

    Um node que levanta derruba o `invoke()` do grafo inteiro e leva embora o
    estado parcial. Devolvendo `atendimento` aqui, a rota seguinte vê que já há
    resposta e termina — e o que já foi medido antes da falha continua valendo:
    se a busca chegou a rodar, ela já custou, e esconder isso mentiria sobre o
    gasto da pergunta.
    """
    logger.warning("falha do Bedrock: %s", erro)
    return {
        "atendimento": Atendimento(
            resposta=RespostaAtendimento(
                resposta=traduzir_erro_do_bedrock(
                    erro, MODELOS[state["modelo"]]["model_id"]
                ),
                tipo=TipoDeAtendimento.OUTRO,
            ),
            tokens_de_entrada=state.get("tokens_de_entrada", 0),
            fontes=state.get("fontes", []),
        )
    }


# --- Nodes ------------------------------------------------------------------


def recuperar(state: EstadoAtendimento) -> dict:
    """Monta o contexto do modo escolhido e registra as fontes.

    Tem o seu próprio tratamento de erro porque falha aqui tem causa e conserto
    diferentes de falha do modelo — e porque o `langchain-postgres` embrulha erro
    de conexão numa Exception genérica. Sem isto, banco fora do ar viraria 500
    com traceback na tela.
    """
    modo = state["modo"]
    try:
        trechos = _trechos_do_modo(modo, state["pergunta"])
    except Exception as erro:
        logger.exception("falha ao montar o contexto no modo %s", modo)
        return {"atendimento": Atendimento(resposta=_falha_da_base(erro))}

    fontes = _fontes(modo, trechos)
    logger.info("[grafo] recuperar modo=%s trechos=%d", modo, len(trechos))
    return {"trechos": trechos, "fontes": fontes}


def conversar(state: EstadoAtendimento) -> dict:
    """Uma ida ao modelo com as ferramentas plugadas.

    Era o corpo do `for` de `responder()`; virou um node que roda **uma** volta.
    Quem repete é a edge: `executar_ferramentas` volta para cá. O contador
    `passos` é o mesmo teto de antes, agora guardado no estado em vez de ser o
    índice de um laço.

    Só a primeira ida tem o token medido: é a que carrega os documentos, e é o
    número que dá para comparar entre os modos.
    """
    mensagens = state.get("mensagens") or _abrir_conversa(state)
    passos = state.get("passos", 0)

    model = montar_modelo(state["modelo"], state["temperatura"])
    logger.info("[grafo] conversar passo=%d", passos + 1)
    try:
        resposta_do_modelo = model.bind_tools(FERRAMENTAS).invoke(mensagens)
    except ClientError as erro:
        return _falha_do_bedrock(state, erro)

    mensagens.append(resposta_do_modelo)
    mudou = {"mensagens": mensagens, "passos": passos + 1}
    if passos == 0:
        mudou["tokens_de_entrada"] = _tokens_de_entrada(resposta_do_modelo)
    return mudou


def executar_ferramentas(state: EstadoAtendimento) -> dict:
    """Roda o que o modelo pediu e devolve cada resultado numa ToolMessage.

    A intenção chega em `tool_calls`, com nome, argumentos e um id. O resultado
    volta amarrado a esse id, senão o modelo não sabe a qual pedido a resposta
    corresponde.
    """
    mensagens = state["mensagens"]
    for chamada in mensagens[-1].tool_calls:
        mensagens.append(
            ToolMessage(
                content=_executar_ferramenta(chamada),
                tool_call_id=chamada["id"],
            )
        )
    return {"mensagens": mensagens}


def formalizar(state: EstadoAtendimento) -> dict:
    """A última ida ao modelo: a mesma conversa, agora exigindo o formato.

    Fica num node separado de `conversar` porque `bind_tools` e
    `with_structured_output` disputam o mesmo mecanismo por baixo — pedir as
    duas coisas na mesma chamada é frágil. A separação que antes eram duas
    etapas dentro de uma função agora é visível na topologia.
    """
    model = montar_modelo(state["modelo"], state["temperatura"])
    logger.info("[grafo] formalizar")
    try:
        resposta = model.with_structured_output(RespostaAtendimento).invoke(
            state["mensagens"]
        )
    except ClientError as erro:
        return _falha_do_bedrock(state, erro)

    return {
        "atendimento": Atendimento(
            resposta=resposta,
            tokens_de_entrada=state.get("tokens_de_entrada", 0),
            fontes=state.get("fontes", []),
        )
    }


# --- Edges condicionais -----------------------------------------------------
#
# Uma rota só escolhe o próximo node: devolve uma string e não toca no estado.
# Manter isso separado do trabalho é o que faz a decisão ser legível — e o que
# permite mudar o caminho sem mexer em quem executa.


def rota_por_modo(state: EstadoAtendimento) -> str:
    """A decisão que era o `if modo` dentro de `_trechos_do_modo`."""
    return "sem_busca" if state["modo"] == MODO_SEM_CONTEXTO else "com_busca"


def rota_apos_recuperar(state: EstadoAtendimento) -> str:
    """Base fora do ar já produziu resposta: não há o que perguntar ao modelo."""
    return "fim" if state.get("atendimento") else "conversar"


def rota_da_ferramenta(state: EstadoAtendimento) -> str:
    """A decisão que era o `break` do laço de ferramenta, com o mesmo teto.

    Sem pedido de ferramenta, o modelo já tem o que precisa. Com pedido, o ciclo
    continua — até `MAXIMO_DE_PASSOS`, porque laço sem limite é onde um agente
    trava: o modelo pode continuar pedindo ferramenta indefinidamente.
    """
    if state.get("atendimento"):
        return "fim"
    if state["mensagens"][-1].tool_calls and state["passos"] < MAXIMO_DE_PASSOS:
        return "ferramenta"
    return "formalizar"


# --- Montagem ---------------------------------------------------------------


def _construir_grafo():
    """Declara os nodes, liga as edges e compila. Roda uma vez, no import."""
    grafo = StateGraph(EstadoAtendimento)

    grafo.add_node("recuperar", recuperar)
    grafo.add_node("conversar", conversar)
    grafo.add_node("executar_ferramentas", executar_ferramentas)
    grafo.add_node("formalizar", formalizar)

    grafo.add_conditional_edges(
        START,
        rota_por_modo,
        {"com_busca": "recuperar", "sem_busca": "conversar"},
    )
    grafo.add_conditional_edges(
        "recuperar",
        rota_apos_recuperar,
        {"conversar": "conversar", "fim": END},
    )
    # O ciclo de ferramenta: a volta `executar_ferramentas → conversar` é a
    # aresta que antes era a próxima iteração do `for`.
    grafo.add_conditional_edges(
        "conversar",
        rota_da_ferramenta,
        {
            "ferramenta": "executar_ferramentas",
            "formalizar": "formalizar",
            "fim": END,
        },
    )
    grafo.add_edge("executar_ferramentas", "conversar")
    grafo.add_edge("formalizar", END)

    return grafo.compile()


# Compilado uma única vez, no import do módulo: compilar por pergunta seria
# refazer a mesma montagem a cada requisição.
GRAFO = _construir_grafo()
