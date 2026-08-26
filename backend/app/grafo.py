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

    START → triagem → rota_apos_triagem (condicional)
              fora_de_escopo → resposta_direta → END
              com_busca      → recuperar → conversar
              sem_busca      →             conversar
    conversar → rota_da_ferramenta (condicional)
              ferramenta → executar_ferramentas → conversar   ← ciclo
              formalizar → formalizar → END
"""
import logging
from typing import Literal, TypedDict

from botocore.exceptions import ClientError
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

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
from app.config import MODELOS, PERFIS_DE_ATENDIMENTO, TEMPERATURA_MINIMA
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
    escopo: str
    mensagens: list
    trechos: list[tuple[str, str]]
    fontes: list[str]
    tokens_de_entrada: int
    passos: int
    top_k: int
    # Saída — o que `responder()` devolve
    atendimento: Atendimento


# A triagem é a primeira decisão do grafo, e a mais barata: uma chamada curta que
# só classifica, sem contexto e sem ferramenta. Ela existe para o que não é
# assunto da loja não chegar a pagar busca vetorial, ciclo de ferramenta e
# chamada de formato.
INSTRUCAO_DA_TRIAGEM = """Você é a triagem do atendimento de uma loja de produtos congelados. Classifique a mensagem do cliente em uma de duas categorias:

- "atendimento": qualquer assunto da loja — pedido, entrega, prazo, rastreamento, troca, devolução, reclamação, pagamento, nota fiscal, cupom, cadastro, assinatura, ou dúvida sobre produto, conservação, validade e preparo.
- "fora_de_escopo": saudação e conversa fiada, e temas alheios à loja (clima, esportes, notícias, piadas, receitas, conhecimento geral, pedido de ajuda com outro assunto).

Na dúvida entre as duas, classifique como "atendimento": o fluxo normal sabe recusar o que não está na base, e barrar um cliente legítimo é o erro mais caro dos dois.

Exemplos:
- "onde está o meu pedido 81030?" → atendimento
- "meu salmão chegou descongelado" → atendimento
- "vocês entregam em Curitiba?" → atendimento
- "posso congelar de novo depois de descongelar?" → atendimento
- "qual a previsão do tempo pra amanhã?" → fora_de_escopo
- "oi, tudo bem?" → fora_de_escopo
- "me conta uma piada" → fora_de_escopo"""

# O que o cliente ouve quando a triagem barra a mensagem. É texto fixo, e não
# uma chamada ao modelo: a recusa é sempre a mesma, então gerá-la seria pagar
# uma segunda ida ao Bedrock para escrever uma frase que já está escrita. O
# node novo economiza chamadas — e economizar duas em vez de uma é a diferença
# entre gastar 1 e gastar 3 nesta pergunta.
TEXTO_FORA_DE_ESCOPO = (
    "Aqui eu consigo ajudar só com o que é da loja: pedidos, entregas, trocas e "
    "dúvidas sobre os produtos. Sobre algum desses, como posso ajudar?"
)


class _Escopo(BaseModel):
    """A saída da triagem: um campo só, com duas opções fechadas.

    Structured output com `Literal` em vez de texto livre porque o valor vai
    alimentar uma rota do grafo. Uma classificação que volta como frase teria de
    ser interpretada por nós, e interpretar texto do modelo para decidir caminho
    é exatamente o que a saída estruturada existe para evitar.
    """

    escopo: Literal["atendimento", "fora_de_escopo"] = Field(
        description=(
            "'atendimento' se a mensagem é assunto da loja; 'fora_de_escopo' "
            "para saudação, conversa fiada e temas alheios à loja."
        )
    )


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


def triagem(state: EstadoAtendimento) -> dict:
    """Classifica a mensagem em "atendimento" ou "fora_de_escopo".

    Roda em temperatura mínima, e não na que a interface escolheu: a temperatura
    é um controle sobre a *redação* da resposta ao cliente, e classificação não
    é redação — a mesma mensagem tem de cair sempre no mesmo lado.

    **Fail-open.** Qualquer falha classifica como "atendimento". Um falso "fora"
    calaria um cliente legítimo; um falso "no escopo" só custa o fluxo normal,
    que já sabe recusar o que não está na base. Os dois erros não são
    simétricos, e a escolha do padrão segue o mais barato dos dois.
    """
    model = montar_modelo(state["modelo"], TEMPERATURA_MINIMA)
    try:
        classificacao = model.with_structured_output(_Escopo).invoke(
            [
                SystemMessage(content=INSTRUCAO_DA_TRIAGEM),
                HumanMessage(content=state["pergunta"]),
            ]
        )
        escopo = classificacao.escopo
    except Exception as erro:
        logger.warning("triagem falhou, seguindo como atendimento: %s", erro)
        escopo = "atendimento"

    logger.info("[triagem] %s pergunta=%r", escopo, state["pergunta"])
    return {"escopo": escopo}


def resposta_direta(state: EstadoAtendimento) -> dict:
    """A saída curta para o que a triagem barrou: nenhuma busca, nenhum modelo.

    Devolve `Atendimento` no mesmo formato de sempre — quem consome a API trata
    uma forma só. `precisa_de_humano` fica falso de propósito: mensagem fora do
    escopo não é trabalho para uma pessoa, e enfileirá-la encheria a fila de
    ruído.
    """
    logger.info("[grafo] resposta_direta (sem busca e sem chamada de geracao)")
    return {
        "atendimento": Atendimento(
            resposta=RespostaAtendimento(
                resposta=TEXTO_FORA_DE_ESCOPO,
                tipo=TipoDeAtendimento.OUTRO,
            )
        )
    }


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


def rota_apos_triagem(state: EstadoAtendimento) -> str:
    """Três saídas numa função só: barrar, buscar contexto, ou ir direto.

    É de propósito uma rota, e não duas condicionais em série (uma para o escopo,
    outra para o modo). O que se decide aqui é uma coisa só — por onde esta
    pergunta entra no fluxo — e ler as três possibilidades lado a lado é o que
    torna a topologia legível.

    A segunda metade é a decisão que era o `if modo` dentro de
    `_trechos_do_modo`. O stuffing entra por "com_busca" junto com os dois modos
    de RAG: ele não faz busca, mas precisa do mesmo node, porque a pergunta que
    o node responde é "o que entra no contexto?".
    """
    if state["escopo"] == "fora_de_escopo":
        return "fora_de_escopo"
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

    grafo.add_node("triagem", triagem)
    grafo.add_node("resposta_direta", resposta_direta)
    grafo.add_node("recuperar", recuperar)
    grafo.add_node("conversar", conversar)
    grafo.add_node("executar_ferramentas", executar_ferramentas)
    grafo.add_node("formalizar", formalizar)

    grafo.add_edge(START, "triagem")
    grafo.add_conditional_edges(
        "triagem",
        rota_apos_triagem,
        {
            "fora_de_escopo": "resposta_direta",
            "com_busca": "recuperar",
            "sem_busca": "conversar",
        },
    )
    grafo.add_edge("resposta_direta", END)
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
