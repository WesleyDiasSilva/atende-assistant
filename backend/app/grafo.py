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
(`compilar_grafo().get_graph().draw_mermaid()`), e capacidade nova entra como
node novo em vez de `if` mais fundo dentro de uma função que já era grande.

O trabalho em si **não** mudou de lugar. As funções que montam o contexto,
executam ferramenta, contam token e traduzem erro do Bedrock continuam em
`assistente.py`, e são chamadas daqui. Este arquivo é a topologia; aquele é o
trabalho.

    START → triagem → rota_apos_triagem (condicional)
              fora_de_escopo → resposta_direta → finalizar
              conversacional → responder_do_historico → finalizar
              composta       → bifurcar ─┬→ ramo_do_pedido ─┬→ juntar → finalizar
                                         └→ ramo_da_regra  ─┘   (em paralelo)
              com_busca      → recuperar → conversar
              sem_busca      →             conversar
    conversar → rota_da_ferramenta (condicional)
              ferramenta → executar_ferramentas → conversar   ← ciclo
              formalizar → formalizar
    formalizar → avaliar_groundedness → rota_apos_resposta (condicional)
              ampliar    → ampliar_busca → recuperar          ← ciclo
              encaminhar → encaminhar → finalizar
              fim        → finalizar
    finalizar → END
"""
import logging
import math
from typing import Annotated, Literal, TypedDict

from botocore.exceptions import ClientError
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app import falhas, retrieval

# As funções de trabalho continuam em `assistente.py`; o que este arquivo faz é
# ligá-las numa ordem legível. O import é nominal de propósito: a lista abaixo é
# o inventário exato do que a topologia precisa para funcionar.
from app.assistente import (
    FERRAMENTAS,
    MAXIMO_DE_PASSOS,
    MODOS_COM_BUSCA,
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
from app.config import (
    MODELOS,
    PERFIS_DE_ATENDIMENTO,
    TEMPERATURA_MINIMA,
    TOP_K_AMPLIADO,
)
from app.schemas import Atendimento, RespostaAtendimento, TipoDeAtendimento
from app.tools import consultar_status_pedido

logger = logging.getLogger(__name__)

# O único modo que não precisa montar contexto nenhum. Todos os outros passam
# pelo node `recuperar` — inclusive o stuffing, que não faz busca mas carrega a
# base do disco. É o mesmo node porque é a mesma pergunta ("o que entra no
# contexto?"), respondida por `_trechos_do_modo`.
MODO_SEM_CONTEXTO = "sem_conhecimento"

# Quantas vezes o fluxo pode ampliar a busca antes de desistir. Uma. O ciclo
# `ampliar_busca → recuperar → conversar → formalizar` volta ao mesmo ponto, e
# ciclo sem teto num grafo é o mesmo problema do laço sem teto numa função: ele
# não termina. O teto vive no estado (`tentativas`), não na estrutura, porque
# quem conta as voltas é o estado.
MAXIMO_DE_TENTATIVAS = 1

# O diagnóstico que vai para a fila humana quando nem a busca ampliada achou. É
# texto para quem vai resolver o caso, não para o cliente — são públicos
# diferentes, e a mesma falha precisa dizer coisas diferentes para cada um.
MOTIVO_DA_AMPLIACAO_SEM_RESULTADO = (
    f"A busca foi ampliada para {TOP_K_AMPLIADO} trechos e a base continuou sem "
    "a resposta. O caso precisa de um atendente."
)


def acumular_trajetoria(atual: list[str] | None, novo: list[str]) -> list[str]:
    """Soma o que cada node escreve no rastro — e reinicia quando vem lista vazia.

    Um **reducer** é o que permite mais de um escritor no mesmo campo do estado.
    Sem ele a regra é a da sobrescrita: o último a escrever apaga o que os
    anteriores escreveram, e o rastro teria sempre um nome só.

    A lista vazia é o pedido de reinício, e ela vem de `ESTADO_DO_TURNO`. Um
    reducer que só soma nunca esquece — e, com o estado gravado entre turnos, o
    rastro de ontem apareceria na frente do rastro de agora. Zerar por atribuição
    não existe num campo com reducer: a atribuição também passa por aqui.
    """
    if not novo:
        return []
    return (atual or []) + novo


class EstadoAtendimento(TypedDict, total=False):
    """O que flui entre os nodes.

    `total=False` porque cada node devolve um **dicionário parcial**: só as
    chaves que ele mudou. O LangGraph sobrescreve essas chaves no estado e passa
    adiante.

    Quase todos os campos são, um a um, as variáveis locais que `responder()`
    tinha. A diferença é que agora elas têm nome no contrato, e qualquer node
    pode ler o que outro escreveu sem ninguém passar parâmetro para ninguém.

    A exceção é `historico`, e ela é de outra natureza: não é variável de um
    turno, é a conversa. Ver o comentário do campo.
    """

    # Entrada — o que a interface escolheu
    pergunta: str
    perfil: str
    modelo: str
    temperatura: float
    modo: str
    auto_corrigir: bool
    # Se este turno usa a conversa anterior. Vive no estado, junto de
    # `auto_corrigir`, e pelo mesmo motivo: é um controle do fluxo, e o fluxo lê
    # os seus controles no estado. Ler a memória e gravar nela são dois pontos
    # diferentes do grafo, e os dois olham para esta chave.
    memoria_ativa: bool
    # A conversa até aqui, e o primeiro campo que acumulou.
    #
    # `add_messages` é um reducer: quando um node devolve algo nesta chave, o
    # LangGraph *soma* ao que já estava lá em vez de substituir. A maioria dos
    # campos sobrescreve, e é isso que se quer deles — `fontes` do turno passado
    # não tem nada a fazer no turno de agora. Aqui é o contrário: histórico que
    # sobrescreve é histórico de um turno só.
    #
    # Guarda apenas texto: um par pergunta/resposta por turno, gravado pelo node
    # `finalizar`. Ver lá por que só o texto entra.
    historico: Annotated[list, add_messages]
    # Por onde esta pergunta passou, na ordem em que os nodes concluíram.
    #
    # O segundo campo com reducer, e por um motivo diferente do `historico`:
    # aqui o campo tem **vários escritores dentro do mesmo turno**. Todo node
    # acrescenta o próprio nome, e num campo de sobrescrita cada um apagaria o
    # anterior. Ao contrário do histórico, este rastro recomeça a cada turno —
    # ver `acumular_trajetoria`.
    #
    # Guarda nome de node e ordem. Nada além disso.
    trajetoria: Annotated[list[str], acumular_trajetoria]
    # Intermediários — o que era variável local do fluxo
    escopo: str
    # As duas metades de uma pergunta composta, e as duas respostas parciais que
    # os ramos produzem a partir delas.
    #
    # São quatro campos de sobrescrita, e não um campo com reducer, porque o que
    # chega aqui não é a mesma coisa vinda de dois lugares: é o dado do pedido de
    # um lado e a regra da base do outro. Somar os dois numa lista entregaria ao
    # node de junção duas respostas sem etiqueta, e ele precisa saber qual é qual
    # para compor — a metade que veio do sistema vale como dado daquele cliente,
    # a que veio da base vale como política. Reducer resolve acúmulo; isto é
    # composição, e composição pede critério.
    #
    # Cada campo tem **um** escritor, que é o que os mantém sem reducer: dois
    # nodes concorrentes escrevendo no mesmo campo de sobrescrita é justamente o
    # que o LangGraph recusa.
    parte_do_pedido: str
    parte_da_regra: str
    resposta_do_pedido: str
    resposta_da_regra: str
    mensagens: list
    trechos: list[tuple[str, str]]
    fontes: list[str]
    tokens_de_entrada: int
    passos: int
    top_k: int
    tentativas: int
    # O quanto a resposta se parece com os trechos recuperados: o cosseno entre
    # o vetor da resposta e o do trecho mais próximo. `None` quando não houve
    # trecho recuperado — não é zero, é "não se aplica".
    groundedness: float | None
    # Saída — o que `responder()` devolve
    #
    # `None` enquanto ninguém produziu resposta ainda, e é isso que as rotas
    # leem para saber se o fluxo já terminou. Por isso a auto-correção precisa
    # zerá-lo antes de tentar de novo: sem reducer, o estado guarda o último
    # valor escrito, e a resposta da passada anterior continuaria ali dizendo
    # "já acabou".
    atendimento: Atendimento | None


# O que volta ao zero no começo de cada turno.
#
# Isto não existia enquanto o estado morria com a execução: cada `invoke()`
# começava de um estado vazio porque não havia estado anterior. Com o estado
# gravado, a execução começa do que ficou lá — e o que ficou lá inclui os
# intermediários do turno passado. `tentativas` valendo 1 de ontem faria a
# ampliação de hoje desistir na primeira falha; `mensagens` do turno passado
# entrariam na conversa deste; `atendimento` preenchido faria a primeira rota
# achar que o fluxo já terminou.
#
# É um efeito de segunda ordem de guardar o estado, e ele não aparece em teste de
# uma pergunta só: aparece na segunda pergunta da mesma conversa.
#
# `historico` é a única chave que **não** entra aqui — ela é justamente a que
# deve atravessar os turnos.
ESTADO_DO_TURNO = {
    "escopo": "",
    "trajetoria": [],
    "parte_do_pedido": "",
    "parte_da_regra": "",
    "resposta_do_pedido": "",
    "resposta_da_regra": "",
    "mensagens": [],
    "trechos": [],
    "fontes": [],
    "tokens_de_entrada": 0,
    "passos": 0,
    "top_k": None,
    "tentativas": 0,
    "groundedness": None,
    "atendimento": None,
}


# A triagem é a primeira decisão do grafo, e a mais barata: uma chamada curta que
# só classifica, sem contexto e sem ferramenta. Ela existe para o que não é
# assunto da loja não chegar a pagar busca vetorial, ciclo de ferramenta e
# chamada de formato.
INSTRUCAO_DA_TRIAGEM = """Você é a triagem do atendimento de uma loja de produtos congelados. Classifique a mensagem do cliente em uma de três categorias:

- "atendimento": qualquer assunto da loja — pedido, entrega, prazo, rastreamento, troca, devolução, reclamação, pagamento, nota fiscal, cupom, cadastro, assinatura, ou dúvida sobre produto, conservação, validade e preparo.
- "conversacional": a mensagem pergunta sobre a própria conversa — o que foi dito, perguntado ou respondido nos turnos anteriores. O assunto dela é o diálogo, e não o sistema da loja.
- "composta": a mensagem traz duas perguntas de naturezas diferentes ao mesmo tempo — uma sobre um pedido específico (dado que só o sistema tem) e outra sobre uma regra, política ou prazo da loja (informação que está nos documentos).
- "fora_de_escopo": saudação e conversa fiada, e temas alheios à loja (clima, esportes, notícias, piadas, receitas, conhecimento geral, pedido de ajuda com outro assunto).

Três desempates, nesta ordem:

1. Entre "atendimento" e "conversacional", escolha "atendimento". Repetir uma consulta é pedir o dado outra vez, e o dado pode ter mudado desde a resposta anterior: uma mensagem só é "conversacional" quando pergunta o que foi **dito**, e não o que está **no sistema**.
2. Entre "atendimento" e "composta", escolha "atendimento". Só é "composta" quando as duas metades existem de verdade **e** são de naturezas diferentes: duas perguntas sobre o mesmo pedido continuam sendo "atendimento", e duas perguntas sobre regras também.
3. Entre "atendimento" e "fora_de_escopo", escolha "atendimento": o fluxo normal sabe recusar o que não está na base, e barrar um cliente legítimo é o erro mais caro dos dois.

Exemplos:
- "onde está o meu pedido 81030?" → atendimento
- "onde está o meu pedido 81030?", perguntado outra vez depois de já respondido → atendimento
- "e quanto tempo demora a entrega?" → atendimento
- "meu salmão chegou descongelado" → atendimento
- "vocês entregam em Curitiba?" → atendimento
- "posso congelar de novo depois de descongelar?" → atendimento
- "e o nome do cliente?" → conversacional
- "qual foi o número do pedido que eu te passei?" → conversacional
- "o que eu te perguntei antes?" → conversacional
- "pode repetir a sua última resposta?" → conversacional
- "onde está o meu pedido 81030 e qual a política de troca?" → composta
- "o pedido 80412 já foi entregue e quantos dias eu tenho para pedir a troca?" → composta
- "onde está o meu pedido 81030 e quando ele chega?" → atendimento
- "qual a política de troca e quanto tempo ela dura?" → atendimento
- "qual a previsão do tempo pra amanhã?" → fora_de_escopo
- "oi, tudo bem?" → fora_de_escopo
- "me conta uma piada" → fora_de_escopo"""

# O que separa este node do resto do fluxo: ele responde do que já está na
# conversa e não vai a fonte nenhuma. A instrução é explícita quanto a isso
# porque o modelo tem conhecimento próprio sobre lojas em geral e completaria a
# lacuna com ele — e uma resposta inventada sobre o que "foi dito" é pior do que
# admitir que a conversa não registra aquilo.
INSTRUCAO_DA_CONVERSA = """A pergunta do cliente é sobre esta conversa: o que foi dito, perguntado ou respondido nos turnos acima.

Responda usando apenas o que está nas mensagens anteriores. Se a informação não estiver nelas, diga que não tem esse registro e peça ao cliente que informe o dado de novo. Não consulte nem cite a base de conhecimento, e nunca invente número, nome, prazo ou valor que não tenha sido dito aqui."""

# O que o cliente ouve quando a triagem barra a mensagem. É texto fixo, e não
# uma chamada ao modelo: a recusa é sempre a mesma, então gerá-la seria pagar
# uma segunda ida ao Bedrock para escrever uma frase que já está escrita. O
# node novo economiza chamadas — e economizar duas em vez de uma é a diferença
# entre gastar 1 e gastar 3 nesta pergunta.
TEXTO_FORA_DE_ESCOPO = (
    "Aqui eu consigo ajudar só com o que é da loja: pedidos, entregas, trocas e "
    "dúvidas sobre os produtos. Sobre algum desses, como posso ajudar?"
)

# A resposta quando a pergunta é sobre a conversa e conversa não há — primeiro
# turno da thread, ou memória desligada. Texto fixo pelo mesmo motivo da recusa
# acima: não há o que o modelo possa acrescentar lendo um histórico vazio, e
# mandá-lo responder assim mesmo é pagar uma chamada para ouvir uma invenção.
TEXTO_SEM_HISTORICO = (
    "Ainda não tenho nada registrado nesta conversa para consultar. Pode me "
    "dizer de novo o que você precisa?"
)


# O ramo do pedido enxerga uma ferramenta só. `FERRAMENTAS` inclui a abertura de
# solicitação de troca, que **grava**: numa pergunta composta o cliente pediu
# informação sobre a política, não a troca em si, e um ramo que abre protocolo por
# conta própria transformaria uma dúvida num registro no sistema.
FERRAMENTAS_DO_PEDIDO = [consultar_status_pedido]

# Quantas idas ao modelo o ramo do pedido pode fazer: uma para pedir a
# ferramenta, outra para escrever com o resultado na mão. O ramo responde a uma
# pergunta só, sobre um pedido só — não é o ciclo aberto de `conversar`.
MAXIMO_DE_PASSOS_DO_RAMO = 2

INSTRUCAO_DA_BIFURCACAO = """A mensagem do cliente traz duas perguntas ao mesmo tempo: uma sobre um pedido específico e outra sobre uma regra, política ou prazo da loja.

Separe as duas. Reescreva cada metade como uma pergunta completa, que se entenda sozinha, usando as palavras do cliente. Não responda nenhuma delas e não acrescente pergunta que ele não fez."""

INSTRUCAO_DO_RAMO_DO_PEDIDO = """Você atende uma loja de produtos congelados e está respondendo apenas à parte da pergunta que é sobre um pedido específico.

Consulte o pedido pela ferramenta e relate o que ela devolveu. Não fale de política, prazo de troca ou condição da loja: outra etapa cuida dessa parte. Escreva em duas ou três frases, sem saudação e sem oferta de ajuda adicional."""

INSTRUCAO_DO_RAMO_DA_REGRA = """Você atende uma loja de produtos congelados e está respondendo apenas à parte da pergunta que é sobre regra, política ou prazo.

Não fale de nenhum pedido específico nem cite número de pedido: outra etapa cuida dessa parte. Escreva em duas ou três frases, sem saudação e sem oferta de ajuda adicional."""

INSTRUCAO_DA_JUNCAO = """Abaixo estão duas respostas parciais à pergunta do cliente: uma foi consultada no sistema, sobre o pedido dele; a outra veio da base de conhecimento, sobre a regra que ele perguntou.

Escreva uma resposta única que entregue as duas, nessa ordem, ligando-as quando uma condiciona a outra — por exemplo, quando a situação do pedido muda o que a regra permite. Use apenas o que está nas duas partes: não acrescente informação, não repita a mesma frase duas vezes e não deixe nenhuma das duas perguntas sem resposta."""


class _Escopo(BaseModel):
    """A saída da triagem: um campo só, com quatro opções fechadas.

    Structured output com `Literal` em vez de texto livre porque o valor vai
    alimentar uma rota do grafo. Uma classificação que volta como frase teria de
    ser interpretada por nós, e interpretar texto do modelo para decidir caminho
    é exatamente o que a saída estruturada existe para evitar.
    """

    escopo: Literal[
        "atendimento", "conversacional", "composta", "fora_de_escopo"
    ] = Field(
        description=(
            "'atendimento' se a mensagem é assunto da loja; 'conversacional' se "
            "ela pergunta sobre o que já foi dito nesta conversa; 'composta' se "
            "ela junta uma pergunta sobre um pedido específico e outra sobre "
            "uma regra da loja; 'fora_de_escopo' para saudação, conversa fiada "
            "e temas alheios à loja."
        )
    )


class _PartesDaPergunta(BaseModel):
    """As duas metades de uma pergunta composta, separadas para os dois ramos.

    Cada ramo recebe só a sua metade, e não a pergunta inteira: a busca vetorial
    do ramo da regra levaria junto o número do pedido, que não descreve assunto
    nenhum e desloca o ranking; e o ramo do pedido levaria junto o texto da
    política, que o faria discutir a regra em vez de consultar o dado.
    """

    parte_do_pedido: str = Field(
        description=(
            "A metade que precisa do sistema: o que o cliente perguntou sobre um "
            "pedido específico, reescrito como pergunta completa e com o número "
            "do pedido."
        )
    )
    parte_da_regra: str = Field(
        description=(
            "A metade que precisa dos documentos: o que o cliente perguntou "
            "sobre regra, política, prazo ou condição da loja, reescrito como "
            "pergunta completa e sem o número do pedido."
        )
    )


# --- Auxiliares dos nodes ---------------------------------------------------


def _abrir_conversa(state: EstadoAtendimento) -> list:
    """Monta as mensagens iniciais: tom, conversa anterior, contexto e pergunta.

    A mensagem de contexto entra depois do `format_messages`, e é aí que ela
    escapa da formatação de template — um `{prazo}` escrito dentro de um
    documento seria lido como variável e a chamada quebraria. Vai antes da
    mensagem do cliente para o modelo ler a base antes de ler a pergunta.

    A ordem das quatro partes não é arbitrária, e é ela que faz a memória
    funcionar: instrução de tom, **conversa anterior**, documentos, pergunta de
    agora. O histórico vai logo depois da instrução porque é conversa, não é
    instrução nem documento — colocá-lo depois do contexto faria o modelo ler
    turnos antigos como se fossem base de conhecimento.

    Com a memória desligada, o histórico simplesmente não entra. Ele continua
    gravado no banco — quem não lê é este turno. Desligar não apaga.
    """
    mensagens = prompt.format_messages(
        instrucao_de_tom=PERFIS_DE_ATENDIMENTO[state["perfil"]]["instrucao_de_tom"],
        pergunta=state["pergunta"],
    )
    historico = state.get("historico") if state.get("memoria_ativa") else None
    if historico:
        mensagens[1:1] = historico
    trechos = state.get("trechos")
    if trechos:
        mensagens.insert(-1, _mensagem_de_contexto(trechos))
    return mensagens


def _texto_da_mensagem(mensagem) -> str:
    """O texto de uma resposta do modelo, venha ela como string ou como blocos.

    O `ChatBedrockConverse` devolve `content` como lista de blocos quando a
    resposta mistura texto e pedido de ferramenta. Os outros nodes não esbarram
    nisso porque leem a resposta pelo schema, com `with_structured_output`; os
    ramos leem o texto cru, e precisam aceitar as duas formas.
    """
    conteudo = mensagem.content
    if isinstance(conteudo, str):
        return conteudo.strip()
    partes = [
        bloco.get("text", "")
        for bloco in conteudo
        if isinstance(bloco, dict) and bloco.get("type", "text") == "text"
    ]
    return "\n".join(parte for parte in partes if parte).strip()


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
            tentativas=state.get("tentativas", 0),
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

    ⚠️ A classificação continua olhando **só a mensagem deste turno**: o
    histórico não entra na chamada. "Conversacional" é decidido pela **forma** da
    pergunta — ela pede o que foi dito —, e não por conferir se a resposta está
    mesmo lá. Quem confere isso é o node, que lê o histórico e, não achando,
    admite não ter o registro.
    """
    _registrar_memoria(state)
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
    return {"escopo": escopo, "trajetoria": ["triagem"]}


def _registrar_memoria(state: EstadoAtendimento) -> None:
    """Diz no log quanto da conversa anterior entrou neste turno.

    Fica no primeiro node de propósito: é o começo do turno, e é onde a leitura
    do estado gravado acabou de acontecer. Uma linha por turno, com o número de
    mensagens — que é o que separa "não lembrou" de "não tinha o que lembrar",
    duas coisas que produzem a mesma resposta e têm consertos diferentes.

    Existe para não ser preciso abrir o banco para responder essa pergunta.
    """
    historico = state.get("historico") or []
    if not state.get("memoria_ativa"):
        logger.info(
            "[memoria] desligada; %d mensagem(ns) gravada(s) e nao lidas",
            len(historico),
        )
        return
    logger.info("[memoria] ativa; %d mensagem(ns) entraram no turno", len(historico))


def resposta_direta(state: EstadoAtendimento) -> dict:
    """A saída curta para o que a triagem barrou: nenhuma busca, nenhum modelo.

    Devolve `Atendimento` no mesmo formato de sempre — quem consome a API trata
    uma forma só. `precisa_de_humano` fica falso de propósito: mensagem fora do
    escopo não é trabalho para uma pessoa, e enfileirá-la encheria a fila de
    ruído.
    """
    logger.info("[grafo] resposta_direta (sem busca e sem chamada de geracao)")
    return {
        "trajetoria": ["resposta_direta"],
        "atendimento": Atendimento(
            resposta=RespostaAtendimento(
                resposta=TEXTO_FORA_DE_ESCOPO,
                tipo=TipoDeAtendimento.OUTRO,
            )
        )
    }


def responder_do_historico(state: EstadoAtendimento) -> dict:
    """Responde a partir da conversa anterior, sem tocar em busca nem ferramenta.

    É o caminho que faltava. A pergunta que fala da própria conversa não tem o
    que recuperar na base — o que ela pede já foi dito — e mandá-la pelo fluxo
    normal só produz uma busca vetorial cuja consulta ("e o nome do cliente?")
    não descreve assunto nenhum.

    Sem histórico, devolve o texto fixo e **não chama o modelo**. Com histórico
    vazio a chamada não teria de onde tirar a resposta: sobraria o conhecimento
    próprio do modelo, e é exatamente dele que uma pergunta sobre a conversa
    precisa ficar longe.

    A instrução de restrição entra como mensagem de sistema imediatamente antes
    da pergunta, e não junto do tom: ela vale para este turno, depois do
    histórico, e é a última coisa que o modelo lê antes do que o cliente pediu.
    """
    historico = state.get("historico") if state.get("memoria_ativa") else None
    if not historico:
        logger.info("[grafo] responder_do_historico sem conversa gravada")
        return {
            "trajetoria": ["responder_do_historico"],
            "atendimento": Atendimento(
                resposta=RespostaAtendimento(
                    resposta=TEXTO_SEM_HISTORICO,
                    tipo=TipoDeAtendimento.OUTRO,
                )
            ),
        }

    mensagens = _abrir_conversa(state)
    mensagens.insert(-1, SystemMessage(content=INSTRUCAO_DA_CONVERSA))

    model = montar_modelo(state["modelo"], state["temperatura"])
    logger.info(
        "[grafo] responder_do_historico com %d mensagem(ns) de conversa",
        len(historico),
    )
    try:
        resposta = model.with_structured_output(RespostaAtendimento).invoke(mensagens)
    except ClientError as erro:
        return {
            "trajetoria": ["responder_do_historico"],
            **_falha_do_bedrock(state, erro),
        }

    # `fontes` fica vazio porque nenhuma busca aconteceu. Declarar fonte aqui
    # seria declarar como documento o que veio da conversa.
    return {
        "trajetoria": ["responder_do_historico"],
        "atendimento": Atendimento(resposta=resposta),
    }


def recuperar(state: EstadoAtendimento) -> dict:
    """Monta o contexto do modo escolhido e registra as fontes.

    Tem o seu próprio tratamento de erro porque falha aqui tem causa e conserto
    diferentes de falha do modelo — e porque o `langchain-postgres` embrulha erro
    de conexão numa Exception genérica. Sem isto, banco fora do ar viraria 500
    com traceback na tela.
    """
    falhas.interromper_se_armado("recuperar")
    modo = state["modo"]
    # `top_k` só está no estado quando a auto-correção passou por aqui. A
    # primeira passada não o define, e `_trechos_do_modo` cai no TOP_K padrão.
    top_k = state.get("top_k")
    try:
        trechos = _trechos_do_modo(modo, state["pergunta"], k=top_k)
    except Exception as erro:
        logger.exception("falha ao montar o contexto no modo %s", modo)
        return {
            "trajetoria": ["recuperar"],
            "atendimento": Atendimento(resposta=_falha_da_base(erro)),
        }

    fontes = _fontes(modo, trechos)
    logger.info(
        "[grafo] recuperar modo=%s top_k=%s trechos=%d",
        modo, top_k or "padrao", len(trechos),
    )
    # ⚠️ A consulta que foi ao retriever é a pergunta do turno, literal. O
    # histórico não chega até aqui: ele entra depois, em `_abrir_conversa`, já
    # como mensagem para o modelo.
    #
    # A consequência aparece num follow-up. "E quanto tempo demora?" é o texto
    # que o retriever recebe — sem o assunto que ficou no turno anterior. O
    # modelo lembra do assunto; a busca não. Nos modos sem busca isso não
    # existe, e é por isso que o registro só sai quando havia histórico e a
    # busca aconteceu.
    if state.get("memoria_ativa") and state.get("historico") and modo in MODOS_COM_BUSCA:
        logger.info(
            "[memoria] a busca usou a pergunta literal do turno: %r "
            "(o historico nao entra na consulta)",
            state["pergunta"],
        )
    return {"trechos": trechos, "fontes": fontes, "trajetoria": ["recuperar"]}


def conversar(state: EstadoAtendimento) -> dict:
    """Uma ida ao modelo com as ferramentas plugadas.

    Era o corpo do `for` de `responder()`; virou um node que roda **uma** volta.
    Quem repete é a edge: `executar_ferramentas` volta para cá. O contador
    `passos` é o mesmo teto de antes, agora guardado no estado em vez de ser o
    índice de um laço.

    Só a primeira ida tem o token medido: é a que carrega os documentos, e é o
    número que dá para comparar entre os modos.
    """
    falhas.interromper_se_armado("conversar")
    mensagens = state.get("mensagens") or _abrir_conversa(state)
    passos = state.get("passos", 0)

    model = montar_modelo(state["modelo"], state["temperatura"])
    logger.info("[grafo] conversar passo=%d", passos + 1)
    try:
        resposta_do_modelo = model.bind_tools(FERRAMENTAS).invoke(mensagens)
    except ClientError as erro:
        return {"trajetoria": ["conversar"], **_falha_do_bedrock(state, erro)}

    mensagens.append(resposta_do_modelo)
    mudou = {
        "mensagens": mensagens,
        "passos": passos + 1,
        "trajetoria": ["conversar"],
    }
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
    return {"mensagens": mensagens, "trajetoria": ["executar_ferramentas"]}


def formalizar(state: EstadoAtendimento) -> dict:
    """A última ida ao modelo: a mesma conversa, agora exigindo o formato.

    Fica num node separado de `conversar` porque `bind_tools` e
    `with_structured_output` disputam o mesmo mecanismo por baixo — pedir as
    duas coisas na mesma chamada é frágil. A separação que antes eram duas
    etapas dentro de uma função agora é visível na topologia.
    """
    falhas.interromper_se_armado("formalizar")
    model = montar_modelo(state["modelo"], state["temperatura"])
    logger.info("[grafo] formalizar")
    try:
        resposta = model.with_structured_output(RespostaAtendimento).invoke(
            state["mensagens"]
        )
    except ClientError as erro:
        return {"trajetoria": ["formalizar"], **_falha_do_bedrock(state, erro)}

    return {
        "trajetoria": ["formalizar"],
        "atendimento": Atendimento(
            resposta=resposta,
            tokens_de_entrada=state.get("tokens_de_entrada", 0),
            fontes=state.get("fontes", []),
            tentativas=state.get("tentativas", 0),
        )
    }


def _cosseno(a: list[float], b: list[float]) -> float:
    """Similaridade cosseno entre dois vetores: 1 é mesma direção, 0 é nada a ver."""
    produto = sum(x * y for x, y in zip(a, b))
    normas = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return produto / normas if normas else 0.0


def avaliar_groundedness(state: EstadoAtendimento) -> dict:
    """Mede o quanto a resposta se apoia nos trechos que a busca trouxe.

    Embute a resposta e cada trecho **com o mesmo modelo de embedding da busca**
    (o Titan do `retrieval.py`) e fica com o maior cosseno: o trecho mais
    parecido com o que foi dito. Mesmo modelo não é detalhe — vetores de modelos
    diferentes não vivem no mesmo espaço, e o cosseno entre eles não mede nada.

    É uma **medição**, não uma decisão: o node não muda a resposta nem o
    caminho. A rota seguinte continua sendo a da auto-correção, que olha o
    próprio campo do modelo. O número vai para a interface e para o LangFuse,
    onde vira métrica comparável entre execuções.

    ⚠️ Score baixo não é sinônimo de resposta errada. "Onde está o pedido 81030"
    no modo RAG responde com o dado da ferramenta, não com a base — e o cosseno
    contra os trechos da base sai baixo, corretamente: a resposta não veio dali.

    Só roda nos modos de busca. Sem trecho recuperado (sem conhecimento) não há
    contra o que medir, e no stuffing a base entra inteira: medir contra tudo não
    diz de onde a resposta saiu. Nos dois casos o valor é `None`. Falha no
    embedding também vira `None` — a medição não pode derrubar a resposta.
    """
    atendimento = state.get("atendimento")
    trechos = state.get("trechos") or []
    if (
        atendimento is None
        or not trechos
        or state.get("modo") not in MODOS_COM_BUSCA
    ):
        return {"groundedness": None, "trajetoria": ["avaliar_groundedness"]}

    try:
        embeddings = retrieval.base_vetorial().embeddings
        vetores = embeddings.embed_documents(
            [atendimento.resposta.resposta] + [conteudo for _, conteudo in trechos]
        )
    except Exception as erro:
        logger.warning("[groundedness] falha ao calcular: %s", erro)
        return {"groundedness": None, "trajetoria": ["avaliar_groundedness"]}

    score = round(max(_cosseno(vetores[0], v) for v in vetores[1:]), 4)
    logger.info("[groundedness] score=%.4f trechos=%d", score, len(trechos))
    return {"groundedness": score, "trajetoria": ["avaliar_groundedness"]}


def ampliar_busca(state: EstadoAtendimento) -> dict:
    """A auto-correção: pede a mesma coisa, olhando mais fundo na base.

    ⚠️ O que este node **não** faz é reescrever a pergunta. A pergunta do cliente
    é o que ela é: trocar as palavras dele por outras muda o que foi perguntado,
    e uma resposta certa para uma pergunta que ninguém fez é pior do que um "não
    encontrei". O que muda aqui é o **alcance da busca** — de `TOP_K` para
    `TOP_K_AMPLIADO` — porque a hipótese testada é outra: o trecho certo pode
    estar na base, só não entre os primeiros colocados.

    Zerar `mensagens` e `passos` é parte da correção, não limpeza: a segunda
    tentativa recomeça a conversa com o contexto novo. Continuar a conversa
    antiga deixaria o contexto estreito ainda dentro dela, e a resposta seria
    formada olhando os dois — sem se saber qual dos dois respondeu.

    E `atendimento` volta a `None` porque é ele que as rotas leem como "o fluxo
    já terminou". Um estado sem reducer guarda o último valor escrito: a
    resposta que acabou de falhar ficaria ali, e a segunda passada morreria na
    primeira rota depois de `recuperar`, sem chegar ao modelo. Foi o que
    aconteceu na primeira versão deste node.
    """
    tentativas = state.get("tentativas", 0) + 1
    logger.info(
        "[auto-correcao] tentativa=%d top_k=%d pergunta=%r (a consulta nao muda)",
        tentativas, TOP_K_AMPLIADO, state["pergunta"],
    )
    return {
        "trajetoria": ["ampliar_busca"],
        "tentativas": tentativas,
        "top_k": TOP_K_AMPLIADO,
        "mensagens": [],
        "passos": 0,
        "atendimento": None,
    }


def encaminhar(state: EstadoAtendimento) -> dict:
    """Sabe desistir: marca o caso para uma pessoa e encerra.

    O degrau que fecha a escada. Sem ele a auto-correção seria só otimismo — o
    valor de tentar de novo depende de existir um ponto em que o sistema para de
    tentar e admite que não sabe.

    O texto ao cliente é o que o modelo escreveu na última tentativa; o que se
    acrescenta é o `motivo`, que é o campo que a fila mostra a quem vai resolver.
    """
    atendimento = state["atendimento"]
    resposta = atendimento.resposta
    resposta.precisa_de_humano = True
    resposta.motivo = (
        f"{resposta.motivo} {MOTIVO_DA_AMPLIACAO_SEM_RESULTADO}".strip()
        if resposta.motivo
        else MOTIVO_DA_AMPLIACAO_SEM_RESULTADO
    )
    logger.info(
        "[grafo] encaminhar apos %d ampliacao(oes): %s",
        atendimento.tentativas, resposta.motivo,
    )
    return {"atendimento": atendimento, "trajetoria": ["encaminhar"]}


def bifurcar(state: EstadoAtendimento) -> dict:
    """Separa a pergunta composta nas duas metades que os ramos vão responder.

    É o node de bifurcação, e ele não decide **se** o fluxo se divide — isso já
    veio da triagem. O que ele faz é preparar a divisão: sem separar as metades,
    os dois ramos receberiam a mesma frase inteira e cada um teria de ignorar
    metade dela por conta própria.

    Roda em temperatura mínima, como a triagem: separar não é redigir, e a mesma
    pergunta tem de se dividir sempre do mesmo jeito.

    Falha aqui não interrompe nada — os dois ramos recebem a pergunta inteira e
    respondem o que conseguirem dela. Metade da resposta é melhor do que erro.
    """
    model = montar_modelo(state["modelo"], TEMPERATURA_MINIMA)
    try:
        partes = model.with_structured_output(_PartesDaPergunta).invoke(
            [
                SystemMessage(content=INSTRUCAO_DA_BIFURCACAO),
                HumanMessage(content=state["pergunta"]),
            ]
        )
        parte_do_pedido = partes.parte_do_pedido
        parte_da_regra = partes.parte_da_regra
    except Exception as erro:
        logger.warning("bifurcacao falhou, os dois ramos recebem a pergunta inteira: %s", erro)
        parte_do_pedido = parte_da_regra = state["pergunta"]

    logger.info(
        "[grafo] bifurcar pedido=%r regra=%r", parte_do_pedido, parte_da_regra
    )
    return {
        "trajetoria": ["bifurcar"],
        "parte_do_pedido": parte_do_pedido,
        "parte_da_regra": parte_da_regra,
    }


def ramo_do_pedido(state: EstadoAtendimento) -> dict:
    """Um dos dois ramos: consulta o dado do pedido e escreve a metade dele.

    Roda ao mesmo tempo que `ramo_da_regra`, e os dois não se veem: cada um lê o
    estado que `bifurcar` deixou e escreve num campo só seu. É isso que permite
    que o LangGraph os execute no mesmo passo — e é por isso que nenhum dos dois
    pode escrever no campo do outro.

    O ciclo de ferramenta daqui tem duas idas e nada mais. Ele não é o
    `conversar`, que conversa até o modelo parar de pedir: aqui a pergunta é uma
    só, sobre um pedido só.
    """
    consulta = state.get("parte_do_pedido") or state["pergunta"]
    model = montar_modelo(state["modelo"], state["temperatura"]).bind_tools(
        FERRAMENTAS_DO_PEDIDO
    )
    mensagens = [
        SystemMessage(content=INSTRUCAO_DO_RAMO_DO_PEDIDO),
        HumanMessage(content=consulta),
    ]
    logger.info("[grafo] ramo_do_pedido consulta=%r", consulta)
    try:
        for _ in range(MAXIMO_DE_PASSOS_DO_RAMO):
            resposta = model.invoke(mensagens)
            mensagens.append(resposta)
            if not resposta.tool_calls:
                break
            for chamada in resposta.tool_calls:
                mensagens.append(
                    ToolMessage(
                        content=_executar_ferramenta(chamada),
                        tool_call_id=chamada["id"],
                    )
                )
    except ClientError as erro:
        # A falha vira a resposta parcial deste ramo, e não uma exceção: o outro
        # ramo continua rodando, e a junção entrega o que conseguiu apurar em vez
        # de derrubar a pergunta inteira.
        return {
            "trajetoria": ["ramo_do_pedido"],
            "resposta_do_pedido": traduzir_erro_do_bedrock(
                erro, MODELOS[state["modelo"]]["model_id"]
            ),
        }

    return {
        "trajetoria": ["ramo_do_pedido"],
        "resposta_do_pedido": _texto_da_mensagem(mensagens[-1]),
    }


def ramo_da_regra(state: EstadoAtendimento) -> dict:
    """O outro ramo: busca a regra nos documentos e escreve a metade dela.

    Faz o trabalho de `recuperar` e o de `conversar` numa passada só, e sem
    ferramenta: a metade que chega aqui pergunta sobre política, e política está
    na base — não no sistema de pedidos.

    A consulta que vai ao retriever é a metade reescrita por `bifurcar`, sem o
    número do pedido. É o único node em que a consulta difere da pergunta do
    cliente, e a diferença não é reescrita: é recorte do que ele mesmo disse.

    É também o único dos dois ramos que escreve `fontes`, `trechos` e
    `tokens_de_entrada` — campos de sobrescrita, que dois escritores no mesmo
    passo tornariam inválidos.
    """
    modo = state["modo"]
    consulta = state.get("parte_da_regra") or state["pergunta"]
    try:
        trechos = _trechos_do_modo(modo, consulta)
    except Exception as erro:
        logger.exception("falha ao montar o contexto no modo %s", modo)
        return {
            "trajetoria": ["ramo_da_regra"],
            "resposta_da_regra": _falha_da_base(erro).resposta,
            "trechos": [],
            "fontes": [],
        }

    mensagens = [SystemMessage(content=INSTRUCAO_DO_RAMO_DA_REGRA)]
    if trechos:
        mensagens.append(_mensagem_de_contexto(trechos))
    mensagens.append(HumanMessage(content=consulta))

    model = montar_modelo(state["modelo"], state["temperatura"])
    logger.info(
        "[grafo] ramo_da_regra consulta=%r trechos=%d", consulta, len(trechos)
    )
    try:
        resposta = model.invoke(mensagens)
    except ClientError as erro:
        return {
            "trajetoria": ["ramo_da_regra"],
            "resposta_da_regra": traduzir_erro_do_bedrock(
                erro, MODELOS[state["modelo"]]["model_id"]
            ),
            "trechos": trechos,
            "fontes": _fontes(modo, trechos),
        }

    return {
        "trajetoria": ["ramo_da_regra"],
        "resposta_da_regra": _texto_da_mensagem(resposta),
        "trechos": trechos,
        "fontes": _fontes(modo, trechos),
        "tokens_de_entrada": _tokens_de_entrada(resposta),
    }


def juntar(state: EstadoAtendimento) -> dict:
    """Só roda quando os dois ramos terminaram, e compõe a resposta final.

    A espera não é programada: ela é a topologia. Um node cujas arestas de
    entrada vêm dos dois ramos só é agendado quando os dois concluíram — não há
    sinalização nem contador em lugar nenhum deste arquivo.

    A composição é uma ida ao modelo, e não uma concatenação, porque as duas
    metades podem se condicionar: um pedido cancelado muda o que a política de
    troca permite, e grudar os dois parágrafos entregaria ao cliente uma regra
    que não vale para o caso dele.

    As duas partes entram etiquetadas — qual veio do sistema e qual veio da base
    — porque elas não têm o mesmo peso: uma é dado daquele cliente, a outra é
    política da loja.
    """
    do_pedido = state.get("resposta_do_pedido") or "Sem informação sobre o pedido."
    da_regra = state.get("resposta_da_regra") or "Sem informação sobre a regra."
    partes = (
        f"PARTE CONSULTADA NO SISTEMA (sobre o pedido do cliente):\n{do_pedido}"
        f"\n\nPARTE VINDA DA BASE DE CONHECIMENTO (sobre a regra da loja):\n{da_regra}"
    )
    mensagens = [
        SystemMessage(
            content=PERFIS_DE_ATENDIMENTO[state["perfil"]]["instrucao_de_tom"]
        ),
        SystemMessage(content=INSTRUCAO_DA_JUNCAO),
        SystemMessage(content=partes),
        HumanMessage(content=state["pergunta"]),
    ]

    model = montar_modelo(state["modelo"], state["temperatura"])
    logger.info("[grafo] juntar as duas respostas parciais")
    try:
        resposta = model.with_structured_output(RespostaAtendimento).invoke(mensagens)
    except ClientError as erro:
        return {"trajetoria": ["juntar"], **_falha_do_bedrock(state, erro)}

    return {
        "trajetoria": ["juntar"],
        "atendimento": Atendimento(
            resposta=resposta,
            tokens_de_entrada=state.get("tokens_de_entrada", 0),
            fontes=state.get("fontes", []),
            tentativas=state.get("tentativas", 0),
        ),
    }


def finalizar(state: EstadoAtendimento) -> dict:
    """Grava no histórico o par pergunta/resposta deste turno.

    É o único node que escreve na conversa, e está sozinho no fim de **todos** os
    caminhos de propósito. Se a gravação morasse em quem produz a resposta, o
    ciclo de ampliação — que passa duas vezes por `formalizar` — gravaria a mesma
    pergunta duas vezes, e a resposta boa apareceria depois de uma pergunta
    repetida. Um node terminal único roda uma vez por execução, seja qual for o
    caminho: é o lugar certo para um efeito que tem de acontecer exatamente uma
    vez.

    **Só o texto entra.** As mensagens de trabalho do turno ficam fora, e não por
    economia: uma `AIMessage` com `tool_calls` sem a `ToolMessage`
    correspondente é conversa pela metade, e no turno seguinte o `bind_tools`
    recebe um pedido de ferramenta sem resultado e a chamada é recusada. O
    histórico guarda o que foi dito, não como foi produzido.

    Com a memória desligada este node não grava nada, e isso é o que torna o
    controle honesto: um turno sem memória não lê a conversa **nem entra nela**.
    Se ele gravasse, desligar seria só esconder — e ao religar apareceria um
    trecho de conversa que o atendente nunca viu.
    """
    if not state.get("memoria_ativa"):
        return {"trajetoria": ["finalizar"]}
    atendimento = state.get("atendimento")
    if atendimento is None:
        return {"trajetoria": ["finalizar"]}
    return {
        "trajetoria": ["finalizar"],
        "historico": [
            HumanMessage(content=state["pergunta"]),
            AIMessage(content=atendimento.resposta.resposta),
        ]
    }


# --- Edges condicionais -----------------------------------------------------
#
# Uma rota só escolhe o próximo node: devolve uma string e não toca no estado.
# Manter isso separado do trabalho é o que faz a decisão ser legível — e o que
# permite mudar o caminho sem mexer em quem executa.


def rota_apos_triagem(state: EstadoAtendimento) -> str:
    """Cinco saídas: barrar, ler a conversa, bifurcar, buscar, ou ir direto.

    É de propósito uma rota, e não duas condicionais em série (uma para o escopo,
    outra para o modo). O que se decide aqui é uma coisa só — por onde esta
    pergunta entra no fluxo — e ler as possibilidades lado a lado é o que torna a
    topologia legível.

    A segunda metade é a decisão que era o `if modo` dentro de
    `_trechos_do_modo`. O stuffing entra por "com_busca" junto com os dois modos
    de RAG: ele não faz busca, mas precisa do mesmo node, porque a pergunta que
    o node responde é "o que entra no contexto?".
    """
    if state["escopo"] == "fora_de_escopo":
        return "fora_de_escopo"
    if state["escopo"] == "conversacional":
        return "conversacional"
    if state["escopo"] == "composta":
        return "composta"
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


def _nao_se_sustentou(atendimento: Atendimento) -> bool:
    """Se a resposta admitiu não ter a informação. É o sensor da auto-correção.

    Quem diz que o contexto não respondeu é o próprio modelo, no campo
    `precisa_de_humano` — a instrução que acompanha o contexto pede exatamente
    isso quando a resposta não está nos documentos. Não é opinião: é o campo do
    schema que a saída estruturada obriga a preencher.

    ⚠️ `fontes` vazio entra como segundo sinal, mas ele **não** é suficiente
    sozinho num modo de busca: `similarity_search` devolve `k` trechos sempre
    que a base tem `k` chunks, então `fontes` só fica vazio aqui se a base
    estiver vazia. Trecho recuperado não é trecho que respondeu — é a mesma
    razão pela qual a interface rotula a lista como "trechos recuperados", e não
    como "fontes".
    """
    return atendimento.resposta.precisa_de_humano or not atendimento.fontes


def rota_apos_resposta(state: EstadoAtendimento) -> str:
    """A escada da auto-correção: ampliar, encaminhar, ou terminar.

    Três guardas antes de decidir tentar de novo, e todas as três importam:

    1. `auto_corrigir` — é um controle da interface, e o padrão é desligado. O
       ciclo custa uma busca e duas idas ao modelo a mais.
    2. `modo in MODOS_COM_BUSCA` — **obrigatória**. Em `sem_conhecimento` e
       `stuffing` não há ranking a ampliar: no primeiro não há base, no segundo
       ela já entrou inteira. Sem esta guarda o ciclo dispararia nesses dois
       modos toda vez que o atendente encaminhasse um caso, e ampliaria uma
       busca que não aconteceu.
    3. o sensor — a resposta precisa ter admitido que não sabe.

    Passadas as três, o teto decide: ainda há tentativa, amplia; não há mais,
    encaminha.
    """
    if not state.get("auto_corrigir"):
        return "fim"
    if state["modo"] not in MODOS_COM_BUSCA:
        return "fim"
    if not _nao_se_sustentou(state["atendimento"]):
        return "fim"
    if state.get("tentativas", 0) < MAXIMO_DE_TENTATIVAS:
        return "ampliar"
    return "encaminhar"


# --- Montagem ---------------------------------------------------------------


def montar_grafo() -> StateGraph:
    """Declara os nodes e liga as edges. Devolve a topologia **sem** compilar.

    A montagem foi separada da compilação porque as duas passaram a acontecer em
    momentos diferentes. A topologia é conhecida no import; o que a compilação
    precisa — a conexão com o banco onde o estado é gravado — só existe depois
    que a aplicação sobe. Manter a montagem pura deixa este arquivo continuar
    sendo lido e desenhado sem banco nenhum de pé.
    """
    grafo = StateGraph(EstadoAtendimento)

    grafo.add_node("triagem", triagem)
    grafo.add_node("resposta_direta", resposta_direta)
    grafo.add_node("responder_do_historico", responder_do_historico)
    grafo.add_node("bifurcar", bifurcar)
    grafo.add_node("ramo_do_pedido", ramo_do_pedido)
    grafo.add_node("ramo_da_regra", ramo_da_regra)
    grafo.add_node("juntar", juntar)
    grafo.add_node("recuperar", recuperar)
    grafo.add_node("conversar", conversar)
    grafo.add_node("executar_ferramentas", executar_ferramentas)
    grafo.add_node("formalizar", formalizar)
    grafo.add_node("avaliar_groundedness", avaliar_groundedness)
    grafo.add_node("ampliar_busca", ampliar_busca)
    grafo.add_node("encaminhar", encaminhar)
    grafo.add_node("finalizar", finalizar)

    grafo.add_edge(START, "triagem")
    grafo.add_conditional_edges(
        "triagem",
        rota_apos_triagem,
        {
            "fora_de_escopo": "resposta_direta",
            "conversacional": "responder_do_historico",
            "composta": "bifurcar",
            "com_busca": "recuperar",
            "sem_busca": "conversar",
        },
    )
    # Nenhum caminho vai direto para END: todos passam por `finalizar`, que é
    # quem grava o turno no histórico.
    grafo.add_edge("resposta_direta", "finalizar")
    grafo.add_edge("responder_do_historico", "finalizar")
    # O fan-out: duas arestas saindo do mesmo node colocam os dois ramos no mesmo
    # passo, e o LangGraph os executa em paralelo. A junção espera os dois porque
    # recebe aresta dos dois — um node só é agendado quando tudo o que entra nele
    # concluiu. A espera é a topologia; não há contador nem sinalização.
    grafo.add_edge("bifurcar", "ramo_do_pedido")
    grafo.add_edge("bifurcar", "ramo_da_regra")
    grafo.add_edge("ramo_do_pedido", "juntar")
    grafo.add_edge("ramo_da_regra", "juntar")
    grafo.add_edge("juntar", "finalizar")
    grafo.add_conditional_edges(
        "recuperar",
        rota_apos_recuperar,
        {"conversar": "conversar", "fim": "finalizar"},
    )
    # O ciclo de ferramenta: a volta `executar_ferramentas → conversar` é a
    # aresta que antes era a próxima iteração do `for`.
    grafo.add_conditional_edges(
        "conversar",
        rota_da_ferramenta,
        {
            "ferramenta": "executar_ferramentas",
            "formalizar": "formalizar",
            "fim": "finalizar",
        },
    )
    grafo.add_edge("executar_ferramentas", "conversar")
    # O ciclo de auto-correção: `ampliar_busca` volta para `recuperar`, e a
    # pergunta passa uma segunda vez pelo mesmo caminho — com a busca mais larga
    # e a conversa recomeçada.
    # A medição fica entre a resposta e a decisão de ampliar: numa segunda
    # passada ela roda de novo, e o número que sai é o da resposta final.
    grafo.add_edge("formalizar", "avaliar_groundedness")
    grafo.add_conditional_edges(
        "avaliar_groundedness",
        rota_apos_resposta,
        {"ampliar": "ampliar_busca", "encaminhar": "encaminhar", "fim": "finalizar"},
    )
    grafo.add_edge("ampliar_busca", "recuperar")
    grafo.add_edge("encaminhar", "finalizar")
    grafo.add_edge("finalizar", END)

    return grafo


def compilar_grafo(checkpointer=None):
    """Compila a topologia, com ou sem estado gravado entre execuções.

    O `checkpointer` é o parâmetro que muda o alcance do estado: com ele, cada
    passo do grafo é gravado sob a chave que vem no config (`thread_id`) e a
    execução seguinte começa de lá. Sem ele, cada execução começa de um estado
    vazio — que é exatamente o comportamento de antes, e é o caminho de quem
    importa este módulo fora da aplicação.

    `None` é um padrão de verdade, não uma conveniência: banco fora do ar
    desliga a memória e mantém o atendimento respondendo.
    """
    return montar_grafo().compile(checkpointer=checkpointer)


# O grafo que a aplicação usa. Fica num módulo e não numa constante de import
# porque compilar passou a depender de um recurso externo: a conexão com o banco
# é aberta pelo ciclo de vida da API, que roda depois do import.
_grafo_compilado = None


def definir_grafo(grafo) -> None:
    """Guarda o grafo que a aplicação vai usar. Chamado uma vez, no boot."""
    global _grafo_compilado
    _grafo_compilado = grafo


def grafo_ativo():
    """O grafo em uso.

    Compila um sem checkpointer se ninguém definiu nenhum — é o caso de quem
    importa este módulo direto, num script ou num teste, sem subir a API.
    """
    global _grafo_compilado
    if _grafo_compilado is None:
        _grafo_compilado = compilar_grafo()
    return _grafo_compilado
