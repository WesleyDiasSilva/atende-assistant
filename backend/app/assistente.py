# -*- coding: utf-8 -*-
"""O atendente: o ciclo de ferramentas e a resposta em formato fixo.

Responder a uma pergunta acontece em duas etapas.

1. O ciclo de ferramentas. O modelo recebe a lista do que pode chamar. Se ele
   pedir uma ferramenta, este arquivo executa a função, devolve o resultado e
   pergunta de novo — até ele não pedir mais nada. O modelo decide o que
   chamar; quem executa é este código.

2. O formato. Com a conversa completa em mãos, uma última chamada exige que a
   resposta venha no formato do schema, e ela chega como objeto validado em vez
   de texto solto.

As duas etapas são separadas porque `bind_tools` e `with_structured_output`
usam o mesmo mecanismo por baixo: pedir as duas coisas na mesma chamada é
frágil.

Antes das duas, uma decisão: o **modo de conhecimento** define quanto da base de
documentos entra na conversa. É a única coisa na interface que muda o que o
atendente sabe — modelo, perfil e temperatura mudam como ele responde.

A **ordem** dessas etapas saiu deste arquivo: ela agora é declarada em
`grafo.py`, como nodes e edges. O que ficou aqui é o trabalho — montar o
contexto, empacotar o contexto numa mensagem, executar a ferramenta que o modelo
pediu, contar token, traduzir erro do Bedrock. Os nodes do grafo chamam estas
funções; nenhuma delas sabe que existe um grafo.
"""
import logging
from functools import lru_cache
from uuid import uuid4

from botocore.exceptions import ClientError, NoCredentialsError
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from app import documentos, retrieval, retrieval_gerenciado
from app.config import (
    MODELOS,
    MODO_PADRAO,
    PERFIS_DE_ATENDIMENTO,
    REGIAO_AWS,
    TEMPERATURA_PADRAO,
)
from app.schemas import Atendimento, RespostaAtendimento, TipoDeAtendimento
from app.tools import FERRAMENTAS, FERRAMENTAS_POR_NOME

# A mensagem de sistema define quem o atendente é. Tem uma parte fixa (a persona)
# e uma variável ({instrucao_de_tom}), preenchida com o perfil que o usuário
# escolheu — templates aceitam variáveis em qualquer papel de mensagem, não só no
# papel do cliente.
MENSAGEM_SISTEMA = """Você é o atendente de pedidos de uma loja de produtos congelados.

{instrucao_de_tom}"""

MENSAGEM_DO_CLIENTE = "{pergunta}"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", MENSAGEM_SISTEMA),
        ("human", MENSAGEM_DO_CLIENTE),
    ]
)

# Abre a mensagem que carrega os documentos. Fica separada de MENSAGEM_SISTEMA
# porque ela só existe quando há contexto: sem documento nenhum, não há o que
# instruir. A regra de não inventar chega junto com a base, não antes dela — e é
# por isso que o modo sem conhecimento continua respondendo de cabeça, com o que
# o modelo aprendeu sobre lojas em geral em vez do que esta loja escreveu.
#
# A recusa vem daqui, e não de um limiar de score na busca, porque o score não
# separa pergunta coberta de pergunta não coberta: medido nesta base, as duas
# faixas se sobrepõem. Quem sabe se o contexto responde é quem lê o contexto.
#
# O parágrafo sobre ferramenta não é detalhe: sem ele, "responda apenas com o
# contexto" faz o modelo desprezar o resultado de uma consulta de pedido, e a
# pergunta "onde está o pedido 81030" para de ser respondida.
CABECALHO_DO_CONTEXTO = """Os documentos abaixo são a base de conhecimento da empresa.

Sobre políticas, prazos, pagamento e produtos, responda usando apenas o que está nesses documentos. Se a resposta não estiver neles, diga que não encontrou essa informação na base e ofereça encaminhar o caso para um atendente. Não complete com conhecimento geral e nunca invente número, prazo, valor ou condição.

O resultado de uma ferramenta não é contexto: é dado do sistema sobre o pedido daquele cliente, e vale como informação confiável.

CONTEXTO:
"""

# Teto de idas ao modelo numa mesma pergunta. O ciclo de ferramenta é um laço,
# e laço sem limite é onde um agente trava: o modelo pode continuar pedindo
# ferramenta indefinidamente. Quatro passos cobrem os casos reais deste
# atendimento com folga.
MAXIMO_DE_PASSOS = 4

# Os modos em que um retriever escolheu os trechos — e que por isso têm fonte a
# declarar. Os dois só diferem em quem executa a busca.
MODOS_COM_BUSCA = ("rag", "rag_gerenciado")

logger = logging.getLogger(__name__)


@lru_cache(maxsize=16)
def montar_modelo(modelo: str, temperatura: float) -> ChatBedrockConverse:
    """Cria o chat model com o que a interface escolheu.

    Cacheado por (modelo, temperatura) porque agora ele é pedido mais de uma vez
    na mesma pergunta: cada node que fala com o Bedrock chama esta função.
    Construir um `ChatBedrockConverse` abre um cliente boto3 — fazer isso três
    vezes por pergunta pagaria três vezes a mesma montagem. O objeto não guarda
    estado de conversa (`bind_tools` e `with_structured_output` devolvem novos
    objetos), então reaproveitá-lo entre requisições é seguro.

    O mesmo padrão de `retrieval.base_vetorial()`, e pelo mesmo motivo.
    """
    return ChatBedrockConverse(
        model_id=MODELOS[modelo]["model_id"],
        region_name=REGIAO_AWS,
        temperature=temperatura,
        max_tokens=1024,
    )


def _trechos_do_modo(
    modo: str, pergunta: str, k: int | None = None
) -> list[tuple[str, str]]:
    """O que vai no contexto, conforme o modo escolhido: [(arquivo, conteudo)].

    No modo sem conhecimento a lista é vazia e nada muda em relação ao
    comportamento anterior. No stuffing, a base inteira entra a cada pergunta —
    é o caminho mais curto para o atendente acertar, e o mais caro. Nos dois modos
    de busca, quem decide são os trechos mais próximos da pergunta.

    Os dois últimos ramos são de propósito quase idênticos: trocam o retriever e
    mais nada. Tudo o que vem depois desta função — a mensagem de contexto, a
    instrução de groundedness, o ciclo de ferramenta — não sabe qual dos dois
    respondeu. É o que faz a comparação medir o retriever, e não o prompt.
    """
    if modo == "stuffing":
        return [(arquivo, conteudo) for arquivo, _, conteudo in documentos.carregar()]
    # `k` chega preenchido quando a auto-correção pediu uma busca mais larga. O
    # padrão continua sendo o `TOP_K` da configuração, e o stuffing ignora o
    # parâmetro: lá não há ranking a ampliar, a base entra inteira de qualquer
    # jeito.
    k = k or retrieval.TOP_K
    if modo == "rag":
        return _do_retriever(retrieval.buscar(pergunta, k=k))
    if modo == "rag_gerenciado":
        return _do_retriever(retrieval_gerenciado.buscar(pergunta, k=k))
    return []


def _do_retriever(trechos: list) -> list[tuple[str, str]]:
    """Converte os `Document` de qualquer um dos dois retrievers no par do contexto."""
    return [(trecho.metadata["arquivo"], trecho.page_content) for trecho in trechos]


def _mensagem_de_contexto(trechos: list[tuple[str, str]]) -> SystemMessage:
    """Empacota os trechos numa mensagem de sistema a mais.

    O contexto **não** entra como variável do ChatPromptTemplate, e isso não é
    estilo: um `{prazo}` escrito dentro de um documento seria lido como variável
    de template e a chamada quebraria com KeyError. Documento é dado de entrada,
    e dado de entrada não passa por formatação de template.

    O nome do arquivo acompanha cada trecho para o modelo poder dizer, no texto,
    de onde tirou o que disse. O campo `fontes` da resposta é outra coisa: quem o
    preenche é o nosso código, a partir do que a busca recuperou.
    """
    corpo = "\n\n".join(f"[{arquivo}]\n{conteudo}" for arquivo, conteudo in trechos)
    return SystemMessage(content=CABECALHO_DO_CONTEXTO + corpo)


def _fontes(modo: str, trechos: list[tuple[str, str]]) -> list[str]:
    """Os documentos que a busca recuperou, sem repetir e na ordem do ranking.

    Dois trechos podem vir do mesmo arquivo, e o cliente não precisa ver o nome
    duas vezes. Só os modos de busca têm fonte a declarar: no stuffing entrou
    tudo, e listar a base inteira não diria de onde a resposta saiu.

    Quem preenche é o nosso código nos dois casos — inclusive no gerenciado, onde
    a busca é de outro. Fonte que o código conhece não pode ser inventada.
    """
    if modo not in MODOS_COM_BUSCA:
        return []
    return list(dict.fromkeys(arquivo for arquivo, _ in trechos))


def _falha_da_base(erro: Exception) -> RespostaAtendimento:
    """A resposta quando a base de conhecimento não pôde ser consultada.

    Sai no mesmo formato de sempre, e não como erro HTTP: quem consome a API trata
    uma forma só. Vai para a fila humana porque o cliente continua sem resposta —
    a pergunta dele não foi respondida, ela foi engolida por uma falha de infra.

    O texto ao cliente não menciona infraestrutura; o diagnóstico vai no `motivo`,
    que é o campo que a fila mostra a quem vai resolver. São públicos diferentes,
    e a mesma falha precisa dizer coisas diferentes para cada um.
    """
    if isinstance(erro, NoCredentialsError):
        # Vale distinguir esta das outras: a chave de API do Bedrock cobre o
        # modelo, mas não a consulta ao Knowledge Base, que exige credencial IAM.
        # É a confusão mais provável de quem só configurou o bearer token.
        detalhe = (
            "Sem credencial IAM para consultar o Knowledge Base. A chave de API "
            "do Bedrock (AWS_BEARER_TOKEN_BEDROCK) autentica o modelo, mas não "
            "esta consulta."
        )
    else:
        detalhe = f"A base de conhecimento não respondeu à consulta: {erro}"

    return RespostaAtendimento(
        resposta=(
            "Não consegui consultar a base de conhecimento agora, então prefiro "
            "não responder de memória. Já encaminhei o caso para um atendente."
        ),
        tipo=TipoDeAtendimento.OUTRO,
        precisa_de_humano=True,
        motivo=detalhe,
    )


def _tokens_de_entrada(resposta_do_modelo) -> int:
    """Quantos tokens o Bedrock contou na entrada desta chamada.

    Medimos a **primeira** ida ao modelo, porque é a que carrega os documentos, e
    é o número que dá para comparar entre os modos: mesma pergunta, mesmas
    ferramentas, só o contexto muda.

    Não é o custo total da pergunta. Uma pergunta faz mais de uma chamada — cada
    volta do ciclo de ferramenta e a chamada final que exige o formato reenviam a
    conversa inteira, contexto incluído. O total é maior; o que está aqui é o
    tamanho do prompt, e é sobre ele que o modo de conhecimento manda.
    """
    return (resposta_do_modelo.usage_metadata or {}).get("input_tokens", 0)


def _executar_ferramenta(chamada: dict) -> str:
    """Roda a função que o modelo pediu e devolve o resultado como texto.

    Quem executa é este código, não o modelo: ele apenas informou o nome e os
    argumentos. Um nome fora do conjunto conhecido vira mensagem de erro em vez
    de exceção — o modelo lê a mensagem e se recupera.
    """
    ferramenta = FERRAMENTAS_POR_NOME.get(chamada["name"])
    if ferramenta is None:
        return f"Ferramenta {chamada['name']} nao existe."

    # Registra a execução: o pedido do modelo e a execução são coisas
    # diferentes, e no log dá para ver as duas acontecendo em sequência.
    logger.info("executando %s com %s", chamada["name"], chamada["args"])

    try:
        return str(ferramenta.invoke(chamada["args"]))
    except Exception as erro:  # a falha da ferramenta não pode derrubar a resposta
        return f"A ferramenta {chamada['name']} falhou: {erro}"


def config_da_conversa(conversa_id: str) -> dict:
    """A chave sob a qual o estado é gravado e lido: `thread_id`.

    É o que separa uma conversa da outra. Duas pessoas conversando ao mesmo
    tempo com o mesmo atendente compartilham o processo, o grafo e o banco, e não
    compartilham nada do que foi dito — porque cada uma entra com uma chave
    diferente.

    Um grafo compilado com checkpointer **recusa** a execução sem `thread_id`:
    ele não tem onde gravar. Quando quem chama não informa nada, o turno ganha
    uma chave nova e descartável: ele roda, grava, e ninguém volta a ler aquilo.
    Vale para chamada solta à API, que não é conversa.
    """
    return {"configurable": {"thread_id": conversa_id or str(uuid4())}}


def responder(
    pergunta: str,
    perfil: str,
    modelo: str,
    temperatura: float = TEMPERATURA_PADRAO,
    modo: str = MODO_PADRAO,
    auto_corrigir: bool = False,
    conversa_id: str = "",
) -> Atendimento:
    """Responde ao cliente executando o grafo do atendimento.

    O que esta função faz é a fronteira: monta o estado do turno, invoca o grafo
    na conversa certa e devolve o que ele deixou no campo de saída. As decisões e
    o ciclo estão em `grafo.py`, onde dá para ler a topologia inteira de uma vez.

    O `ESTADO_DO_TURNO` no meio do estado inicial é obrigatório, e o motivo não é
    óbvio: com o estado gravado, a execução **não** começa vazia — começa do que
    ficou do turno anterior. Zerar os intermediários é o que impede que a
    contagem de ampliações e as mensagens de trabalho de ontem entrem no turno de
    hoje. O que vem escrito depois do `**` vence: a pergunta e os controles
    sobrescrevem o zero.

    O import é local para não fechar um ciclo: `grafo` importa as funções de
    trabalho deste módulo.
    """
    from app.grafo import ESTADO_DO_TURNO, grafo_ativo

    estado_final = grafo_ativo().invoke(
        {
            **ESTADO_DO_TURNO,
            "pergunta": pergunta,
            "perfil": perfil,
            "modelo": modelo,
            "temperatura": temperatura,
            "modo": modo,
            "auto_corrigir": auto_corrigir,
        },
        config=config_da_conversa(conversa_id),
    )
    return estado_final["atendimento"]


def traduzir_erro_do_bedrock(erro: ClientError, model_id: str) -> str:
    """Transforma o código de erro do Bedrock numa mensagem que o usuário entende."""
    detalhes = erro.response.get("Error", {})
    codigo = detalhes.get("Code", "")
    mensagem_do_servico = detalhes.get("Message", str(erro))

    if codigo in ("UnrecognizedClientException", "InvalidSignatureException"):
        return (
            "Não consegui autenticar no Amazon Bedrock: a chave de API é "
            "inválida ou expirou."
        )
    if codigo in ("ResourceNotFoundException", "ValidationException"):
        return (
            f"O identificador do modelo não foi aceito: {model_id}. "
            "Ele exige um prefixo de inference profile, como 'global.' ou 'us.', "
            f"compatível com a região {REGIAO_AWS}."
        )
    if codigo in ("AccessDeniedException", "AccessDenied"):
        return (
            "O Amazon Bedrock negou o acesso. Normalmente é a chave de API "
            "inválida ou expirada; menos comum é o modelo não estar habilitado "
            f"em 'Model access' na região {REGIAO_AWS}."
        )
    if codigo == "ThrottlingException":
        return (
            "O Amazon Bedrock está limitando as chamadas agora. "
            "Tente de novo em alguns segundos."
        )

    return f"Erro do Amazon Bedrock ({codigo or 'sem código'}): {mensagem_do_servico}"
