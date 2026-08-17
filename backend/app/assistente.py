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
"""
import logging

from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from app import documentos, retrieval
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

logger = logging.getLogger(__name__)


def montar_modelo(modelo: str, temperatura: float) -> ChatBedrockConverse:
    """Cria o chat model com o que a interface escolheu."""
    return ChatBedrockConverse(
        model_id=MODELOS[modelo]["model_id"],
        region_name=REGIAO_AWS,
        temperature=temperatura,
        max_tokens=1024,
    )


def _trechos_do_modo(modo: str, pergunta: str) -> list[tuple[str, str]]:
    """O que vai no contexto, conforme o modo escolhido: [(arquivo, conteudo)].

    No modo sem conhecimento a lista é vazia e nada muda em relação ao
    comportamento anterior. No stuffing, a base inteira entra a cada pergunta —
    é o caminho mais curto para o atendente acertar, e o mais caro. No RAG, a
    busca decide: entram só os trechos mais próximos da pergunta.
    """
    if modo == "stuffing":
        return [(arquivo, conteudo) for arquivo, _, conteudo in documentos.carregar()]
    if modo == "rag":
        return [
            (trecho.metadata["arquivo"], trecho.page_content)
            for trecho in retrieval.buscar(pergunta)
        ]
    return []


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
    duas vezes. Só o modo RAG tem fonte a declarar: no stuffing entrou tudo, e
    listar a base inteira não diria de onde a resposta saiu.
    """
    if modo != "rag":
        return []
    return list(dict.fromkeys(arquivo for arquivo, _ in trechos))


def _falha_da_base() -> RespostaAtendimento:
    """A resposta quando a base de conhecimento não pôde ser consultada.

    Sai no mesmo formato de sempre, e não como erro HTTP: quem consome a API trata
    uma forma só. Vai para a fila humana porque o cliente continua sem resposta —
    a pergunta dele não foi respondida, ela foi engolida por uma falha de infra.
    """
    return RespostaAtendimento(
        resposta=(
            "Não consegui consultar a base de conhecimento agora, então prefiro "
            "não responder de memória. Já encaminhei o caso para um atendente."
        ),
        tipo=TipoDeAtendimento.OUTRO,
        precisa_de_humano=True,
        motivo="A base de conhecimento não respondeu à consulta.",
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


def responder(
    pergunta: str,
    perfil: str,
    modelo: str,
    temperatura: float = TEMPERATURA_PADRAO,
    modo: str = MODO_PADRAO,
) -> Atendimento:
    """Responde ao cliente, usando ferramentas quando precisa de dado externo.

    São duas etapas, e a separação é proposital: `bind_tools` e
    `with_structured_output` disputam o mesmo mecanismo por baixo, então
    primeiro roda o ciclo de ferramentas até o modelo parar de pedir, e só
    depois se exige o formato final da resposta.

    O `modo` decide o que o atendente sabe. Ele é ortogonal às ferramentas: o
    contexto responde sobre política e produto, a ferramenta responde sobre o
    pedido daquele cliente. As duas fontes convivem na mesma conversa.
    """
    model = montar_modelo(modelo, temperatura)
    model_com_ferramentas = model.bind_tools(FERRAMENTAS)

    mensagens = prompt.format_messages(
        instrucao_de_tom=PERFIS_DE_ATENDIMENTO[perfil]["instrucao_de_tom"],
        pergunta=pergunta,
    )

    # Montar o contexto sai da máquina: no modo de busca são uma consulta ao
    # Postgres e uma chamada de embedding ao Bedrock. Tem o seu próprio try porque
    # falha aqui tem causa e conserto diferentes de falha do modelo — e porque o
    # langchain-postgres embrulha erro de conexão numa Exception genérica, que o
    # except ClientError de baixo não pegaria. Sem isto, banco fora do ar vira 500
    # com traceback na tela.
    try:
        trechos = _trechos_do_modo(modo, pergunta)
    except Exception:
        logger.exception("falha ao montar o contexto no modo %s", modo)
        return Atendimento(resposta=_falha_da_base())

    # A mensagem de contexto entra depois do format_messages, e é aí que ela escapa
    # da formatação de template. Vai antes da mensagem do cliente para o modelo ler
    # a base antes de ler a pergunta.
    if trechos:
        mensagens.insert(-1, _mensagem_de_contexto(trechos))

    fontes = _fontes(modo, trechos)
    tokens_de_entrada = 0

    try:
        for passo in range(MAXIMO_DE_PASSOS):
            resposta_do_modelo = model_com_ferramentas.invoke(mensagens)
            mensagens.append(resposta_do_modelo)

            if passo == 0:
                tokens_de_entrada = _tokens_de_entrada(resposta_do_modelo)

            # Sem pedido de ferramenta, o modelo já tem o que precisa.
            if not resposta_do_modelo.tool_calls:
                break

            # A intenção chega em tool_calls, com nome, argumentos e um id. O
            # resultado volta numa ToolMessage amarrada a esse id, senão o
            # modelo não sabe a qual pedido a resposta corresponde.
            for chamada in resposta_do_modelo.tool_calls:
                mensagens.append(
                    ToolMessage(
                        content=_executar_ferramenta(chamada),
                        tool_call_id=chamada["id"],
                    )
                )

        # Etapa final: a mesma conversa, agora exigindo o formato de saída.
        resposta = model.with_structured_output(RespostaAtendimento).invoke(mensagens)
        return Atendimento(
            resposta=resposta, tokens_de_entrada=tokens_de_entrada, fontes=fontes
        )
    except ClientError as erro:
        # O LangChain não embrulha os erros do Bedrock: eles continuam sendo do
        # botocore, e ClientError é a classe-mãe de todos. Por isso um único
        # except cobre falha de credencial, modelo inválido e acesso negado.
        #
        # A falha também sai no formato do schema: quem consome a API trata uma
        # forma só, com erro ou sem erro.
        return Atendimento(
            resposta=RespostaAtendimento(
                resposta=traduzir_erro_do_bedrock(erro, MODELOS[modelo]["model_id"]),
                tipo=TipoDeAtendimento.OUTRO,
            ),
            # O que já foi medido antes da falha continua valendo: se a busca
            # chegou a rodar, ela já custou, e esconder isso mentiria sobre o
            # gasto da pergunta.
            tokens_de_entrada=tokens_de_entrada,
            fontes=fontes,
        )


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
