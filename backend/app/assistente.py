# -*- coding: utf-8 -*-
"""A chain do atendente: prompt | model estruturado.

O prompt monta as mensagens e o model responde preenchendo um formato fixo.
O parser de texto saiu: quem garante o formato agora é o schema, e a saída já
chega como objeto validado em vez de string.
"""
import logging

from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from app.config import (
    MODELOS,
    PERFIS_DE_ATENDIMENTO,
    REGIAO_AWS,
    TEMPERATURA_PADRAO,
)
from app.schemas import RespostaAtendimento, TipoDeAtendimento
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
) -> RespostaAtendimento:
    """Responde ao cliente, usando ferramentas quando precisa de dado externo.

    São duas etapas, e a separação é proposital: `bind_tools` e
    `with_structured_output` disputam o mesmo mecanismo por baixo, então
    primeiro roda o ciclo de ferramentas até o modelo parar de pedir, e só
    depois se exige o formato final da resposta.
    """
    model = montar_modelo(modelo, temperatura)
    model_com_ferramentas = model.bind_tools(FERRAMENTAS)

    mensagens = prompt.format_messages(
        instrucao_de_tom=PERFIS_DE_ATENDIMENTO[perfil]["instrucao_de_tom"],
        pergunta=pergunta,
    )

    try:
        for _ in range(MAXIMO_DE_PASSOS):
            resposta_do_modelo = model_com_ferramentas.invoke(mensagens)
            mensagens.append(resposta_do_modelo)

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
        return model.with_structured_output(RespostaAtendimento).invoke(mensagens)
    except ClientError as erro:
        # O LangChain não embrulha os erros do Bedrock: eles continuam sendo do
        # botocore, e ClientError é a classe-mãe de todos. Por isso um único
        # except cobre falha de credencial, modelo inválido e acesso negado.
        #
        # A falha também sai no formato do schema: quem consome a API trata uma
        # forma só, com erro ou sem erro.
        return RespostaAtendimento(
            resposta=traduzir_erro_do_bedrock(erro, MODELOS[modelo]["model_id"]),
            tipo=TipoDeAtendimento.OUTRO,
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
