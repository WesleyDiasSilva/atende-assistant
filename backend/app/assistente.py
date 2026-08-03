# -*- coding: utf-8 -*-
"""A chain do atendente: prompt | model | parser.

Três peças ligadas pelo operador pipe. O prompt monta as mensagens, o model
responde, o parser devolve texto puro.
"""
from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import (
    MODELOS,
    PERFIS_DE_ATENDIMENTO,
    REGIAO_AWS,
    TEMPERATURA_PADRAO,
)

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

# Converte o AIMessage devolvido pelo modelo em string. É o que faz a chain
# entregar texto pronto para a API, em vez de um objeto do LangChain.
parser = StrOutputParser()


def montar_chain(modelo: str, temperatura: float):
    """Liga as três peças com o pipe.

    Montada a cada pergunta porque modelo e temperatura vêm da interface e mudam
    entre chamadas. Compor uma chain é barato: são três objetos encadeados.
    """
    model = ChatBedrockConverse(
        model_id=MODELOS[modelo]["model_id"],
        region_name=REGIAO_AWS,
        temperature=temperatura,
        max_tokens=1024,
    )
    return prompt | model | parser


def responder(
    pergunta: str,
    perfil: str,
    modelo: str,
    temperatura: float = TEMPERATURA_PADRAO,
) -> str:
    """Executa a chain e devolve o texto da resposta."""
    chain = montar_chain(modelo, temperatura)
    try:
        return chain.invoke(
            {
                "instrucao_de_tom": PERFIS_DE_ATENDIMENTO[perfil]["instrucao_de_tom"],
                "pergunta": pergunta,
            }
        )
    except ClientError as erro:
        # O LangChain não embrulha os erros do Bedrock: eles continuam sendo do
        # botocore, e ClientError é a classe-mãe de todos. Por isso um único
        # except cobre falha de credencial, modelo inválido e acesso negado.
        return traduzir_erro_do_bedrock(erro, MODELOS[modelo]["model_id"])


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
