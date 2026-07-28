"""Chamada direta a um modelo de linguagem via Amazon Bedrock."""

import os
import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

PERGUNTA = "Explique em uma frase o que é um assistente de atendimento."


def main() -> int:
    load_dotenv()

    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print(
            "Erro: a variável de ambiente AWS_BEARER_TOKEN_BEDROCK não está "
            "definida.\n"
            "Gere uma chave de API no console do Amazon Bedrock e defina a "
            "variável (por exemplo, no arquivo .env).",
            file=sys.stderr,
        )
        return 1

    region = os.environ.get("AWS_REGION", "us-east-1")
    model_id = os.environ.get("MODEL_ID")

    if not model_id:
        print(
            "Erro: a variável de ambiente MODEL_ID não está definida.\n"
            "Informe o identificador do modelo ou do inference profile.",
            file=sys.stderr,
        )
        return 1

    client = boto3.client("bedrock-runtime", region_name=region)

    try:
        resposta = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": PERGUNTA}]}],
        )
    except ClientError as erro:
        return _reportar_erro_do_servico(erro, model_id, region)
    except BotoCoreError as erro:
        print(f"Erro ao comunicar com o Amazon Bedrock: {erro}", file=sys.stderr)
        return 1

    texto = resposta["output"]["message"]["content"][0]["text"]
    print(texto)
    return 0


def _reportar_erro_do_servico(erro: ClientError, model_id: str, region: str) -> int:
    codigo = erro.response.get("Error", {}).get("Code", "")
    detalhe = erro.response.get("Error", {}).get("Message", str(erro))

    if codigo in ("UnrecognizedClientException", "InvalidSignatureException"):
        print(
            "Erro de autenticação: a chave de API do Bedrock é inválida ou "
            "expirou.\n"
            "Confira o valor de AWS_BEARER_TOKEN_BEDROCK (sem espaços ou "
            "quebras de linha) ou gere uma nova chave no console.",
            file=sys.stderr,
        )
    elif codigo in ("ResourceNotFoundException", "ValidationException"):
        print(
            f"Modelo não encontrado ou identificador inválido: {model_id}\n"
            "Confira o inference profile ID (ele costuma exigir um prefixo "
            "como 'global.', 'us.' ou 'eu.') e a região configurada "
            f"(AWS_REGION={region}).\n"
            f"Detalhe do serviço: {detalhe}",
            file=sys.stderr,
        )
    elif codigo in ("AccessDeniedException", "AccessDenied"):
        print(
            "Acesso negado pelo Amazon Bedrock.\n"
            "Causa provável: a chave de API é inválida ou expirou. Confira o "
            "valor de AWS_BEARER_TOKEN_BEDROCK (sem espaços ou quebras de "
            "linha) ou gere uma nova chave no console.\n"
            "Causa menos comum: a chave não tem permissão para a operação "
            f"bedrock:InvokeModel na região {region}, ou o modelo não está "
            "habilitado em 'Model access'.",
            file=sys.stderr,
        )
    else:
        print(
            f"Erro do Amazon Bedrock ({codigo or 'sem código'}): {detalhe}",
            file=sys.stderr,
        )

    return 1


if __name__ == "__main__":
    sys.exit(main())
