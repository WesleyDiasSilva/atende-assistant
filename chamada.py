"""Chamada direta a um modelo de linguagem via Amazon Bedrock."""

import os
import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

PERGUNTA = "Explique em uma frase o que é um assistente de atendimento."


def main() -> int:
    load_dotenv()

    # A chamada ao modelo (client bedrock-runtime + operacao converse) sera
    # implementada aqui.
    print("A chamada ao modelo ainda não foi implementada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
