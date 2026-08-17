# -*- coding: utf-8 -*-
"""A configuração de log, compartilhada pela API e pelos scripts.

Fica num módulo só porque são três pontos de entrada com a mesma necessidade — e
porque o motivo de cada linha aqui vale ser explicado uma vez, não três.
"""
import logging

# O langchain-aws registra em INFO o ResponseMetadata inteiro de cada chamada de
# embedding. Indexar a base são dezenas de chamadas, e dezenas de blocos de
# metadados enterram as linhas que interessam: qual documento, quantos chunks,
# qual modelo. Este logger sobe para WARNING e volta a falar quando der erro.
LOGGERS_RUIDOSOS = ["langchain_aws.embeddings.bedrock"]


def configurar() -> None:
    """Liga o log da aplicação e cala o que atrapalha a leitura.

    O uvicorn configura só os loggers dele. Sem o basicConfig o logger raiz fica
    em WARNING e os nossos `logger.info` — a execução de ferramenta, as etapas da
    indexação, o ranking da busca — nunca chegam ao terminal.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
    for nome in LOGGERS_RUIDOSOS:
        logging.getLogger(nome).setLevel(logging.WARNING)
