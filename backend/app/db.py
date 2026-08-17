# -*- coding: utf-8 -*-
"""A conexão com o Postgres, montada das variáveis de ambiente.

O projeto ganha um banco por um motivo só: guardar vetores. Os `.jsonl` do
`dados.py` continuam onde estão — o formato de uma linha por registro se lê a
olho nu, e trocá-lo por tabela não resolveria nenhum problema que exista hoje.

O que um arquivo não sabe fazer é responder "quais destes trechos estão mais perto
desta pergunta". Isso exige comparar vetores, e é o que a extensão pgvector
acrescenta ao Postgres.
"""
import logging
import os

import psycopg

logger = logging.getLogger(__name__)


def _variaveis() -> dict:
    """Os cinco valores da conexão, com o padrão de dentro do compose.

    `DB_HOST=db` é o nome do serviço na rede do compose. Quem roda o uvicorn na
    máquina aponta para `localhost` e para a porta publicada no host.
    """
    return {
        "host": os.getenv("DB_HOST", "db"),
        "port": os.getenv("DB_PORT", "5432"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
        "dbname": os.getenv("DB_NAME", "atende_assistant"),
    }


def dsn() -> str:
    """Conexão no formato do psycopg — usada para SQL direto."""
    return " ".join(f"{chave}={valor}" for chave, valor in _variaveis().items())


def url_sqlalchemy() -> str:
    """A mesma conexão no formato que o langchain-postgres espera.

    O `+psycopg` no esquema não é decoração: sem ele o SQLAlchemy tenta o driver
    psycopg2, que não está instalado.
    """
    valores = _variaveis()
    return (
        f"postgresql+psycopg://{valores['user']}:{valores['password']}"
        f"@{valores['host']}:{valores['port']}/{valores['dbname']}"
    )


def garantir_extensao_vector() -> bool:
    """Cria a extensão pgvector se ela ainda não existir.

    O `db/init.sql` já faz isso, mas só na primeira inicialização do volume:
    quem já tinha o volume criado antes nunca vê aquele script rodar. Esta função
    cobre esse caso, e é idempotente.

    Uma falha aqui é registrada e não derruba o boot — a API tem rotas que não
    dependem do banco, e vale mais deixá-las de pé com um aviso no log do que
    recusar a subir.
    """
    try:
        with psycopg.connect(dsn(), connect_timeout=5) as conexao:
            conexao.execute("CREATE EXTENSION IF NOT EXISTS vector")
        logger.info("banco conectado, extensao pgvector garantida")
        return True
    except Exception as erro:
        logger.warning("banco indisponivel no boot: %s", erro)
        return False
