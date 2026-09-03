# -*- coding: utf-8 -*-
"""O estado do grafo, gravado fora do processo.

O grafo já tinha estado — mas na memória do processo e só durante um `invoke()`.
No fim da execução ele era descartado, e a pergunta seguinte começava de um
estado vazio. Era por isso que cada pergunta era a primeira.

Um **checkpointer** muda o alcance desse estado. A cada passo do grafo ele grava
o estado inteiro sob uma chave — a `thread_id` — e no começo de cada execução lê
de volta o que estava gravado ali. Duas consequências, e as duas importam:

- Conversas diferentes não se misturam, porque cada uma tem a sua chave.
- O estado sobrevive ao processo. Reiniciar a API não apaga a conversa: o estado
  nunca esteve na API, esteve no Postgres.

O checkpointer que vem de fábrica no langgraph é o `InMemorySaver`. Ele resolve o
primeiro ponto e não o segundo — morre junto com o processo. Aqui a gravação vai
para o mesmo Postgres que já guarda os vetores: é um banco a menos para operar, e
as tabelas `checkpoint*` convivem com as `langchain_pg_*` sem se conhecerem.
"""
import logging

from psycopg import Connection
from psycopg.rows import dict_row

from app import db

logger = logging.getLogger(__name__)

# Os tipos deste projeto que entram no estado do grafo e precisam voltar da
# leitura como o mesmo tipo, e não como dicionário.
#
# O serializador do langgraph tenta primeiro o msgpack, que só aceita classes de
# módulos declarados, e cai para JSON no que sobra. A queda não é silenciosa: ela
# avisa, a cada gravação, que o tipo não está registrado. Declarar aqui resolve na
# origem — em vez de conviver com o aviso ou desligá-lo.
TIPOS_PROPRIOS_NO_ESTADO = [
    ("app.schemas", "Atendimento"),
    ("app.schemas", "RespostaAtendimento"),
    ("app.schemas", "TipoDeAtendimento"),
]


def _serializador():
    """O serializador do checkpointer, ciente dos tipos próprios do projeto.

    Fica numa função e não numa constante de módulo porque a classe vem do
    pacote do checkpointer: importá-la no topo faria este módulo exigir a
    dependência só para ser lido.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=TIPOS_PROPRIOS_NO_ESTADO)


def abrir_checkpointer():
    """Abre a conexão do checkpointer e prepara as tabelas. `(conexao, saver)`.

    Os dois parâmetros da conexão são obrigatórios, e por motivos diferentes:

    - `autocommit=True` porque o checkpointer executa cada gravação como
      instrução isolada, sem abrir transação própria. Numa conexão transacional o
      psycopg abre uma transação implícita no primeiro comando e nada é
      confirmado até um commit que ninguém dá — o estado pareceria gravado
      enquanto o processo vivesse, e não estaria no banco.
    - `row_factory=dict_row` porque o checkpointer lê as colunas pelo nome. Com a
      fábrica padrão, que devolve tupla, a leitura falha.

    `setup()` cria as tabelas se ainda não existirem. Roda no boot, e não na
    primeira pergunta, porque é a última hora em que uma falha de banco ainda
    aparece no log em vez de na frente de quem está usando.

    Falha aqui **não** derruba a API: devolve `(None, None)`, o grafo é compilado
    sem checkpointer e o sistema volta a responder sem memória entre execuções.
    Perder a memória é ruim; não subir é pior.
    """
    try:
        conexao = Connection.connect(
            db.dsn(), autocommit=True, row_factory=dict_row, connect_timeout=5
        )
    except Exception as erro:
        logger.warning("sem memoria entre execucoes, banco indisponivel: %s", erro)
        return None, None

    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        checkpointer = PostgresSaver(conexao, serde=_serializador())
        checkpointer.setup()
    except Exception as erro:
        logger.warning("sem memoria entre execucoes, checkpointer falhou: %s", erro)
        conexao.close()
        return None, None

    logger.info("[memoria] checkpointer em Postgres pronto (tabelas checkpoint*)")
    return conexao, checkpointer
