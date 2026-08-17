# -*- coding: utf-8 -*-
"""Indexação e busca: transformar documento em vetor, e pergunta em trechos.

Três decisões estão neste arquivo, e cada uma tem um número que dá para mexer:
qual modelo transforma texto em vetor, de que tamanho são os pedaços, e quantos
pedaços a busca devolve.

O caminho da indexação acontece uma vez por documento: parte o texto em chunks,
manda cada chunk para o modelo de embedding e grava o vetor no Postgres. O
caminho da busca acontece a cada pergunta: transforma a pergunta em vetor pelo
**mesmo** modelo e pede ao banco os vetores mais próximos. Modelos diferentes nos
dois lados projetam em espaços diferentes, e distância entre espaços diferentes
não mede semelhança nenhuma.
"""
import logging
import os

from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

import psycopg

from app import db
from app.config import REGIAO_AWS

logger = logging.getLogger(__name__)

# O Titan v2 devolve vetores de 1024 dimensões. Contra esta base, ele deu o mesmo
# primeiro colocado que o Cohere multilingual em 10 de 10 perguntas de teste —
# não houve ganho de qualidade que justificasse um modelo cinco vezes mais caro.
# Trocar o modelo aqui invalida os vetores já gravados: reindexe a base.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")

# Tamanho do pedaço, em caracteres. É uma troca: pedaço grande preserva o
# contexto em volta da frase mas dilui a relevância, porque a similaridade passa
# a ser medida contra texto que não tem nada a ver com a pergunta; pedaço pequeno
# é preciso mas corta a ideia no meio. Os documentos daqui são curtos, e 800
# mantém uma seção inteira por pedaço.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))

# Sobreposição entre pedaços vizinhos, para uma frase cortada na fronteira ainda
# aparecer inteira em um dos dois.
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Quantos trechos a busca devolve. Também é uma troca: poucos trechos deixam a
# passagem certa de fora, muitos gastam contexto com ruído. Medido nesta base,
# quatro perguntas de dez erram com k=1 e acertam com k=4 — o trecho certo estava
# lá, só não em primeiro lugar.
TOP_K = int(os.getenv("TOP_K", "4"))

# Nome lógico do conjunto de vetores. Agrupa os embeddings de um mesmo domínio e
# os isola de outras coleções que dividam o mesmo banco.
COLLECTION = "base_de_conhecimento"

_partidor = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def base_vetorial() -> PGVector:
    """A collection de vetores, pronta para gravar ou buscar."""
    return PGVector(
        embeddings=BedrockEmbeddings(
            model_id=EMBEDDING_MODEL, region_name=REGIAO_AWS
        ),
        collection_name=COLLECTION,
        connection=db.url_sqlalchemy(),
        use_jsonb=True,
    )


def indexar(documentos: list[tuple[str, str, str]]) -> int:
    """Indexa [(arquivo, titulo, conteudo)] e devolve o total de chunks gravados.

    O id de cada chunk é `arquivo::posição`, e é isso que torna a operação
    idempotente: reindexar o mesmo documento sobrescreve os mesmos ids em vez de
    duplicar o conteúdo na base.

    O log de cada etapa é o que permite acompanhar a indexação acontecendo. Sem
    ele, indexar é uma pausa silenciosa e depois uma busca que funciona.
    """
    chunks: list[Document] = []
    ids: list[str] = []

    for arquivo, titulo, conteudo in documentos:
        pedacos = _partidor.split_text(conteudo)
        logger.info(
            "indexando %s: %d caracteres -> %d chunk(s)",
            arquivo, len(conteudo), len(pedacos),
        )
        for posicao, pedaco in enumerate(pedacos):
            chunks.append(
                Document(
                    page_content=pedaco,
                    metadata={"arquivo": arquivo, "titulo": titulo, "chunk": posicao},
                )
            )
            ids.append(f"{arquivo}::{posicao}")

    if not chunks:
        return 0

    logger.info(
        "gerando embeddings de %d chunk(s) com %s", len(chunks), EMBEDDING_MODEL
    )
    base_vetorial().add_documents(chunks, ids=ids)
    logger.info("%d chunk(s) gravados na collection '%s'", len(chunks), COLLECTION)
    return len(chunks)


def buscar(pergunta: str, k: int = TOP_K) -> list[Document]:
    """Os k trechos mais próximos da pergunta, do mais próximo para o mais longe.

    O score vem anexado em `metadata['score']` — é a distância no espaço
    vetorial, então menor é mais perto. Ele serve para inspecionar o ranking, e
    **não** para decidir se a pergunta tem resposta na base: medido nesta base, as
    faixas de score de pergunta coberta e não coberta se sobrepõem. Uma pergunta
    sem resposta na base pontua melhor que várias que têm. Quem recusa é a
    instrução no prompt, não um limiar aqui.
    """
    encontrados = base_vetorial().similarity_search_with_score(pergunta, k=k)

    trechos = []
    for posicao, (trecho, score) in enumerate(encontrados, start=1):
        trecho.metadata["score"] = score
        trechos.append(trecho)
        logger.info(
            "busca %dº %s score=%.4f", posicao, trecho.metadata.get("arquivo"), score
        )
    return trechos


def contar() -> int:
    """Quantos chunks estão indexados agora.

    Lê a tabela interna do langchain-postgres porque o PGVector não expõe
    contagem. Base que nunca foi criada conta zero em vez de estourar.
    """
    sql = (
        "SELECT count(*) FROM langchain_pg_embedding e "
        "JOIN langchain_pg_collection c ON c.uuid = e.collection_id "
        "WHERE c.name = %s"
    )
    try:
        with psycopg.connect(db.dsn(), connect_timeout=5) as conexao:
            return conexao.execute(sql, (COLLECTION,)).fetchone()[0]
    except Exception:
        return 0


def esvaziar() -> None:
    """Apaga a collection inteira. Ela é recriada na próxima indexação."""
    base_vetorial().delete_collection()
    logger.info("collection '%s' esvaziada", COLLECTION)
