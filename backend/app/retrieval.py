# -*- coding: utf-8 -*-
"""Indexação e busca: transformar documento em vetor, e pergunta em trechos.

Três decisões estão neste arquivo, e todas são ajustáveis por variável de
ambiente: qual modelo transforma texto em vetor, de que tamanho são os pedaços
(com quanta sobreposição entre eles), e quantos pedaços a busca devolve.

O caminho da indexação acontece uma vez por documento: parte o texto em chunks,
manda cada chunk para o modelo de embedding e grava o vetor no Postgres. O
caminho da busca acontece a cada pergunta: transforma a pergunta em vetor pelo
**mesmo** modelo e pede ao banco os vetores mais próximos. Modelos diferentes nos
dois lados projetam em espaços diferentes, e distância entre espaços diferentes
não mede semelhança nenhuma.
"""
import logging
import os
from functools import lru_cache

import psycopg
from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

_divisor = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


@lru_cache(maxsize=1)
def base_vetorial() -> PGVector:
    """A collection de vetores, pronta para gravar ou buscar.

    Construir um PGVector abre um engine do SQLAlchemy, com pool de conexões
    próprio. Fazer isso a cada pergunta deixaria um pool novo para trás em cada
    uma — daí o cache: o objeto é montado na primeira chamada e reaproveitado.

    `esvaziar()` limpa o cache, porque depois de apagar a collection o objeto
    guardado aponta para algo que não existe mais.
    """
    return PGVector(
        embeddings=BedrockEmbeddings(
            model_id=EMBEDDING_MODEL, region_name=REGIAO_AWS
        ),
        collection_name=COLLECTION,
        connection=db.url_sqlalchemy(),
        use_jsonb=True,
    )


def indexar(documentos_para_indexar: list[tuple[str, str, str]]) -> int:
    """Indexa [(arquivo, titulo, conteudo)] e devolve o total de chunks gravados.

    O id de cada chunk é `arquivo::posição`, e é isso que torna a operação
    idempotente: reindexar o mesmo documento sobrescreve os mesmos ids em vez de
    duplicar o conteúdo na base.

    Cada chunk leva `arquivo`, `titulo` e a própria posição no metadata. O
    `arquivo` é o que a busca e a remoção usam; os outros dois existem para quem
    for olhar as linhas da tabela no psql e precisar saber de onde cada vetor veio.

    O log de cada etapa é o que permite acompanhar a indexação acontecendo. Sem
    ele, indexar é uma pausa silenciosa e depois uma busca que funciona.
    """
    chunks: list[Document] = []
    ids: list[str] = []

    for arquivo, titulo, conteudo in documentos_para_indexar:
        pedacos = _divisor.split_text(conteudo)
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

    Usa `similarity_search_with_score` para o score entrar no log: é a distância
    no espaço vetorial, então menor é mais perto, e ver o ranking com os números
    ao lado é o que torna o efeito do top-k observável.

    O score fica **só** no log, e de propósito. Ele não serve para decidir se a
    pergunta tem resposta na base: medido nesta base, as faixas de score de
    pergunta coberta e não coberta se sobrepõem, e uma pergunta que a base não
    cobre pontua melhor que várias que ela cobre. Quem recusa é a instrução no
    prompt, não um limiar aqui — devolver o score para o resto da aplicação seria
    convidar exatamente esse erro.
    """
    encontrados = base_vetorial().similarity_search_with_score(pergunta, k=k)

    trechos = []
    for posicao, (trecho, score) in enumerate(encontrados, start=1):
        trechos.append(trecho)
        logger.info(
            "busca %dº %s score=%.4f", posicao, trecho.metadata.get("arquivo"), score
        )
    return trechos


def contar_por_arquivo() -> dict[str, int]:
    """Quantos chunks cada documento tem na base, pelo metadata gravado.

    É o que permite mostrar o que está *indexado*, e não só o que está no disco:
    os dois podem divergir, e a divergência é a informação útil.

    Lê a tabela interna do langchain-postgres porque o PGVector não expõe
    contagem. Base que nunca foi criada devolve vazio em vez de estourar — mas com
    aviso no log, porque "vazio" e "banco fora do ar" dão o mesmo resultado na
    tela e só o log distingue os dois.
    """
    sql = (
        "SELECT e.cmetadata->>'arquivo' AS arquivo, count(*) "
        "FROM langchain_pg_embedding e "
        "JOIN langchain_pg_collection c ON c.uuid = e.collection_id "
        "WHERE c.name = %s GROUP BY arquivo"
    )
    try:
        with psycopg.connect(db.dsn(), connect_timeout=5) as conexao:
            return dict(conexao.execute(sql, (COLLECTION,)).fetchall())
    except Exception as erro:
        logger.warning("nao foi possivel contar os chunks indexados: %s", erro)
        return {}


def contar() -> int:
    """O total de chunks indexados."""
    return sum(contar_por_arquivo().values())


def remover(arquivo: str) -> int:
    """Apaga os chunks de um documento e devolve quantos saíram.

    Vai por SQL porque a exclusão é por metadata, não por id: quantos chunks o
    documento gerou é coisa que só a indexação sabia.

    Falha volta como 0 para a rota responder limpo em vez de estourar 500, mas
    **com aviso no log**: um 0 silencioso aqui é o que deixaria chunk órfão para
    trás, e a busca continuaria encontrando texto de um documento apagado.
    """
    sql = (
        "DELETE FROM langchain_pg_embedding e "
        "USING langchain_pg_collection c "
        "WHERE e.collection_id = c.uuid AND c.name = %s "
        "AND e.cmetadata->>'arquivo' = %s"
    )
    try:
        with psycopg.connect(db.dsn(), connect_timeout=5) as conexao:
            removidos = conexao.execute(sql, (COLLECTION, arquivo)).rowcount
        logger.info("removidos %d chunk(s) de %s", removidos, arquivo)
        return removidos
    except Exception as erro:
        logger.warning("nao foi possivel remover os chunks de %s: %s", arquivo, erro)
        return 0


def esvaziar() -> None:
    """Apaga a collection inteira. Ela é recriada na próxima indexação."""
    base_vetorial().delete_collection()
    # O objeto guardado no cache aponta para uma collection que já não existe.
    base_vetorial.cache_clear()
    logger.info("collection '%s' esvaziada", COLLECTION)
