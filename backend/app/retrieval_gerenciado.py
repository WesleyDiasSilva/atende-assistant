# -*- coding: utf-8 -*-
"""A mesma busca, feita por um serviço gerenciado: Amazon Bedrock Knowledge Base.

Este arquivo existe para ser lido ao lado do `retrieval.py`. Os dois respondem à
mesma pergunta — "quais trechos da base estão mais perto disto" — e a diferença
entre eles é o assunto:

    retrieval.py             este arquivo
    ------------             ------------
    nós partimos o texto     a AWS parte, e mede em tokens, não em caracteres
    nós chamamos o embedding a AWS chama
    nós guardamos no pgvector a AWS guarda (S3 Vectors)
    nós fazemos a query      a AWS faz
    ~90 linhas               uma chamada

O que **não** muda: o `TOP_K` é o mesmo dial, aqui com o nome `numberOfResults`;
o contexto entra na conversa pela mesma `SystemMessage`; e a instrução de
groundedness é literalmente a mesma. Se as respostas divergirem, a diferença está
no retriever, não no prompt — é isso que faz a comparação valer.

O preço da conveniência é o controle: o chunking, a normalização e o índice
passam a ser decisões de outra pessoa. Trocá-los exige reindexar do lado de lá.
"""
import logging
import os

from langchain_core.documents import Document

from app.config import REGIAO_AWS
from app.retrieval import TOP_K

logger = logging.getLogger(__name__)

# O identificador do Knowledge Base provisionado na conta. Fica em variável de
# ambiente porque é um recurso de nuvem, não uma escolha de código: quem clona o
# repositório sem uma conta AWS não tem esse valor, e o modo se desabilita em vez
# de quebrar a aplicação.
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID", "").strip()


def esta_configurado() -> bool:
    """Se dá para usar o modo gerenciado nesta instalação."""
    return bool(KNOWLEDGE_BASE_ID)


def buscar(pergunta: str, k: int = TOP_K) -> list[Document]:
    """Os k trechos que o Knowledge Base considera mais próximos da pergunta.

    Devolve `Document` como o `retrieval.buscar`, e com a mesma chave de metadata
    (`arquivo`), para o resto da aplicação não saber de qual dos dois veio. Foi o
    que permitiu acrescentar este modo sem tocar no `responder()` além de um
    `elif`.

    O import é local de propósito: quem não configurou o Knowledge Base não paga
    o custo de carregar o retriever a cada boot.
    """
    from langchain_aws import AmazonKnowledgeBasesRetriever

    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        region_name=REGIAO_AWS,
        # `numberOfResults` é o `TOP_K` com outro nome. O mesmo dial, do outro
        # lado da fronteira: aqui quem o aplica é o serviço.
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": k}},
    )

    encontrados = retriever.invoke(pergunta)

    trechos = []
    for posicao, trecho in enumerate(encontrados, start=1):
        arquivo = _nome_do_arquivo(trecho)
        # Reescreve a metadata para a chave que o resto do código já lê. O KB
        # devolve a origem como URI do S3; o nosso pgvector, como nome de arquivo.
        trecho.metadata["arquivo"] = arquivo
        trechos.append(trecho)
        score = trecho.metadata.get("score")
        logger.info(
            "busca gerenciada %dº %s score=%s", posicao, arquivo,
            f"{score:.4f}" if isinstance(score, (int, float)) else "n/d",
        )
    return trechos


def _nome_do_arquivo(trecho: Document) -> str:
    """Extrai o nome do `.md` da localização que o Knowledge Base devolveu.

    A origem chega como URI completa (`s3://bucket/politica-frete.md`), e a
    interface mostra fonte por nome de arquivo. Sem isto, a mesma resposta citaria
    fontes com cara diferente em cada modo, e a comparação ficaria confusa por um
    motivo que não é o retriever.
    """
    local = trecho.metadata.get("location") or {}
    uri = (local.get("s3Location") or {}).get("uri", "")
    return uri.rsplit("/", 1)[-1] or "(origem não informada)"
