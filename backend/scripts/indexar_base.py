# -*- coding: utf-8 -*-
"""Indexa a pasta `dados/base/` inteira na base vetorial.

    python -m scripts.indexar_base

Roda uma vez, fora da aplicação. Indexar é caro (uma chamada de embedding por
chunk) e não muda entre perguntas: fazer isso no boot da API custaria a mesma
conta a cada reinício, e fazer a cada pergunta custaria a cada pergunta.

É idempotente — os ids dos chunks são estáveis, então rodar de novo sobrescreve
em vez de duplicar.
"""
from dotenv import load_dotenv

load_dotenv()

from app import db, documentos, log, retrieval  # noqa: E402  (depois do load_dotenv)

log.configurar()


def main() -> None:
    db.garantir_extensao_vector()

    base = documentos.carregar()
    print(f"{len(base)} documento(s) em {documentos.PASTA}")

    chunks = retrieval.indexar(base)
    print(
        f"\n{len(base)} documento(s) -> {chunks} chunk(s) "
        f"(chunk_size={retrieval.CHUNK_SIZE}, overlap={retrieval.CHUNK_OVERLAP})"
    )
    print(f"total na collection '{retrieval.COLLECTION}': {retrieval.contar()} chunk(s)")


if __name__ == "__main__":
    main()
