# -*- coding: utf-8 -*-
"""Esvazia a base vetorial, sem tocar nos arquivos `.md`.

    python -m scripts.limpar_base

Os documentos continuam no disco: o que se apaga aqui são os vetores. Depois
disto, o modo de busca não encontra nada até a base ser indexada de novo.
"""
from dotenv import load_dotenv

load_dotenv()

from app import log, retrieval  # noqa: E402  (depois do load_dotenv)

log.configurar()


def main() -> None:
    antes = retrieval.contar()
    retrieval.esvaziar()
    print(f"{antes} chunk(s) apagados. Agora: {retrieval.contar()} chunk(s).")
    print("Os arquivos .md de dados/base/ continuam intactos.")


if __name__ == "__main__":
    main()
