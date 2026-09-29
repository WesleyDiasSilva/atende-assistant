# -*- coding: utf-8 -*-
"""O que a suíte isola para medir sempre a mesma coisa.

Uma das ferramentas grava: abrir troca deixa um registro na fila, e a regra de
negócio recusa a segunda troca do mesmo pedido. Sem isolamento, a rodada
seguinte encontraria a troca que a rodada anterior abriu e responderia outra
coisa — um vermelho que não vem do sistema, vem da suíte.
"""
from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path

from app import dados


@contextmanager
def fila_isolada():
    """Aponta a fila de solicitações para um arquivo temporário vazio.

    Cada caso começa com a fila vazia, e ela é descartada no fim. Funciona
    porque o caminho do arquivo é lido no momento da chamada, e não capturado
    no import. O código do sistema não muda: muda o arquivo que ele encontra.
    """
    original = dados.ARQUIVO_DE_SOLICITACOES
    with tempfile.TemporaryDirectory() as pasta:
        dados.ARQUIVO_DE_SOLICITACOES = Path(pasta) / "solicitacoes.jsonl"
        try:
            yield
        finally:
            dados.ARQUIVO_DE_SOLICITACOES = original
