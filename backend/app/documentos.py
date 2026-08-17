# -*- coding: utf-8 -*-
"""A base de conhecimento em arquivos: os `.md` de `dados/base/`.

São os documentos da empresa — políticas, prazos, guias de produto —, o
conhecimento que não está no cadastro do pedido. Ficam numa subpasta própria
porque `dados/` já guarda outra coisa: o cadastro de pedidos e os registros
gravados em execução.

Isto é encanamento de arquivo, não a chain: aqui só se lê e se escreve `.md`.
Quem decide se esse conteúdo chega ao modelo, e quanto dele chega, é o assistente.
"""
from pathlib import Path
from typing import Optional

PASTA_DA_BASE =Path(__file__).resolve().parent.parent / "dados" / "base"


def _titulo(conteudo: str) -> str:
    """O título é a primeira linha do `.md`, sem o `#`."""
    primeira = conteudo.lstrip().splitlines()[0] if conteudo.strip() else ""
    return primeira.lstrip("#").strip() or "(sem título)"


def carregar() -> list[tuple[str, str, str]]:
    """Devolve [(arquivo, titulo, conteudo)] em ordem alfabética de arquivo.

    Ordem alfabética para o contexto montado a partir daqui sair sempre igual:
    a mesma pergunta com os mesmos documentos tem que dar a mesma resposta.
    """
    itens = []
    for caminho in sorted(PASTA_DA_BASE.glob("*.md")):
        conteudo = caminho.read_text(encoding="utf-8")
        itens.append((caminho.name, _titulo(conteudo), conteudo))
    return itens


def listar() -> list[dict]:
    """Só arquivo e título — é o que a interface precisa para listar a base."""
    return [{"arquivo": arquivo, "titulo": titulo} for arquivo, titulo, _ in carregar()]


def _nome_seguro(arquivo: str) -> str:
    """Reduz o nome recebido de fora ao nome do arquivo, sem caminho.

    Um upload chega com o nome que o cliente enviar, e `../../app/main.py` é um
    nome válido de arquivo. `Path(...).name` descarta qualquer diretório, então o
    que sobra só pode cair dentro de `dados/base/`.
    """
    return Path(arquivo).name


def gravar(arquivo: str, conteudo: str) -> tuple[str, str]:
    """Grava um `.md` na base e devolve (arquivo, titulo).

    Sobrescreve quando o nome já existe: a base é uma pasta, e mandar o mesmo
    documento de novo é substituí-lo, não criar um segundo.
    """
    caminho = PASTA_DA_BASE / _nome_seguro(arquivo)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(conteudo, encoding="utf-8")
    return caminho.name, _titulo(conteudo)


def remover(arquivo: str) -> Optional[str]:
    """Apaga um `.md` da base e devolve o nome apagado, ou None se não existia.

    Devolve o nome já reduzido pelo `_nome_seguro`, e não o que chegou de fora: é
    esse nome que está gravado no metadata dos chunks, e é com ele que os vetores
    do documento precisam ser apagados. Quem apagasse o arquivo por um nome e os
    vetores por outro deixaria a base incoerente — arquivo fora, vetor dentro.
    """
    caminho = PASTA_DA_BASE / _nome_seguro(arquivo)
    if not caminho.is_file():
        return None
    caminho.unlink()
    return caminho.name
