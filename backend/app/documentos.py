# -*- coding: utf-8 -*-
"""A base de conhecimento em arquivos: os `.md` de `dados/base/`.

São os documentos da empresa — políticas, prazos, guias de produto —, o
conhecimento que não está no cadastro do pedido e que hoje o atendente não tem.
Ficam numa subpasta própria porque `dados/` já guarda outra coisa: o cadastro de
pedidos e os registros gravados em execução.

Isto é encanamento de arquivo, não a chain: aqui só se lê e se escreve `.md`.
Quem decide o que fazer com o conteúdo é o assistente.
"""
from pathlib import Path

PASTA = Path(__file__).resolve().parent.parent / "dados" / "base"


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
    for caminho in sorted(PASTA.glob("*.md")):
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
    caminho = PASTA / _nome_seguro(arquivo)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(conteudo, encoding="utf-8")
    return caminho.name, _titulo(conteudo)


def remover(arquivo: str) -> bool:
    """Apaga um `.md` da base. Devolve False se ele não existia."""
    caminho = PASTA / _nome_seguro(arquivo)
    if not caminho.is_file():
        return False
    caminho.unlink()
    return True
