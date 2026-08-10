# -*- coding: utf-8 -*-
"""Persistência em arquivo.

O projeto não tem banco: cada registro é uma linha JSON num arquivo, formato
que se lê a olho nu e se acompanha crescendo durante uma demonstração. A pasta
fica fora do pacote da aplicação para separar código de dado gravado.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

PASTA_DE_DADOS = Path(__file__).resolve().parent.parent / "dados"
ARQUIVO_DE_ATENDIMENTOS = PASTA_DE_DADOS / "atendimentos.jsonl"
ARQUIVO_DE_SOLICITACOES = PASTA_DE_DADOS / "solicitacoes.jsonl"
ARQUIVO_DE_PEDIDOS = PASTA_DE_DADOS / "pedidos.json"


def _agora() -> str:
    """Horário em UTC, no formato ISO — ordenável como texto."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _acrescentar_linha(arquivo: Path, registro: dict) -> None:
    """Grava um registro por linha, criando a pasta na primeira escrita.

    O modo append evita reescrever o arquivo inteiro a cada gravação e mantém
    o histórico na ordem em que aconteceu.
    """
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    with arquivo.open("a", encoding="utf-8") as destino:
        destino.write(json.dumps(registro, ensure_ascii=False) + "\n")


def _ler_linhas(arquivo: Path) -> list[dict]:
    """Lê todos os registros de um arquivo de linhas JSON.

    Linha corrompida é ignorada em vez de derrubar a leitura: um arquivo
    editado à mão durante uma demonstração não pode quebrar a aplicação.
    """
    if not arquivo.exists():
        return []

    registros = []
    with arquivo.open(encoding="utf-8") as origem:
        for linha in origem:
            linha = linha.strip()
            if not linha:
                continue
            try:
                registros.append(json.loads(linha))
            except json.JSONDecodeError:
                continue
    return registros


def registrar_atendimento(tipo: str, pergunta: str, modelo: str) -> None:
    """Guarda uma linha por mensagem respondida.

    É o que alimenta a contagem por assunto: sem o campo `tipo` vindo da
    resposta, não haveria como classificar sem reler todas as conversas.
    """
    _acrescentar_linha(
        ARQUIVO_DE_ATENDIMENTOS,
        {
            "quando": _agora(),
            "tipo": tipo,
            "pergunta": pergunta,
            "modelo": modelo,
        },
    )


def contar_por_tipo(tipos_conhecidos: list[str]) -> dict:
    """Total de atendimentos e quantos de cada assunto.

    Recebe a lista de tipos para que um assunto sem nenhuma ocorrência apareça
    zerado, em vez de sumir da tela.
    """
    registros = _ler_linhas(ARQUIVO_DE_ATENDIMENTOS)
    contagem = {tipo: 0 for tipo in tipos_conhecidos}
    for registro in registros:
        tipo = registro.get("tipo")
        if tipo in contagem:
            contagem[tipo] += 1
    return {"total": len(registros), "por_tipo": contagem}


def ultimos_atendimentos(quantidade: int = 10) -> list[dict]:
    """Os atendimentos mais recentes, do mais novo para o mais antigo."""
    return list(reversed(_ler_linhas(ARQUIVO_DE_ATENDIMENTOS)))[:quantidade]


# ---------------------------------------------------------------------------
# Pedidos
#
# Substitui o sistema de pedidos que existiria em produção. O tempo é gravado
# em dias corridos desde a entrega, e não em data absoluta, para que os casos
# continuem valendo em qualquer dia em que o projeto for executado.
# ---------------------------------------------------------------------------


def buscar_pedido(numero_pedido: str) -> Optional[dict]:
    """Devolve o pedido pelo número, ou None quando ele não existe."""
    if not ARQUIVO_DE_PEDIDOS.exists():
        return None

    with ARQUIVO_DE_PEDIDOS.open(encoding="utf-8") as origem:
        pedidos = json.load(origem)

    return pedidos.get(str(numero_pedido).strip())


# ---------------------------------------------------------------------------
# Fila de trabalho humano
#
# Uma fila só, alimentada por caminhos diferentes: o encaminhamento, quando o
# atendimento não resolve o caso, e a ação executada em nome do cliente. O
# campo `origem` distingue os dois sem exigir duas listas.
# ---------------------------------------------------------------------------


def _novo_protocolo() -> str:
    """Identificador curto o suficiente para alguém ditar por telefone."""
    return "SOL-" + uuid4().hex[:8].upper()


def registrar_solicitacao(
    origem: str,
    assunto: str,
    motivo: str,
    pergunta: str = "",
    numero_pedido: Optional[str] = None,
) -> dict:
    """Coloca um item na fila e devolve o registro criado.

    Nasce sempre como aberta: quem decide que o caso terminou é a pessoa que
    atende, não o sistema.
    """
    solicitacao = {
        "protocolo": _novo_protocolo(),
        "quando": _agora(),
        "origem": origem,
        "assunto": assunto,
        "motivo": motivo,
        "pergunta": pergunta,
        "numero_pedido": numero_pedido,
        "situacao": "aberta",
        "resolvida_em": None,
    }
    _acrescentar_linha(ARQUIVO_DE_SOLICITACOES, solicitacao)
    return solicitacao


def listar_solicitacoes() -> list[dict]:
    """A fila inteira, da mais recente para a mais antiga."""
    return list(reversed(_ler_linhas(ARQUIVO_DE_SOLICITACOES)))


def resolver_solicitacao(protocolo: str) -> Optional[dict]:
    """Marca um item como resolvido. Devolve None se o protocolo não existir.

    Reescreve o arquivo inteiro porque a linha muda de conteúdo, e o formato de
    uma linha por registro não permite edição no lugar. Com uma fila de
    atendimento o custo é irrelevante; num volume maior isto seria um banco.
    """
    registros = _ler_linhas(ARQUIVO_DE_SOLICITACOES)
    alterado = None

    for registro in registros:
        if registro.get("protocolo") == protocolo:
            registro["situacao"] = "resolvida"
            registro["resolvida_em"] = _agora()
            alterado = registro
            break

    if alterado is None:
        return None

    with ARQUIVO_DE_SOLICITACOES.open("w", encoding="utf-8") as destino:
        for registro in registros:
            destino.write(json.dumps(registro, ensure_ascii=False) + "\n")

    return alterado
