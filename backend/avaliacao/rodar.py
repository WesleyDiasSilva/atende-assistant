# -*- coding: utf-8 -*-
"""Runner da suíte de avaliação.

Carrega os casos de `avaliacao/casos.json`, invoca o grafo uma vez por caso e
mostra o que cada execução produziu. Não há framework de teste envolvido: um
caso é um dicionário de dados e a suíte é um laço sobre eles.

Execução, dentro do conteiner, onde as dependências estão instaladas:

    docker compose exec backend python -m avaliacao.rodar

O grafo é compilado **sem checkpointer**. A suíte não conversa com a API nem
com o banco de estado: cada caso começa do zero, e quando um caso precisa de
conversa anterior, ela vem declarada no próprio caso e entra pelo estado
inicial — o campo `historico` usa o reducer `add_messages`, que aceita mensagens
já no input.

O estado inicial é o mesmo que a aplicação monta a cada turno
(`ESTADO_DO_TURNO` mais os controles), importado e não copiado: se o fluxo
ganhar um campo de turno, a suíte o recebe junto.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from app import dados
from app.grafo import ESTADO_DO_TURNO, compilar_grafo

CASOS_PATH = Path(__file__).resolve().parent / "casos.json"

# Os controles da interface, com os valores que a suíte usa quando o caso não
# diz outra coisa. Perfil objetivo porque resposta curta é mais fácil de ler e
# mais barata de julgar; temperatura zero porque a suíte quer a resposta mais
# repetível que o sistema consegue dar; modo `rag` porque é o caminho que tem
# base conhecida e contável.
ENTRADAS_PADRAO = {
    "perfil": "objetivo",
    "modelo": "rapido",
    "temperatura": 0.0,
    "modo": "rag",
    "auto_corrigir": False,
    "memoria_ativa": True,
}

# Papéis aceitos no histórico declarado de um caso.
PAPEIS = {"cliente": HumanMessage, "atendente": AIMessage}


# --- Carga dos casos ---------------------------------------------------------


def carregar_dados(path: Path = CASOS_PATH) -> dict:
    """Lê o arquivo de dados da suíte.

    Os casos são dado, não código: acrescentar um caso é editar o JSON, sem
    tocar no runner.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def montar_historico(caso: dict) -> list:
    """Converte o histórico declarado no caso em mensagens do LangChain."""
    return [PAPEIS[m["papel"]](content=m["texto"]) for m in caso.get("historico", [])]


def estado_inicial(caso: dict) -> dict:
    """O estado com que o grafo começa este caso."""
    entradas = {**ENTRADAS_PADRAO, **caso.get("entradas", {})}
    return {
        **ESTADO_DO_TURNO,
        **entradas,
        "pergunta": caso["pergunta"],
        "historico": montar_historico(caso),
    }


# --- Execução ----------------------------------------------------------------


@contextmanager
def fila_isolada():
    """Aponta a fila de solicitações para um arquivo temporário durante o caso.

    Uma das ferramentas grava: abrir troca deixa um registro na fila, e a regra
    de negócio recusa a segunda troca do mesmo pedido. Sem isolamento, a rodada
    seguinte encontraria a troca que a rodada anterior abriu e responderia
    outra coisa — um vermelho que não vem do sistema, vem da suíte. Cada caso
    começa com a fila vazia e ela é descartada no fim.

    Funciona porque o caminho do arquivo é lido no momento da chamada, e não
    capturado no import.
    """
    original = dados.ARQUIVO_DE_SOLICITACOES
    with tempfile.TemporaryDirectory() as pasta:
        dados.ARQUIVO_DE_SOLICITACOES = Path(pasta) / "solicitacoes.jsonl"
        try:
            yield
        finally:
            dados.ARQUIVO_DE_SOLICITACOES = original


def executar(grafo, caso: dict, config: dict | None = None) -> dict:
    """Invoca o grafo para um caso e devolve o estado final completo."""
    with fila_isolada():
        return grafo.invoke(estado_inicial(caso), config=config or {})


# --- Entrada -----------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Suíte de avaliação do atendimento.")
    p.add_argument("--caso", help="Executa apenas o caso com este id.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    casos = carregar_dados()["casos"]
    if args.caso:
        casos = [c for c in casos if c["id"] == args.caso]
    if not casos:
        print("Nenhum caso corresponde à seleção.")
        return 1

    grafo = compilar_grafo()
    print(f"Suíte de avaliação — {len(casos)} casos")
    for caso in casos:
        estado = executar(grafo, caso)
        atendimento = estado.get("atendimento")
        print()
        print(caso["id"])
        print(f"  caminho   {' → '.join(estado.get('trajetoria') or [])}")
        if atendimento is not None:
            print(f"  resposta  {atendimento.resposta.resposta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
