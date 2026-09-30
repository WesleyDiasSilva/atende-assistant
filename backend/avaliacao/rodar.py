# -*- coding: utf-8 -*-
"""Runner da suíte de avaliação.

Carrega os casos de `avaliacao/casos.json`, invoca o grafo uma vez por caso e
confronta o que a execução produziu com o gabarito declarado no caso. Não há
framework de teste envolvido: um caso é um dicionário de dados e a suíte é um
laço sobre eles.

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
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from app.grafo import ESTADO_DO_TURNO, compilar_grafo
from avaliacao import juiz, regua
from avaliacao.isolamento import fila_isolada
from avaliacao.sondas import SondaDaExecucao

CASOS_PATH = Path(__file__).resolve().parent / "casos.json"

# Critérios que alguém sabe medir: os determinísticos da régua e o do juiz. Um
# critério declarado num caso e ausente daqui aparece como não avaliado, para
# não passar por aprovado no silêncio.
RECONHECIDOS = set(regua.CRITERIOS) | {juiz.CRITERIO}

# Largura da coluna do nome do caso, para o veredito ficar alinhado.
COLUNA = 44

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


def executar(grafo, caso: dict, sonda: SondaDaExecucao) -> dict:
    """Invoca o grafo para um caso e devolve o estado final completo.

    A sonda entra pelo config, a mesma porta do tracing: ela observa, e o
    fluxo não sabe que está sendo observado.
    """
    with fila_isolada():
        return grafo.invoke(estado_inicial(caso), config={"callbacks": [sonda]})


def avaliar_caso(grafo, caso: dict, execucoes: dict | None = None):
    """Executa um caso e aplica os critérios. Devolve (vereditos, não avaliados, estado).

    Um caso pode declarar `mesma_execucao_de`: em vez de invocar o grafo de novo,
    ele reaproveita a execução de outro caso desta rodada e só aplica os próprios
    critérios. É o que permite comparar dois critérios sobre a **mesma** resposta
    — sem isso, a diferença de veredito poderia vir da resposta ter mudado, e o
    que se quer isolar é a redação do critério. Rodado sozinho, o caso executa.

    Erro na invocação vira veredito reprovado: uma exceção não pode passar por
    caso aprovado, nem derrubar a suíte inteira.
    """
    criterios = caso.get("criterios", {})
    execucoes = execucoes if execucoes is not None else {}
    origem = caso.get("mesma_execucao_de")
    if origem and origem in execucoes:
        estado, sonda = execucoes[origem]
    else:
        sonda = SondaDaExecucao()
        try:
            estado = executar(grafo, caso, sonda)
        except Exception as erro:
            return [regua.Veredito("execucao", False, f"{type(erro).__name__}: {erro}")], [], None
    execucoes[caso["id"]] = (estado, sonda)
    vereditos = regua.avaliar(estado, sonda, criterios)
    # O juiz roda depois da régua, mas não recebe o resultado dela: saber que os
    # critérios em código passaram o inclinaria a concordar com eles.
    if juiz.CRITERIO in criterios:
        vereditos.append(juiz.avaliar(caso, estado, criterios[juiz.CRITERIO]))
    nao_avaliados = [nome for nome in criterios if nome not in RECONHECIDOS]
    return vereditos, nao_avaliados, estado


def passou_caso(vereditos: list[regua.Veredito]) -> bool:
    """Um caso passa quando todo critério efetivamente medido foi aprovado."""
    return all(v.ok for v in vereditos if v.avaliado)


# --- Saída -------------------------------------------------------------------


def _cabecalho(nome: str, veredito: str) -> str:
    """Nome do caso com preenchimento pontilhado até a coluna do veredito."""
    return f"{nome} {'.' * max(3, COLUNA - len(nome))} {veredito}"


def imprimir_caso(caso: dict, vereditos, nao_avaliados, estado) -> None:
    print()
    print(_cabecalho(caso["id"], "passou" if passou_caso(vereditos) else "FALHOU"))
    for v in vereditos:
        marca = "·" if not v.avaliado else ("✓" if v.ok else "✗")
        print(f"  {marca} {v.criterio:<14}{v.detalhe}")
    for nome in nao_avaliados:
        print(f"  · {nome:<14}(nenhum critério com esse nome — não avaliado)")
    if estado is not None:
        print(f"  · {'caminho':<14}{' → '.join(estado.get('trajetoria') or [])}")


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
    resultados: dict[str, bool] = {}
    execucoes: dict = {}
    for caso in casos:
        vereditos, nao_avaliados, estado = avaliar_caso(grafo, caso, execucoes)
        imprimir_caso(caso, vereditos, nao_avaliados, estado)
        resultados[caso["id"]] = passou_caso(vereditos)

    reprovados = [cid for cid, ok in resultados.items() if not ok]
    print()
    print("─" * (COLUNA + 12))
    print(f"{len(casos) - len(reprovados)}/{len(casos)} casos passaram")
    if reprovados:
        print(f"reprovados: {', '.join(reprovados)}")
    return 1 if reprovados else 0


if __name__ == "__main__":
    sys.exit(main())
