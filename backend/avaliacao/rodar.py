# -*- coding: utf-8 -*-
"""Runner da suíte de avaliação.

Carrega os casos de `avaliacao/casos.json`, invoca o grafo uma vez por caso e
confronta o que a execução produziu com o gabarito declarado no caso. Não há
framework de teste envolvido: um caso é um dicionário de dados e a suíte é um
laço sobre eles.

Execução, dentro do conteiner, onde as dependências estão instaladas:

    docker compose exec backend python -m avaliacao.rodar
    docker compose exec backend python -m avaliacao.rodar --rapido
    docker compose exec backend python -m avaliacao.rodar --caso regra-prazo-troca
    docker compose exec backend python -m avaliacao.rodar --caso regra-prazo-troca --repeticoes 5

Ao fim de cada rodada o runner compara o resultado com o da rodada anterior e
imprime o delta, no formato "12/14 → 9/14, 3 regressões". A base de comparação
fica em `avaliacao/.ultima-rodada.json`, fora do git.

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
from avaliacao import juiz, plataforma, regua
from avaliacao.isolamento import fila_isolada
from avaliacao.sondas import SondaDaExecucao

CASOS_PATH = Path(__file__).resolve().parent / "casos.json"

# Critérios que alguém sabe medir: os determinísticos da régua e o do juiz. Um
# critério declarado num caso e ausente daqui aparece como não avaliado, para
# não passar por aprovado no silêncio.
RECONHECIDOS = set(regua.CRITERIOS) | {juiz.CRITERIO}

# Resultado da última rodada, base de comparação do delta. Fica fora do git: é
# estado local de quem roda a suíte, não conteúdo do projeto.
ULTIMA_PATH = Path(__file__).resolve().parent / ".ultima-rodada.json"

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


def selecionar(casos: list[dict], ids: str | None, rapido: bool) -> list[dict]:
    """Aplica os filtros de seleção, que compõem entre si.

    `ids` aceita vários, separados por vírgula. Um caso que reaproveita a
    execução de outro puxa esse outro junto, antes dele — sem a execução de
    origem, ele rodaria sozinho e compararia uma resposta diferente.
    """
    selecionados = casos
    if rapido:
        selecionados = [c for c in selecionados if c.get("rapido")]
    if ids:
        pedidos = {i.strip() for i in ids.split(",")}
        selecionados = [c for c in selecionados if c["id"] in pedidos]
    ids_escolhidos = {c["id"] for c in selecionados}
    for c in selecionados:
        if c.get("mesma_execucao_de"):
            ids_escolhidos.add(c["mesma_execucao_de"])
    return [c for c in casos if c["id"] in ids_escolhidos]


# --- Comparação com a rodada anterior ----------------------------------------


def carregar_ultima() -> dict[str, bool]:
    """O resultado por caso da rodada anterior. Ausente ou ilegível → vazio."""
    try:
        return json.loads(ULTIMA_PATH.read_text(encoding="utf-8"))["resultados"]
    except Exception:
        return {}


def gravar_ultima(anterior: dict[str, bool], atual: dict[str, bool]) -> None:
    """Grava o resultado por caso, mesclando com o que já havia.

    A mescla preserva o resultado dos casos que não rodaram nesta seleção — sem
    ela, uma rodada no modo rápido apagaria a base de comparação dos demais.
    """
    ULTIMA_PATH.write_text(
        json.dumps({"resultados": {**anterior, **atual}}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def calcular_delta(anterior: dict[str, bool], atual: dict[str, bool]) -> dict | None:
    """O placar antes e agora, e quem regrediu ou recuperou.

    A comparação cobre só os casos com resultado antes **e** agora: comparar
    contagens de conjuntos diferentes de casos daria um número que parece medir
    regressão e não mede. `None` quando não há nada comparável.
    """
    comparaveis = [cid for cid in atual if cid in anterior]
    if not comparaveis:
        return None
    return {
        "total": len(comparaveis),
        "antes": sum(1 for cid in comparaveis if anterior[cid]),
        "agora": sum(1 for cid in comparaveis if atual[cid]),
        "regressoes": [cid for cid in comparaveis if anterior[cid] and not atual[cid]],
        "recuperacoes": [cid for cid in comparaveis if not anterior[cid] and atual[cid]],
        "sem_base": len(atual) - len(comparaveis),
    }


def imprimir_delta(delta: dict | None) -> None:
    if delta is None:
        print("delta: sem rodada anterior para comparar")
        return
    total = delta["total"]
    print(
        f"delta: {delta['antes']}/{total} → {delta['agora']}/{total}, "
        f"{len(delta['regressoes'])} regressões"
    )
    if delta["regressoes"]:
        print(f"  regrediram:  {', '.join(delta['regressoes'])}")
    if delta["recuperacoes"]:
        print(f"  recuperaram: {', '.join(delta['recuperacoes'])}")
    if delta["sem_base"]:
        print(f"  ({delta['sem_base']} caso(s) sem rodada anterior, fora da comparação)")


# --- Execução ----------------------------------------------------------------


def executar(grafo, caso: dict, sonda: SondaDaExecucao) -> tuple[dict, str | None]:
    """Invoca o grafo para um caso. Devolve o estado final e o id da trace.

    A sonda entra pelo config, a mesma porta do tracing: ela observa, e o
    fluxo não sabe que está sendo observado. O id da trace é lido logo depois
    do `invoke()`, quando o handler o conhece; sem LangFuse, é `None`.
    """
    config, handler = plataforma.config_do_caso(caso["id"], [sonda])
    with fila_isolada():
        estado = grafo.invoke(estado_inicial(caso), config=config)
    return estado, getattr(handler, "last_trace_id", None)


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
    reaproveitado = bool(origem and origem in execucoes)
    if reaproveitado:
        estado, sonda, trace_id = execucoes[origem]
    else:
        sonda = SondaDaExecucao()
        try:
            estado, trace_id = executar(grafo, caso, sonda)
        except Exception as erro:
            return [regua.Veredito("execucao", False, f"{type(erro).__name__}: {erro}")], [], None
    execucoes[caso["id"]] = (estado, sonda, trace_id)
    vereditos = regua.avaliar(estado, sonda, criterios)
    # O juiz roda depois da régua, mas não recebe o resultado dela: saber que os
    # critérios em código passaram o inclinaria a concordar com eles.
    if juiz.CRITERIO in criterios:
        vereditos.append(juiz.avaliar(caso, estado, criterios[juiz.CRITERIO]))
    nao_avaliados = [nome for nome in criterios if nome not in RECONHECIDOS]
    plataforma.enviar_scores(trace_id, caso["id"], vereditos, reaproveitado)
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
        # O groundedness é sempre mostrado e nunca decide: é um cosseno, cego à
        # negação, e dá nota baixa justamente à recusa correta.
        g = estado.get("groundedness")
        print(f"  · {'groundedness':<14}{'—' if g is None else g}  (reportado, nunca decide)")


def imprimir_repeticoes(caso: dict, rodadas: list[list[regua.Veredito]], nao_avaliados) -> None:
    """O resultado agregado de um caso executado N vezes.

    Relata todo critério que reprovou em alguma rodada, separando o que reprova
    **sempre** do que **oscilou** — um critério que muda de veredito com a mesma
    entrada é informação, não ruído a ser suprimido.
    """
    total = len(rodadas)
    passaram = sum(1 for vereditos in rodadas if passou_caso(vereditos))
    print()
    print(_cabecalho(caso["id"], f"{passaram}/{total} passaram"))
    por_criterio: dict[str, list[regua.Veredito]] = {}
    for vereditos in rodadas:
        for v in vereditos:
            por_criterio.setdefault(v.criterio, []).append(v)
    if passaram == total:
        print(f"  ✓ {'todos':<14}{total} de {total} rodadas com todos os critérios aprovados")
    for criterio, vs in por_criterio.items():
        nao_medidos = [v for v in vs if not v.avaliado]
        if nao_medidos:
            print(f"  · {criterio:<14}não avaliado em {len(nao_medidos)} de {total} rodadas")
        medidos = [v for v in vs if v.avaliado]
        resultados = {v.ok for v in medidos}
        if not medidos or resultados == {True}:
            continue
        reprovas = sum(1 for v in medidos if not v.ok)
        oscilou = len(resultados) > 1
        rotulo = (
            f"OSCILOU — {reprovas} de {len(medidos)} rodadas reprovaram"
            if oscilou
            else f"reprovou {reprovas} de {len(medidos)}"
        )
        print(f"  {'!' if oscilou else '✗'} {criterio:<14}{rotulo}")
        # Motivos distintos: num critério que oscilou, é a comparação entre eles
        # que mostra o que mudou de uma rodada para a outra.
        for detalhe in sorted({f"[{'aprovado' if v.ok else 'reprovado'}] {v.detalhe}" for v in medidos}):
            print(f"      {detalhe}")
    for nome in nao_avaliados:
        print(f"  · {nome:<14}(nenhum critério com esse nome — não avaliado)")


# --- Entrada -----------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Suíte de avaliação do atendimento.")
    p.add_argument("--caso", help="Executa só os casos com estes ids, separados por vírgula.")
    p.add_argument(
        "--rapido", action="store_true",
        help='Executa só o conjunto curto (casos marcados com "rapido" em casos.json).',
    )
    p.add_argument(
        "--repeticoes", type=int, default=1,
        help="Executa cada caso N vezes e reporta quantas passaram (expõe oscilação).",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    casos = selecionar(carregar_dados()["casos"], args.caso, args.rapido)
    if not casos:
        print("Nenhum caso corresponde à seleção.")
        return 1

    grafo = compilar_grafo()
    escopo = "conjunto rápido" if args.rapido else f"{len(casos)} casos"
    sufixo = f" × {args.repeticoes} repetições" if args.repeticoes > 1 else ""
    print(f"Suíte de avaliação — {escopo}{sufixo}")
    if plataforma.disponivel():
        print(f"LangFuse: traces e scores na sessão {plataforma.RODADA_ID}")
    resultados: dict[str, bool] = {}
    if args.repeticoes > 1:
        for caso in casos:
            rodadas, nao_avaliados = [], []
            for _ in range(args.repeticoes):
                # Cada repetição executa de novo, inclusive o caso que
                # reaproveitaria outro: o que se mede aqui é a variação.
                vereditos, nao_avaliados, _ = avaliar_caso(grafo, {**caso, "mesma_execucao_de": None})
                rodadas.append(vereditos)
            imprimir_repeticoes(caso, rodadas, nao_avaliados)
            resultados[caso["id"]] = all(passou_caso(r) for r in rodadas)
    else:
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

    # Regressão se mede contra a rodada anterior, não contra um número absoluto.
    # Só a execução única alimenta a base: com repetições o critério é mais
    # estrito (todas têm de passar), e misturar os dois tornaria o delta
    # incomparável.
    anterior = carregar_ultima()
    imprimir_delta(calcular_delta(anterior, resultados))
    if args.repeticoes == 1:
        gravar_ultima(anterior, resultados)

    # Falha de envio é reportada e não muda o código de saída: a plataforma é
    # destino do resultado, não parte do critério.
    falhas_plataforma = plataforma.finalizar()
    if falhas_plataforma:
        print(f"LangFuse: {len(falhas_plataforma)} falha(s) de envio")
        for f in falhas_plataforma[:5]:
            print(f"  {f}")
    return 1 if reprovados else 0


if __name__ == "__main__":
    sys.exit(main())
