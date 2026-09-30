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

A rodada também sai em HTML, em `avaliacao/relatorio/rodada.html`, reescrito a
cada caso (ver `relatorio.py`): abrir o arquivo no navegador é acompanhar a
rodada caso a caso.

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
import os
import sys
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from app.grafo import ESTADO_DO_TURNO, compilar_grafo
from avaliacao import juiz, plataforma, regua
from avaliacao.isolamento import fila_isolada
from avaliacao.relatorio import ARQUIVO as RELATORIO, RelatorioAoVivo
from avaliacao.sondas import SondaDaExecucao, falha_de_infraestrutura

CASOS_PATH = Path(__file__).resolve().parent / "casos.json"

# Critérios que alguém sabe medir: os determinísticos da régua e o do juiz. Um
# critério declarado num caso e ausente daqui aparece como não avaliado, para
# não passar por aprovado no silêncio.
RECONHECIDOS = set(regua.CRITERIOS) | {juiz.CRITERIO}

# Resultado da última rodada, base de comparação do delta. Fica fora do git: é
# estado local de quem roda a suíte, não conteúdo do projeto.
ULTIMA_PATH = Path(__file__).resolve().parent / ".ultima-rodada.json"

# Pausa entre casos, em segundos. O Bedrock limita requisições por minuto, e uma
# suíte é uma rajada: a pergunta composta sozinha faz quatro chamadas, duas
# delas em paralelo.
PAUSA_ENTRE_CASOS = float(os.getenv("AVALIACAO_PAUSA", "2.0"))

# Quantas vezes um caso é executado de novo quando a execução falhou por
# infraestrutura, e a espera antes de cada nova tentativa. Esgotadas, o caso sai
# como não avaliado — nunca como reprovado.
NOVAS_TENTATIVAS = (10, 25)

# Largura da coluna do nome do caso, para o veredito ficar alinhado.
COLUNA = 44

# Cor no terminal só quando a saída é um terminal de verdade: redirecionada para
# arquivo ou lida por uma integração contínua, código de cor vira lixo no texto.
COR = sys.stdout.isatty() and not os.getenv("NO_COLOR")
CORES = {"verde": "32", "vermelho": "31", "cinza": "90", "magenta": "35", "negrito": "1"}


def pintar(texto: str, cor: str) -> str:
    return f"\033[{CORES[cor]}m{texto}\033[0m" if COR else texto

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


def verificar_base(esperada: dict) -> list[str]:
    """Confere a base vetorial contra o estado declarado em casos.json.

    Uma suíte sem estado conhecido não mede nada: com um documento a menos, a
    resposta muda, e o vermelho fica ambíguo entre defeito do sistema e base
    diferente. Conferir antes troca esse vermelho ambíguo por uma mensagem que
    diz o que está diferente. Devolve as divergências; vazio quando está igual.
    """
    from app import retrieval

    indexados = retrieval.contar_por_arquivo()
    if not indexados:
        return ["a base vetorial está vazia ou o banco não respondeu"]
    divergencias: list[str] = []
    esperados = esperada.get("arquivos", [])
    faltando = [a for a in esperados if a not in indexados]
    sobrando = [a for a in indexados if a not in esperados]
    if faltando:
        divergencias.append(f"documentos ausentes da base: {', '.join(faltando)}")
    if sobrando:
        divergencias.append(f"documentos a mais na base: {', '.join(sobrando)}")
    chunks_esperados = esperada.get("chunks")
    chunks_atuais = sum(indexados.values())
    if chunks_esperados is not None and chunks_atuais != chunks_esperados:
        divergencias.append(f"chunks: esperado {chunks_esperados}, encontrado {chunks_atuais}")
    return divergencias


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


def calcular_delta(
    anterior: dict[str, bool], atual: dict[str, bool], vermelhos_esperados: set[str] = frozenset()
) -> dict | None:
    """O placar antes e agora, e quem regrediu ou recuperou.

    A comparação cobre só os casos com resultado antes **e** agora: comparar
    contagens de conjuntos diferentes de casos daria um número que parece medir
    regressão e não mede. `None` quando não há nada comparável.

    O vermelho esperado fica fora das duas listas. Ele passar não é recuperação —
    é o defeito documentado que deixou de aparecer, e isso pede outra leitura
    (o sistema foi consertado, ou o caso deixou de exercitar o que dizia?). Ele
    voltar a reprovar também não é regressão. As duas mudanças saem à parte.
    """
    comparaveis = [cid for cid in atual if cid in anterior]
    if not comparaveis:
        return None
    mudou = [cid for cid in comparaveis if anterior[cid] != atual[cid]]
    return {
        "total": len(comparaveis),
        "antes": sum(1 for cid in comparaveis if anterior[cid]),
        "agora": sum(1 for cid in comparaveis if atual[cid]),
        "regressoes": [cid for cid in mudou if not atual[cid] and cid not in vermelhos_esperados],
        "recuperacoes": [cid for cid in mudou if atual[cid] and cid not in vermelhos_esperados],
        "vermelho_esperado_mudou": [cid for cid in mudou if cid in vermelhos_esperados],
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
    if delta["vermelho_esperado_mudou"]:
        print(f"  vermelho esperado mudou: {', '.join(delta['vermelho_esperado_mudou'])}")
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


def avaliar_caso(grafo, caso: dict, execucoes: dict | None = None) -> dict:
    """Executa um caso e aplica os critérios. Devolve o resultado do caso.

    Um caso pode declarar `mesma_execucao_de`: em vez de invocar o grafo de novo,
    ele reaproveita a execução de outro caso desta rodada e só aplica os próprios
    critérios. É o que permite comparar dois critérios sobre a **mesma** resposta
    — sem isso, a diferença de veredito poderia vir da resposta ter mudado, e o
    que se quer isolar é a redação do critério. Rodado sozinho, o caso executa.

    Execução que falhou por infraestrutura é refeita (`NOVAS_TENTATIVAS`) e, se
    continuar falhando, o caso volta com `infraestrutura` preenchido e nenhum
    veredito: não mediu o sistema, então não aprova nem reprova.

    Erro na invocação vira veredito reprovado: uma exceção não pode passar por
    caso aprovado, nem derrubar a suíte inteira. A exceção é, ao contrário,
    infraestrutura quando uma chamada ao modelo falhou antes dela. E o juiz que
    não responde depois das novas tentativas também deixa o caso sem medida.
    """
    criterios = caso.get("criterios", {})
    execucoes = execucoes if execucoes is not None else {}
    resultado = {"caso": caso, "vereditos": [], "nao_avaliados": [], "estado": None, "infraestrutura": None}
    origem = caso.get("mesma_execucao_de")
    reaproveitado = bool(origem and origem in execucoes)
    if reaproveitado:
        estado, sonda, trace_id, infra = execucoes[origem]
    else:
        esperas = iter(NOVAS_TENTATIVAS)
        while True:
            sonda = SondaDaExecucao()
            try:
                estado, trace_id = executar(grafo, caso, sonda)
            except Exception as erro:
                # Exceção com chamada ao modelo falhando antes dela é o provedor
                # fora do ar (rede, tempo esgotado), não defeito do sistema.
                if not sonda.erros:
                    resultado["vereditos"] = [regua.Veredito("execucao", False, f"{type(erro).__name__}: {erro}")]
                    return resultado
                estado, trace_id = None, None
                infra = f"{type(erro).__name__}: {erro}"
            else:
                infra = falha_de_infraestrutura(estado, sonda)
            espera = next(esperas, None) if infra else None
            if espera is None:
                break
            time.sleep(espera)
    execucoes[caso["id"]] = (estado, sonda, trace_id, infra)
    resultado["estado"] = estado
    if infra:
        resultado["infraestrutura"] = infra
        return resultado
    vereditos = regua.avaliar(estado, sonda, criterios)
    # O juiz roda depois da régua, mas não recebe o resultado dela: saber que os
    # critérios em código passaram o inclinaria a concordar com eles.
    if juiz.CRITERIO in criterios:
        veredito_do_juiz = julgar_com_novas_tentativas(caso, estado, criterios[juiz.CRITERIO])
        if not veredito_do_juiz.avaliado:
            # Sem o veredito do juiz o caso não foi medido: contar os outros
            # critérios como o caso inteiro daria um verde que ninguém mediu.
            resultado["infraestrutura"] = veredito_do_juiz.detalhe
            return resultado
        vereditos.append(veredito_do_juiz)
    resultado["vereditos"] = vereditos
    resultado["nao_avaliados"] = [nome for nome in criterios if nome not in RECONHECIDOS]
    plataforma.enviar_scores(trace_id, caso["id"], vereditos, reaproveitado)
    return resultado


def julgar_com_novas_tentativas(caso: dict, estado: dict, criterio: str) -> regua.Veredito:
    """Chama o juiz, refazendo com as mesmas esperas da execução se ele falhar."""
    esperas = iter(NOVAS_TENTATIVAS)
    while True:
        veredito = juiz.avaliar(caso, estado, criterio)
        espera = None if veredito.avaliado else next(esperas, None)
        if espera is None:
            return veredito
        time.sleep(espera)


def passou_caso(vereditos: list[regua.Veredito]) -> bool:
    """Um caso passa quando mediu alguma coisa e todo critério medido foi aprovado.

    Sem nenhum critério medido não há o que aprovar: `all()` de uma lista vazia
    é verdadeiro, e seria um verde que ninguém mediu.
    """
    medidos = [v for v in vereditos if v.avaliado]
    return bool(medidos) and all(v.ok for v in medidos)


def rotular(caso: dict, passou: bool) -> tuple[str, bool]:
    """O rótulo do resultado e se ele é **notícia** — diferente do declarado.

    Um caso pode declarar `esperado_vermelho`: ele documenta um defeito
    conhecido e reprova de propósito. Reprovar não é novidade nesse caso; a
    novidade seria passar, porque o defeito descrito teria deixado de existir e
    o caso não documentaria mais o que diz documentar.

    É essa distinção que dá sentido ao código de saída. Sem ela, uma suíte com
    vermelho documentado sai com 1 em toda execução, e o portão é desligado na
    primeira semana.
    """
    esperado_vermelho = caso.get("esperado_vermelho", False)
    if passou and esperado_vermelho:
        return "PASSOU (esperava vermelho)", True
    if passou:
        return "passou", False
    if esperado_vermelho:
        return "falhou (esperado)", False
    return "FALHOU", True


# --- Saída -------------------------------------------------------------------


def _cabecalho(nome: str, veredito: str, cor: str = "negrito") -> str:
    """Nome do caso com preenchimento pontilhado até a coluna do veredito."""
    return f"{nome} {pintar('.' * max(3, COLUNA - len(nome)), 'cinza')} {pintar(veredito, cor)}"


def _marca(v: regua.Veredito) -> str:
    if not v.avaliado:
        return pintar("·", "cinza")
    return pintar("✓", "verde") if v.ok else pintar("✗", "vermelho")


def situacao_do_caso(caso: dict, passou: bool) -> str:
    """A situação do caso no relatório: passou, falhou, esperado ou surpresa."""
    if caso.get("esperado_vermelho"):
        return "surpresa" if passou else "esperado"
    return "passou" if passou else "falhou"


COR_DA_SITUACAO = {"passou": "verde", "falhou": "vermelho", "esperado": "cinza", "surpresa": "vermelho"}


def imprimir_caso(resultado: dict) -> None:
    caso, estado = resultado["caso"], resultado["estado"]
    print()
    if resultado["infraestrutura"]:
        print(_cabecalho(caso["id"], "não avaliado (infraestrutura)", "cinza"))
        print(f"  · {'motivo':<14}{resultado['infraestrutura'][:160]}")
        return
    passou = passou_caso(resultado["vereditos"])
    rotulo, _ = rotular(caso, passou)
    print(_cabecalho(caso["id"], rotulo, COR_DA_SITUACAO[situacao_do_caso(caso, passou)]))
    for v in resultado["vereditos"]:
        print(f"  {_marca(v)} {v.criterio:<14}{v.detalhe}")
    for nome in resultado["nao_avaliados"]:
        print(f"  · {nome:<14}(nenhum critério com esse nome — não avaliado)")
    if estado is not None:
        print(f"  · {'caminho':<14}{' → '.join(estado.get('trajetoria') or [])}")
        # O groundedness é sempre mostrado e nunca decide: é um cosseno, cego à
        # negação, e dá nota baixa justamente à recusa correta.
        g = estado.get("groundedness")
        print(f"  · {'groundedness':<14}{'—' if g is None else g}  (reportado, nunca decide)")


def imprimir_repeticoes(caso: dict, rodadas: list[list[regua.Veredito]], nao_avaliados, sem_medida: int = 0) -> None:
    """O resultado agregado de um caso executado N vezes.

    Relata todo critério que reprovou em alguma rodada, separando o que reprova
    **sempre** do que **oscilou** — um critério que muda de veredito com a mesma
    entrada é informação, não ruído a ser suprimido.
    """
    total = len(rodadas)
    passaram = sum(1 for vereditos in rodadas if passou_caso(vereditos))
    marca_esperado = " (vermelho esperado)" if caso.get("esperado_vermelho") else ""
    print()
    cor = "verde" if passaram == total else ("cinza" if caso.get("esperado_vermelho") else "vermelho")
    print(_cabecalho(caso["id"], f"{passaram}/{total} passaram{marca_esperado}", cor))
    if sem_medida:
        print(f"  · {'infra':<14}{sem_medida} rodada(s) sem medida, fora da conta")
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
    p.add_argument(
        "--ignorar-base", dest="ignorar_base", action="store_true",
        help="Mede mesmo com a base fora do estado esperado, só avisando.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dados_da_suite = carregar_dados()
    casos = selecionar(dados_da_suite["casos"], args.caso, args.rapido)
    if not casos:
        print("Nenhum caso corresponde à seleção.")
        return 1

    # Pré-condição: a base tem de estar no estado que os casos pressupõem.
    divergencias = verificar_base(dados_da_suite.get("base_esperada") or {})
    if divergencias and not args.ignorar_base:
        print("A base vetorial não está no estado que os casos pressupõem:")
        for d in divergencias:
            print(f"  {d}")
        print()
        print("Restaure a base antes de medir:")
        print("  documento ausente: devolva o .md a dados/base (git checkout, ou reenvie")
        print("  pelo painel da base de conhecimento) e reindexe:")
        print("    docker compose exec backend python -m scripts.indexar_base")
        print("  documento a mais: remova pelo painel da base de conhecimento")
        print()
        print("Ou rode com --ignorar-base para medir mesmo assim.")
        return 2
    if divergencias:
        print("AVISO: base fora do estado esperado, medindo mesmo assim:")
        for d in divergencias:
            print(f"  {d}")

    grafo = compilar_grafo()
    escopo = "conjunto rápido" if args.rapido else f"{len(casos)} casos"
    sufixo = f" × {args.repeticoes} repetições" if args.repeticoes > 1 else ""
    print(f"Suíte de avaliação — {escopo}{sufixo}")
    if plataforma.disponivel():
        print(f"LangFuse: traces e scores na sessão {plataforma.RODADA_ID}")
    relatorio = RelatorioAoVivo(casos, f"{escopo}{sufixo}")
    print(f"relatório ao vivo: backend/avaliacao/{RELATORIO.parent.name}/{RELATORIO.name}")
    resultados: dict[str, bool] = {}
    sem_medida: list[str] = []
    if args.repeticoes > 1:
        for caso in casos:
            rodadas, nao_avaliados, infra = [], [], 0
            relatorio.rodando(caso["id"])
            ultimo = None
            for _ in range(args.repeticoes):
                # Cada repetição executa de novo, inclusive o caso que
                # reaproveitaria outro: o que se mede aqui é a variação.
                r = avaliar_caso(grafo, {**caso, "mesma_execucao_de": None})
                if r["infraestrutura"]:
                    infra += 1
                else:
                    rodadas.append(r["vereditos"])
                    nao_avaliados = r["nao_avaliados"]
                    ultimo = r
                time.sleep(PAUSA_ENTRE_CASOS)
            imprimir_repeticoes(caso, rodadas, nao_avaliados, infra)
            if rodadas:
                todas = all(passou_caso(r) for r in rodadas)
                resultados[caso["id"]] = todas
                passaram = sum(1 for r in rodadas if passou_caso(r))
                relatorio.registrar(
                    caso["id"], situacao_do_caso(caso, todas), ultimo["vereditos"], ultimo["estado"],
                    rotulo=f"{passaram}/{len(rodadas)} passaram",
                )
            else:
                sem_medida.append(caso["id"])
                relatorio.registrar(caso["id"], "infra", infraestrutura="nenhuma rodada pôde ser medida")
    else:
        execucoes: dict = {}
        for caso in casos:
            relatorio.rodando(caso["id"])
            r = avaliar_caso(grafo, caso, execucoes)
            imprimir_caso(r)
            if r["infraestrutura"]:
                sem_medida.append(caso["id"])
                relatorio.registrar(caso["id"], "infra", estado=r["estado"], infraestrutura=r["infraestrutura"])
            else:
                passou = passou_caso(r["vereditos"])
                resultados[caso["id"]] = passou
                relatorio.registrar(caso["id"], situacao_do_caso(caso, passou), r["vereditos"], r["estado"])
                if caso.get("mesma_execucao_de") is None:
                    time.sleep(PAUSA_ENTRE_CASOS)

    por_id = {c["id"]: c for c in casos}
    reprovados = [cid for cid, ok in resultados.items() if not ok]
    vermelhos_esperados = [cid for cid in reprovados if por_id[cid].get("esperado_vermelho")]
    inesperados = [cid for cid in reprovados if cid not in vermelhos_esperados]
    # Notícia = resultado diferente do declarado no caso: o que era verde
    # reprovou, ou o vermelho documentado deixou de reprovar. É o que o código de
    # saída sinaliza.
    noticias = [cid for cid, ok in resultados.items() if rotular(por_id[cid], ok)[1]]
    passou_o_que_devia_falhar = [cid for cid in noticias if resultados[cid]]

    print()
    print("─" * (COLUNA + 12))
    print(f"{len(resultados) - len(reprovados)}/{len(resultados)} casos passaram")
    if vermelhos_esperados:
        print(f"vermelhos esperados: {', '.join(vermelhos_esperados)}")
    if inesperados:
        print(f"reprovados: {', '.join(inesperados)}")
    if passou_o_que_devia_falhar:
        print(f"passaram mas eram vermelho esperado: {', '.join(passou_o_que_devia_falhar)}")
    if sem_medida:
        print(f"não avaliados (infraestrutura): {', '.join(sem_medida)}")

    # Regressão se mede contra a rodada anterior, não contra um número absoluto.
    # Só a execução única alimenta a base: com repetições o critério é mais
    # estrito (todas têm de passar), e misturar os dois tornaria o delta
    # incomparável.
    anterior = carregar_ultima()
    delta = calcular_delta(
        anterior, resultados, {cid for cid, c in por_id.items() if c.get("esperado_vermelho")}
    )
    imprimir_delta(delta)
    if args.repeticoes == 1:
        gravar_ultima(anterior, resultados)
    aviso = None
    if divergencias:
        aviso = "Base fora do estado esperado: " + "; ".join(divergencias)
    relatorio.concluir(delta if args.repeticoes == 1 else None, aviso)

    # Falha de envio é reportada e não muda o código de saída: a plataforma é
    # destino do resultado, não parte do critério.
    falhas_plataforma = plataforma.finalizar()
    if falhas_plataforma:
        print(f"LangFuse: {len(falhas_plataforma)} falha(s) de envio")
        for f in falhas_plataforma[:5]:
            print(f"  {f}")

    # 1: há notícia — o sistema mudou em relação ao que os casos declaram.
    # 2: nada mudou no que foi medido, mas parte não pôde ser medida; um portão
    #    de integração não deve deixar passar uma rodada que não mediu tudo.
    # 0: tudo como declarado.
    if noticias:
        return 1
    return 2 if sem_medida else 0


if __name__ == "__main__":
    sys.exit(main())
