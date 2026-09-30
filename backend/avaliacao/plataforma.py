# -*- coding: utf-8 -*-
"""Os resultados da suíte na plataforma de observabilidade (LangFuse).

Cada caso já produz uma trace, porque o callback do LangFuse entra no config do
`invoke()` — o mesmo mecanismo que a aplicação usa. Este módulo acrescenta a essa
trace os vereditos como **scores nomeados** ("regua" e "juiz", valor 1 ou 0,
com o motivo no comentário): o resultado da avaliação fica ao lado da execução
que o produziu, e não só no terminal de quem rodou.

Best-effort em todos os pontos: sem as chaves no ambiente tudo vira no-op e a
suíte roda igual; falha de envio é contada e não altera veredito nenhum. A
plataforma é destino do resultado, nunca parte do critério.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from app import observabilidade

# Identificador desta rodada, usado como sessão no LangFuse: agrupa as traces de
# todos os casos de uma execução da suíte numa vista só.
RODADA_ID = f"avaliacao-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"

_falhas: list[str] = []


def disponivel() -> bool:
    """As chaves do LangFuse estão no ambiente."""
    return observabilidade.langfuse_configurado()


def config_do_caso(caso_id: str, callbacks: list) -> tuple[dict, object | None]:
    """O config do `invoke()` de um caso, com o handler do LangFuse quando houver.

    Um handler por caso, como a aplicação faz por pergunta: o id da trace fica no
    próprio handler (`last_trace_id`), e é por ele que o score encontra a trace
    certa. O nome da trace é o id do caso, e a sessão é a rodada — é o que
    permite achar um caso na interface sem procurar por horário.
    """
    handler = observabilidade.handler_langfuse()
    if handler is None:
        return {"callbacks": callbacks}, None
    return {
        "callbacks": [*callbacks, handler],
        "run_name": caso_id,
        "metadata": {
            "langfuse_trace_name": caso_id,
            "langfuse_session_id": RODADA_ID,
            "langfuse_tags": ["avaliacao"],
            "caso": caso_id,
            "rodada": RODADA_ID,
        },
    }, handler


def enviar_scores(trace_id: str | None, caso_id: str, vereditos: list, reaproveitado: bool = False) -> None:
    """Pendura os vereditos do caso na trace da execução.

    Dois scores: `regua`, que é 1 só se todos os critérios em código passaram, e
    `juiz`, com o motivo como comentário. O critério não medido não vira score —
    zero diria "reprovado", e não foi isso que aconteceu.

    Um caso que reaproveita a execução de outro manda o seu juiz para a mesma
    trace, com o id do caso no nome: são dois julgamentos da mesma resposta, e
    é justamente lado a lado que eles têm de aparecer.
    """
    if trace_id is None or not disponivel():
        return
    try:
        from langfuse import get_client

        cliente = get_client()
        medidos = [v for v in vereditos if v.avaliado]
        regua = [v for v in medidos if v.criterio != "juiz"]
        if regua and not reaproveitado:
            reprovados = [f"{v.criterio}: {v.detalhe}" for v in regua if not v.ok]
            cliente.create_score(
                trace_id=trace_id,
                name="regua",
                value=0 if reprovados else 1,
                comment="; ".join(reprovados) or "todos os critérios em código aprovados",
            )
        for v in medidos:
            if v.criterio == "juiz":
                cliente.create_score(
                    trace_id=trace_id,
                    name=f"juiz · {caso_id}" if reaproveitado else "juiz",
                    value=1 if v.ok else 0,
                    comment=v.detalhe,
                )
    except Exception as erro:
        _falhas.append(f"{caso_id}: {type(erro).__name__}: {erro}")


def finalizar() -> list[str]:
    """Esvazia a fila de envio do SDK e devolve as falhas acumuladas.

    O flush é obrigatório: o SDK envia em lote, e o processo terminaria antes.
    """
    if disponivel():
        try:
            from langfuse import get_client

            get_client().flush()
        except Exception as erro:
            _falhas.append(f"flush: {type(erro).__name__}: {erro}")
    return list(_falhas)
