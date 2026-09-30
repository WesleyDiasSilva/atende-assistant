# -*- coding: utf-8 -*-
"""O que a suíte observa da execução sem mexer nela.

O estado final do grafo não guarda tudo. O ramo do pedido, na pergunta composta,
consulta a ferramenta com mensagens locais e devolve só o texto da resposta:
qual ferramenta rodou, e com que argumento, não fica em campo nenhum.

Acrescentar esse campo ao estado seria alterar o sistema para poder medi-lo — e
aí o que se mede já não é o sistema de antes. A saída é a mesma porta que o
tracing usa: um **callback** no config do `invoke()`. O config se propaga para
dentro de cada node, inclusive dos que rodam em paralelo, e o callback recebe o
aviso de cada ferramenta que começa e termina, com nome, argumentos e resultado.

O mesmo callback resolve um segundo problema: o fluxo engole a falha do Bedrock
e devolve uma resposta normal com o texto do erro — é o que ele deve fazer com o
cliente. Para a suíte isso é um perigo: um limite de requisições no meio da
rodada viraria "FALHOU" e sujaria o delta com uma regressão que o sistema não
teve. O callback é avisado de toda chamada ao modelo que falhou, mesmo as que o
fluxo tratou, e a suíte separa o caso como não avaliado.
"""
from __future__ import annotations

import ast

from langchain_core.callbacks import BaseCallbackHandler


class SondaDaExecucao(BaseCallbackHandler):
    """Registra toda ferramenta executada durante um `invoke()`.

    Uma sonda por execução: ela acumula em si mesma, e reaproveitá-la entre
    casos misturaria o que um caso fez com o que o outro fez.
    """

    def __init__(self) -> None:
        self.ferramentas: list[dict] = []
        self.erros: list[str] = []
        self._em_andamento: dict = {}

    def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kwargs):
        argumentos = inputs if isinstance(inputs, dict) else _ler_argumentos(input_str)
        registro = {
            "nome": (serialized or {}).get("name") or kwargs.get("name"),
            "argumentos": argumentos,
            "saida": None,
        }
        self._em_andamento[run_id] = registro
        self.ferramentas.append(registro)

    def on_tool_end(self, output, *, run_id, **kwargs):
        registro = self._em_andamento.pop(run_id, None)
        if registro is not None:
            registro["saida"] = str(getattr(output, "content", output))

    def on_llm_error(self, error, **kwargs):
        self.erros.append(f"{type(error).__name__}: {error}")


def falha_de_infraestrutura(estado: dict | None, sonda: SondaDaExecucao) -> str | None:
    """Por que esta execução não mediu o sistema — ou `None`, se mediu.

    Duas causas: uma chamada ao modelo falhou durante o caso (limite de
    requisições, rede, credencial), ou a busca não devolveu trecho nenhum num
    modo de busca. A base conhecida tem dezenas de trechos e a busca devolve
    sempre `k` deles: lista vazia ali é banco fora do ar, não resposta do sistema.
    """
    if sonda.erros:
        return sonda.erros[0]
    if estado is None:
        return None
    buscou = "recuperar" in (estado.get("trajetoria") or []) or "ramo_da_regra" in (
        estado.get("trajetoria") or []
    )
    if buscou and estado.get("modo") in ("rag", "rag_gerenciado") and not estado.get("trechos"):
        return "a busca não devolveu trecho nenhum: base fora do ar ou vazia"
    return None


def _ler_argumentos(texto: str) -> dict:
    """Os argumentos da ferramenta quando o callback só entrega o texto deles."""
    try:
        valor = ast.literal_eval(texto)
    except (ValueError, SyntaxError):
        return {"entrada": texto}
    return valor if isinstance(valor, dict) else {"entrada": valor}
