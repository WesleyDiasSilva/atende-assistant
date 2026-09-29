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


def _ler_argumentos(texto: str) -> dict:
    """Os argumentos da ferramenta quando o callback só entrega o texto deles."""
    try:
        valor = ast.literal_eval(texto)
    except (ValueError, SyntaxError):
        return {"entrada": texto}
    return valor if isinstance(valor, dict) else {"entrada": valor}
