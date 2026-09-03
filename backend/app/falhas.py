# -*- coding: utf-8 -*-
"""Interrupção controlada da execução, num node escolhido.

Gravar o estado a cada passo tem uma consequência que só aparece quando algo
quebra: o passo que falhou **não** leva embora o que os passos anteriores
produziram. O que já foi feito está gravado, e a execução pode continuar de lá
em vez de recomeçar.

Isso só é verificável se houver como provocar a falha. Provocá-la editando código
na hora não serve: o que se quer observar é o comportamento do código que está no
ar, e não de uma versão alterada para o teste.

Este módulo é essa alavanca. Quem chama a API pede a interrupção num node pelo
nome; o node consulta aqui e levanta.

A condição é de **um disparo só** — consumida na primeira consulta. É isso que
torna a retomada possível: a segunda passagem pelo mesmo node encontra a condição
já gasta e conclui. Uma condição permanente reproduziria o mesmo erro para
sempre, e não haveria o que retomar.

O armamento vive no processo, e não no banco: reiniciar a API o descarta. Um
interruptor de teste que sobrevive a restart é uma armadilha.
"""
import logging

logger = logging.getLogger(__name__)

# Os nodes que aceitam interrupção. Conjunto fechado de propósito: o nome chega
# no corpo da requisição, e nome livre viraria pedido de interrupção em qualquer
# coisa. São os três que fazem trabalho caro — busca e idas ao modelo —, que é
# onde interessa observar se a retomada refaz ou não refaz.
NODES_INTERROMPIVEIS = ("recuperar", "conversar", "formalizar")

# O que está armado agora. Um conjunto, e não um booleano, porque a interrupção é
# por node: dá para armar em `recuperar` e observar uma coisa, armar em
# `formalizar` e observar outra.
_armados: set[str] = set()


class ExecucaoInterrompida(RuntimeError):
    """A falha provocada.

    Carrega o nome do node onde a execução parou. Quem trata precisa desse nome
    para dizer de onde a retomada vai continuar — uma exceção genérica obrigaria
    a adivinhar isso pelo texto.
    """

    def __init__(self, node: str):
        super().__init__(f"Execucao interrompida em '{node}'.")
        self.node = node


def armar(node: str) -> bool:
    """Arma a interrupção para a próxima passagem por `node`. Falso se não vale."""
    if node not in NODES_INTERROMPIVEIS:
        return False
    _armados.add(node)
    logger.info("[falha] armada em %s, um disparo", node)
    return True


def interromper_se_armado(node: str) -> None:
    """Levanta uma vez, se estiver armado para este node.

    Consome o armamento **antes** de levantar, e a ordem importa: se o consumo
    viesse depois, ele nunca aconteceria — a exceção sai da função primeiro — e a
    retomada cairia no mesmo erro indefinidamente.
    """
    if node not in _armados:
        return
    _armados.discard(node)
    logger.warning(
        "[falha] disparando em %s; o que os passos anteriores gravaram continua la",
        node,
    )
    raise ExecucaoInterrompida(node)
