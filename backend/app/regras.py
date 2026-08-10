# -*- coding: utf-8 -*-
"""As regras de negócio do atendimento.

Ficam separadas das ferramentas de propósito. O modelo decide *pedir* uma
ação; o que pode ou não acontecer é decisão deste código, e continua valendo
mesmo que o modelo mude, erre ou seja convencido pelo cliente a abrir uma
exceção.
"""
from typing import Optional

from app import dados

SITUACOES_QUE_NAO_PERMITEM_TROCA = {
    "em transporte": "o pedido ainda nao foi entregue",
    "em separacao": "o pedido ainda nao foi entregue",
    "cancelado": "o pedido foi cancelado",
}


def impedimento_para_troca(pedido: dict) -> Optional[str]:
    """Devolve o motivo pelo qual a troca não pode ser aberta, ou None.

    A frase de retorno vai para o modelo, então descreve a recusa em
    linguagem que ele consegue repassar ao cliente.
    """
    situacao = pedido.get("situacao")

    motivo_da_situacao = SITUACOES_QUE_NAO_PERMITEM_TROCA.get(situacao)
    if motivo_da_situacao:
        return (
            f"Nao e possivel abrir troca para o pedido {pedido['numero']}: "
            f"{motivo_da_situacao}."
        )

    dias_desde_a_entrega = pedido.get("entregue_ha_dias")
    prazo = pedido.get("prazo_de_troca_em_dias", 0)
    if dias_desde_a_entrega is not None and dias_desde_a_entrega > prazo:
        return (
            f"Nao e possivel abrir troca para o pedido {pedido['numero']}: a "
            f"entrega foi ha {dias_desde_a_entrega} dias e o prazo de troca e "
            f"de {prazo} dias. O caso precisa de um atendente humano para "
            "avaliar uma excecao."
        )

    if dados.solicitacao_aberta_para_pedido(pedido["numero"]):
        return (
            f"Ja existe uma solicitacao de troca em aberto para o pedido "
            f"{pedido['numero']}. Nao abra outra; informe o cliente de que o "
            "caso ja esta em andamento."
        )

    return None
