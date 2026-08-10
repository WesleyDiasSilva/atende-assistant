# -*- coding: utf-8 -*-
"""As ferramentas que o atendente pode decidir chamar.

Cada função é uma capacidade a mais: sem elas o modelo só sabe o que está no
prompt. O texto da docstring não é documentação interna — é o que o modelo lê
para decidir se chama a ferramenta e com quais argumentos.
"""
from langchain_core.tools import tool

from app import dados


@tool
def consultar_status_pedido(numero_pedido: str) -> str:
    """Consulta a situação e o prazo de entrega de um pedido pelo número.

    Use sempre que o cliente perguntar onde está o pedido, quando ele chega,
    se já foi entregue ou o que veio nele. O número tem apenas dígitos.
    """
    pedido = dados.buscar_pedido(numero_pedido)

    # A ferramenta não levanta exceção quando não encontra: devolve uma frase
    # que o modelo entende e consegue transformar em resposta ao cliente.
    if pedido is None:
        return (
            f"Pedido {numero_pedido} nao encontrado. Confirme o numero com o "
            "cliente."
        )

    itens = ", ".join(pedido["itens"])
    situacao = pedido["situacao"]

    if situacao == "entregue":
        return (
            f"Pedido {pedido['numero']} de {pedido['cliente']}: entregue ha "
            f"{pedido['entregue_ha_dias']} dia(s). Itens: {itens}. "
            f"Valor: R$ {pedido['valor']:.2f}. O prazo para pedir troca e de "
            f"{pedido['prazo_de_troca_em_dias']} dias apos a entrega."
        )

    if situacao == "cancelado":
        return (
            f"Pedido {pedido['numero']} de {pedido['cliente']}: cancelado. "
            f"Itens: {itens}. Valor: R$ {pedido['valor']:.2f}."
        )

    return (
        f"Pedido {pedido['numero']} de {pedido['cliente']}: {situacao}, com "
        f"previsao de entrega em {pedido['previsao_em_dias']} dia(s). "
        f"Itens: {itens}. Valor: R$ {pedido['valor']:.2f}."
    )


# O conjunto é explícito e curto: o modelo só pode chamar o que está aqui, e
# quanto menor a lista, menor a chance de ele escolher a ferramenta errada.
FERRAMENTAS = [consultar_status_pedido]
FERRAMENTAS_POR_NOME = {ferramenta.name: ferramenta for ferramenta in FERRAMENTAS}
