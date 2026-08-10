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


@tool
def abrir_solicitacao_de_troca(numero_pedido: str, motivo: str) -> str:
    """Abre uma solicitação de troca para um pedido já entregue.

    Use quando o cliente pedir troca ou devolução de um produto que recebeu.
    O motivo deve descrever, numa frase, o problema relatado pelo cliente.
    Só chame esta ferramenta quando o cliente pedir a troca de fato — para
    dúvidas sobre a política, responda sem abrir nada.
    """
    pedido = dados.buscar_pedido(numero_pedido)

    if pedido is None:
        return (
            f"Pedido {numero_pedido} nao encontrado. Nenhuma solicitacao foi "
            "aberta."
        )

    solicitacao = dados.registrar_solicitacao(
        origem="troca",
        assunto="troca",
        motivo=motivo,
        pergunta=f"Troca do pedido {pedido['numero']} ({pedido['cliente']}).",
        numero_pedido=pedido["numero"],
    )

    return (
        f"Solicitacao de troca aberta para o pedido {pedido['numero']} sob o "
        f"protocolo {solicitacao['protocolo']}. Informe o protocolo ao cliente "
        "e avise que a equipe entrara em contato."
    )


# O conjunto é explícito e curto: o modelo só pode chamar o que está aqui, e
# quanto menor a lista, menor a chance de ele escolher a ferramenta errada.
# Note a diferença entre as duas: uma lê, a outra deixa um registro no sistema.
FERRAMENTAS = [consultar_status_pedido, abrir_solicitacao_de_troca]
FERRAMENTAS_POR_NOME = {ferramenta.name: ferramenta for ferramenta in FERRAMENTAS}
