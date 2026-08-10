# -*- coding: utf-8 -*-
"""O contrato da resposta do atendente.

Descreve os campos que a resposta sempre tem. Quem consome deixa de receber um
texto solto e passa a receber um objeto com posições fixas — o que permite ao
sistema usar a resposta sem procurar informação no meio da frase.
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TipoDeAtendimento(str, Enum):
    """Os assuntos que o atendimento reconhece.

    Conjunto fechado de propósito: o modelo escolhe um destes e não pode
    inventar um valor novo, o que mantém a contagem por assunto comparável ao
    longo do tempo. Herda de str para serializar como texto no JSON.
    """

    STATUS_PEDIDO = "status_pedido"
    TROCA = "troca"
    DUVIDA_PRODUTO = "duvida_produto"
    OUTRO = "outro"


class RespostaAtendimento(BaseModel):
    """Formato que o modelo é obrigado a preencher.

    As descrições dos campos são lidas pelo modelo como instrução — não são
    comentários para quem mantém o código.
    """

    resposta: str = Field(
        description="A resposta ao cliente, no tom do perfil de atendimento."
    )
    tipo: TipoDeAtendimento = Field(
        description=(
            "O assunto da mensagem do cliente. Use status_pedido quando ele "
            "pergunta onde está ou quando chega um pedido; troca quando quer "
            "trocar, devolver ou reclamar de um produto recebido; "
            "duvida_produto quando pergunta sobre o produto, conservação, "
            "prazo de validade ou preparo; outro para o restante."
        )
    )
    precisa_de_humano: bool = Field(
        default=False,
        description=(
            "Verdadeiro quando o caso deve ser encaminhado a um atendente "
            "humano: pedido de exceção à política, reclamação grave, cobrança "
            "indevida, ou quando você não tem a informação necessária para "
            "resolver. Falso quando a sua resposta já resolve o caso."
        ),
    )
    motivo: Optional[str] = Field(
        default=None,
        description=(
            "Quando precisa_de_humano for verdadeiro, uma frase curta dizendo "
            "o que o atendente precisa resolver. Vazio nos demais casos."
        ),
    )
