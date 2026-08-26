# -*- coding: utf-8 -*-
"""O contrato da resposta do atendente.

Descreve os campos que a resposta sempre tem. Quem consome deixa de receber um
texto solto e passa a receber um objeto com posições fixas — o que permite ao
sistema usar a resposta sem procurar informação no meio da frase.

São dois contratos, e a fronteira entre eles importa: `RespostaAtendimento` é o
que o **modelo** preenche, `Atendimento` é o que o **nosso código** observou em
volta da chamada.
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


class Atendimento(BaseModel):
    """A resposta do modelo, mais o que o nosso código mediu em volta dela.

    Fica separado de `RespostaAtendimento` porque a origem do dado é diferente:
    lá são campos que o modelo preenche, aqui é observação do código. Contagem
    de tokens não é opinião do modelo — é o número que o Bedrock informou. Pedir
    que ele mesmo declarasse isso seria pedir que inventasse.

    O mesmo vale para as fontes, e é o que as torna auditáveis: quem sabe quais
    documentos entraram na conversa é a busca, não o modelo.
    """

    resposta: RespostaAtendimento
    tokens_de_entrada: int = 0
    # Os documentos que a busca recuperou, na ordem em que ela os ranqueou. Fica
    # vazio nos outros dois modos: sem busca não houve seleção, e citar a base
    # inteira como "fonte" não informaria nada.
    fontes: list[str] = []
    # Quantas vezes o fluxo ampliou a busca antes de chegar nesta resposta. Zero
    # é o caminho normal — a primeira passada bastou. Também é observação do
    # código, e não do modelo: quem contou as voltas foi o grafo.
    tentativas: int = 0
