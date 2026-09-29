# -*- coding: utf-8 -*-
"""Critério julgado por modelo, para o que não tem gabarito.

A régua determinística cobre tudo que o sistema registra em campo estruturado:
caminho, classificação, trechos, ferramentas, contadores. O que ela não alcança
é o texto da resposta — se ele afirmou um número que ninguém forneceu, se
enunciou uma regra que o contexto não traz. Aí entra o juiz.

Duas escolhas definem o formato:

- O veredito é **binário** ("aprovado" / "reprovado") com um motivo, nunca uma
  nota de 1 a 5. Ninguém sabe dizer o que separa um 3 de um 4 — nem o modelo —,
  e o número varia entre execuções sem nada ter mudado. Binário obriga a
  decidir; o motivo obriga a explicar, e é ele que se audita.
- O critério de cada caso é uma **pergunta fechada que descreve um defeito**. Se
  o defeito está presente, reprova; se não está, aprova.

O juiz recebe a pergunta, o contexto que estava disponível e a resposta. **Não**
recebe o resultado dos critérios em código: se soubesse que eles passaram,
tenderia a concordar com eles.
"""
from __future__ import annotations

from typing import Literal

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import REGIAO_AWS
from avaliacao.regua import Veredito

# Nome do critério, como aparece em casos.json.
CRITERIO = "juiz"

# O modelo do juiz é declarado aqui, e não lido da configuração do sistema,
# mesmo sendo hoje o mesmo modelo: o juiz avalia o sistema, e não pode trocar de
# modelo em silêncio quando o sistema avaliado troca. Usar um modelo diferente,
# e mais forte, é uma decisão de custo — e é nesta linha que ela se toma.
MODELO_JUIZ = "global.anthropic.claude-haiku-4-5-20251001-v1:0"


class _VereditoJuiz(BaseModel):
    """Saída estruturada do juiz: a decisão binária e o motivo."""

    veredito: Literal["aprovado", "reprovado"] = Field(
        description=(
            "'reprovado' se o defeito descrito no critério está presente na "
            "resposta; 'aprovado' se não está."
        )
    )
    motivo: str = Field(
        description=(
            "Uma frase dizendo o que na resposta sustenta o veredito, citando o "
            "trecho relevante."
        )
    )


SYSTEM_JUIZ = (
    "Você é o juiz de uma suíte de avaliação de um assistente de atendimento. "
    "Recebe uma pergunta, o contexto que estava disponível para respondê-la e a "
    "resposta que o assistente produziu.\n\n"
    "O critério é uma pergunta fechada que descreve um DEFEITO a procurar na "
    "resposta:\n"
    "- Se o defeito ESTÁ presente, o veredito é \"reprovado\".\n"
    "- Se o defeito NÃO está presente, o veredito é \"aprovado\".\n\n"
    "Julgue somente o critério enunciado. Não avalie estilo, tom, extensão, "
    "cordialidade nem qualquer outra coisa que o critério não pergunte — uma "
    "resposta pode ser seca ou incompleta e ainda assim estar aprovada no "
    "critério em questão.\n\n"
    "Não avalie COMPLETUDE e não sugira o que a resposta deveria ter dito. Não "
    "é defeito a resposta deixar de mencionar algo, ser mais curta do que você "
    "escreveria ou não oferecer alternativas — a menos que o critério pergunte "
    "exatamente isso. Você julga o que a resposta AFIRMA, não o que ela omite.\n\n"
    "Baseie-se apenas na pergunta, no contexto e na resposta fornecidos. Não "
    "use conhecimento externo e não suponha contexto que não foi mostrado.\n\n"
    "No campo motivo, diga em uma frase o que na resposta sustenta o veredito, "
    "citando o trecho relevante. Não repita o enunciado do critério."
)

_modelo = ChatBedrockConverse(
    model_id=MODELO_JUIZ,
    region_name=REGIAO_AWS,
    temperature=0.0,
    max_tokens=512,
).with_structured_output(_VereditoJuiz)


def montar_contexto(caso: dict, estado: dict) -> str:
    """O contexto que estava disponível ao assistente, conforme o caminho.

    Na pergunta composta são as duas respostas parciais que a junção recebeu; nas
    rotas de busca, os trechos recuperados; na rota conversacional, o histórico
    declarado no caso. Sem nenhum dos três, isso é dito explicitamente: o juiz
    precisa saber que a resposta não tinha material em que se apoiar.
    """
    if estado.get("resposta_do_pedido") or estado.get("resposta_da_regra"):
        return (
            "PARTE CONSULTADA NO SISTEMA (sobre o pedido):\n"
            f"{estado.get('resposta_do_pedido') or '(vazia)'}\n\n"
            "PARTE VINDA DA BASE DE CONHECIMENTO (sobre a regra):\n"
            f"{estado.get('resposta_da_regra') or '(vazia)'}"
        )
    trechos = estado.get("trechos") or []
    if trechos:
        return "\n\n---\n\n".join(f"[{arquivo}]\n{conteudo}" for arquivo, conteudo in trechos)
    historico = caso.get("historico") or []
    if historico:
        return "\n".join(f"{m['papel']}: {m['texto']}" for m in historico)
    return "(nenhum contexto: a resposta não se apoiou em documento nem em conversa)"


def avaliar(caso: dict, estado: dict, criterio: str) -> Veredito:
    """Submete a resposta ao juiz e devolve o veredito no formato da régua.

    Falha na chamada (rede, cota, limite de requisições) marca o critério como
    **não avaliado**, e não como reprovado: a suíte não pode acusar defeito no
    sistema por causa de um problema do avaliador.
    """
    atendimento = estado.get("atendimento")
    if atendimento is None:
        return Veredito(CRITERIO, False, "execução sem resposta para julgar")
    conteudo = (
        f"Critério: {criterio}\n\n"
        f"Pergunta: {caso['pergunta']}\n\n"
        f"Contexto disponível:\n{montar_contexto(caso, estado)}\n\n"
        f"Resposta do assistente:\n{atendimento.resposta.resposta}"
    )
    try:
        julgado = _modelo.invoke(
            [SystemMessage(content=SYSTEM_JUIZ), HumanMessage(content=conteudo)]
        )
    except Exception as erro:
        return Veredito(
            CRITERIO, False, f"juiz indisponível ({type(erro).__name__}: {erro})", avaliado=False
        )
    return Veredito(CRITERIO, julgado.veredito == "aprovado", julgado.motivo.strip())
