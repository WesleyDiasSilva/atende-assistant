# -*- coding: utf-8 -*-
"""API do atendente de pedidos.

    GET  /api/saude        estado da API
    GET  /api/configuracao modelos, perfis e faixa de temperatura
    POST /api/responder    recebe a pergunta e devolve a resposta da chain
"""
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import assistente, dados
from app.schemas import TipoDeAtendimento
from app.config import (
    MODELOS,
    PERFIS_DE_ATENDIMENTO,
    TEMPERATURA_MAXIMA,
    TEMPERATURA_MINIMA,
    TEMPERATURA_PADRAO,
    TEMPERATURA_PASSO,
)

# Lê o .env da raiz do projeto, onde fica a chave do Bedrock.
load_dotenv()

app = FastAPI(title="Atendente de Pedidos")

# A interface roda em outra porta durante o desenvolvimento, então o navegador
# precisa de permissão explícita para chamar esta API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PerguntaDoCliente(BaseModel):
    """Corpo do POST /api/responder.

    O Field com ge/le rejeita temperatura fora da faixa antes de chegar na chain,
    devolvendo 422 em vez de deixar o Bedrock recusar.
    """

    pergunta: str = Field(min_length=1)
    perfil: str = "padrao"
    modelo: str = "rapido"
    temperatura: float = Field(
        default=TEMPERATURA_PADRAO, ge=TEMPERATURA_MINIMA, le=TEMPERATURA_MAXIMA
    )


@app.get("/api/saude")
def saude():
    return {"status": "ok"}


@app.get("/api/configuracao")
def configuracao():
    """Entrega o catálogo para a interface montar os controles.

    A interface não conhece nome de modelo nem de perfil: ela desenha o que vem
    daqui. Acrescentar um modelo no config.py já o faz aparecer na tela.
    """
    return {
        "modelos": [
            {
                "id": chave,
                "nome": modelo["nome"],
                "model_id": modelo["model_id"],
                "custo_relativo": modelo["custo_relativo"],
            }
            for chave, modelo in MODELOS.items()
        ],
        "perfis": [
            {"id": chave, "nome": perfil["nome"]}
            for chave, perfil in PERFIS_DE_ATENDIMENTO.items()
        ],
        "temperatura": {
            "padrao": TEMPERATURA_PADRAO,
            "minima": TEMPERATURA_MINIMA,
            "maxima": TEMPERATURA_MAXIMA,
            "passo": TEMPERATURA_PASSO,
        },
    }


@app.post("/api/responder")
def responder(pergunta_do_cliente: PerguntaDoCliente):
    if pergunta_do_cliente.perfil not in PERFIS_DE_ATENDIMENTO:
        raise HTTPException(status_code=400, detail="Perfil de atendimento desconhecido.")
    if pergunta_do_cliente.modelo not in MODELOS:
        raise HTTPException(status_code=400, detail="Modelo desconhecido.")

    resposta = assistente.responder(
        pergunta_do_cliente.pergunta,
        pergunta_do_cliente.perfil,
        pergunta_do_cliente.modelo,
        pergunta_do_cliente.temperatura,
    )
    # O campo `tipo` da resposta é o que torna a contagem por assunto possível:
    # a classificação chega junto com o texto, sem uma segunda chamada ao modelo.
    dados.registrar_atendimento(
        tipo=resposta.tipo.value,
        pergunta=pergunta_do_cliente.pergunta,
        modelo=pergunta_do_cliente.modelo,
    )

    # O retorno é uma instância do schema, não texto: a API devolve os campos e
    # a interface lê cada um pelo nome, sem procurar informação dentro da frase.
    return resposta.model_dump()


@app.get("/api/metricas")
def metricas():
    """Contagem de atendimentos por assunto, para o painel.

    Os tipos vêm do enum do schema, então um assunto novo aparece no painel
    assim que passa a existir no contrato — sem mexer aqui.
    """
    resumo = dados.contar_por_tipo([tipo.value for tipo in TipoDeAtendimento])
    return {
        "total": resumo["total"],
        "por_tipo": resumo["por_tipo"],
        "ultimos": dados.ultimos_atendimentos(),
    }
