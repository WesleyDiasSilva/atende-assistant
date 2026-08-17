# -*- coding: utf-8 -*-
"""API do atendente de pedidos.

    GET  /api/saude         estado da API
    GET  /api/configuracao  modelos, perfis, modos de conhecimento e temperatura
    POST /api/responder     recebe a pergunta e devolve a resposta em campos
    GET  /api/metricas      contagem de atendimentos por assunto
    GET  /api/solicitacoes  fila de trabalho humano
    POST /api/solicitacoes/{protocolo}/resolver   fecha um item da fila
    GET  /api/base/documentos   os documentos da base de conhecimento

Responder faz três coisas: pede a resposta ao atendente, registra o
atendimento para a contagem por assunto e, quando o caso precisa de uma
pessoa, coloca-o na fila.
"""
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import assistente, dados, db, documentos, log
from app.schemas import TipoDeAtendimento
from app.config import (
    MODELOS,
    MODO_PADRAO,
    MODOS_DE_CONHECIMENTO,
    PERFIS_DE_ATENDIMENTO,
    TEMPERATURA_MAXIMA,
    TEMPERATURA_MINIMA,
    TEMPERATURA_PADRAO,
    TEMPERATURA_PASSO,
)

# Lê o .env da raiz do projeto, onde fica a chave do Bedrock.
load_dotenv()

# Liga o log da aplicação: é nele que aparecem a execução de cada ferramenta, as
# etapas da indexação e o ranking de cada busca.
log.configurar()


@asynccontextmanager
async def ciclo_de_vida(_: FastAPI):
    """Roda uma vez, quando a API sobe.

    Tocar o banco no boot é de propósito: é o último momento em que ainda dá
    para ver no log que a conexão não funciona, antes de a primeira pergunta
    falhar na frente de alguém.
    """
    db.garantir_extensao_vector()
    yield


app = FastAPI(title="Atendente de Pedidos", lifespan=ciclo_de_vida)

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
    modo: str = MODO_PADRAO
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
        "modos": [
            {"id": chave, "nome": modo["nome"], "descricao": modo["descricao"]}
            for chave, modo in MODOS_DE_CONHECIMENTO.items()
        ],
        "modo_padrao": MODO_PADRAO,
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
    if pergunta_do_cliente.modo not in MODOS_DE_CONHECIMENTO:
        raise HTTPException(status_code=400, detail="Modo de conhecimento desconhecido.")

    atendimento = assistente.responder(
        pergunta_do_cliente.pergunta,
        pergunta_do_cliente.perfil,
        pergunta_do_cliente.modelo,
        pergunta_do_cliente.temperatura,
        pergunta_do_cliente.modo,
    )
    resposta = atendimento.resposta
    # O campo `tipo` da resposta é o que torna a contagem por assunto possível:
    # a classificação chega junto com o texto, sem uma segunda chamada ao modelo.
    dados.registrar_atendimento(
        tipo=resposta.tipo.value,
        pergunta=pergunta_do_cliente.pergunta,
        modelo=pergunta_do_cliente.modelo,
    )

    # O encaminhamento é uma decisão do modelo, mas quem resolve é uma pessoa:
    # o caso entra na fila e fica aberto até alguém fechá-lo.
    if resposta.precisa_de_humano:
        dados.registrar_solicitacao(
            origem="encaminhamento",
            assunto=resposta.tipo.value,
            motivo=resposta.motivo or "Sem motivo informado.",
            pergunta=pergunta_do_cliente.pergunta,
        )

    # O retorno é uma instância do schema, não texto: a API devolve os campos e
    # a interface lê cada um pelo nome, sem procurar informação dentro da frase.
    #
    # Os dois contratos saem achatados num JSON só. A interface não precisa saber
    # que um campo veio do modelo e o outro da medição — mas o código precisa, e
    # é por isso que eles são separados até aqui.
    return {
        **resposta.model_dump(),
        "tokens_de_entrada": atendimento.tokens_de_entrada,
    }


@app.get("/api/solicitacoes")
def solicitacoes():
    """A fila de trabalho humano, com as duas origens na mesma lista."""
    itens = dados.listar_solicitacoes()
    return {
        "itens": itens,
        "abertas": sum(1 for item in itens if item.get("situacao") == "aberta"),
    }


@app.post("/api/solicitacoes/{protocolo}/resolver")
def resolver(protocolo: str):
    """Fecha um item da fila. É a ponta humana do fluxo."""
    solicitacao = dados.resolver_solicitacao(protocolo)
    if solicitacao is None:
        raise HTTPException(status_code=404, detail="Solicitação não encontrada.")
    return solicitacao


@app.get("/api/base/documentos")
def base_documentos():
    """Os documentos da base de conhecimento, com arquivo e título.

    A base é uma pasta de `.md` no disco. Esta rota só a mostra: nada aqui ainda
    chega ao atendente.
    """
    itens = documentos.listar()
    return {"itens": itens, "total": len(itens)}


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
