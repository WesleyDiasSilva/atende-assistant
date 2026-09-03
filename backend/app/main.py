# -*- coding: utf-8 -*-
"""API do atendente de pedidos.

    GET    /api/saude          estado da API
    GET    /api/configuracao   modelos, perfis, modos e faixa de temperatura
    POST   /api/responder      recebe a pergunta e devolve a resposta em campos
    GET    /api/metricas       contagem de atendimentos por assunto
    GET    /api/solicitacoes   fila de trabalho humano
    POST   /api/solicitacoes/{protocolo}/resolver  fecha um item da fila
    GET    /api/base/documentos            os documentos da base de conhecimento
    POST   /api/base/documentos            grava um .md e o indexa na hora
    DELETE /api/base/documentos/{arquivo}  remove o arquivo e os chunks dele

Responder faz três coisas: pede a resposta ao atendente, registra o
atendimento para a contagem por assunto e, quando o caso precisa de uma
pessoa, coloca-o na fila.
"""
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Lê o .env da raiz do projeto **antes** de importar os nossos módulos, e a ordem
# é obrigatória: `config.py` e `retrieval.py` leem `os.getenv` no corpo do módulo,
# ou seja, no momento do import. Importar primeiro e carregar o .env depois faria
# cada um deles ficar com o valor padrão para sempre — e trocar `EMBEDDING_MODEL`
# no .env passaria a não ter efeito nenhum, que é justamente a divergência entre
# indexação e busca que o retrieval.py existe para evitar.
load_dotenv()

from app import (  # noqa: E402  (depois do load_dotenv, de propósito)
    assistente,
    dados,
    db,
    documentos,
    grafo,
    log,
    memoria,
    retrieval,
    retrieval_gerenciado,
)
from app.config import (  # noqa: E402
    MODELOS,
    MODO_PADRAO,
    MODOS_DE_CONHECIMENTO,
    PERFIS_DE_ATENDIMENTO,
    TEMPERATURA_MAXIMA,
    TEMPERATURA_MINIMA,
    TEMPERATURA_PADRAO,
    TEMPERATURA_PASSO,
)
from app.schemas import TipoDeAtendimento  # noqa: E402

# Liga o log da aplicação: é nele que aparecem a execução de cada ferramenta, as
# etapas da indexação e o ranking de cada busca.
log.configurar()


# A conexão do checkpointer, guardada para ser fechada no encerramento. Fica em
# módulo porque o ciclo de vida é o único lugar que a abre e o único que a fecha.
_conexao_da_memoria = None


@asynccontextmanager
async def ciclo_de_vida(_: FastAPI):
    """Roda uma vez, quando a API sobe.

    Tocar o banco no boot é de propósito: é o último momento em que ainda dá
    para ver no log que a conexão não funciona, antes de a primeira pergunta
    falhar na frente de alguém.

    O grafo passou a ser compilado aqui, e não no import. A compilação precisa da
    conexão onde o estado será gravado, e essa conexão é aberta neste ponto — uma
    só, viva enquanto a API viver. Abrir uma por pergunta pagaria o handshake com
    o Postgres a cada turno, e o checkpointer serializa o acesso por conta
    própria, então uma basta.
    """
    db.garantir_extensao_vector()

    global _conexao_da_memoria
    _conexao_da_memoria, checkpointer = memoria.abrir_checkpointer()
    grafo.definir_grafo(grafo.compilar_grafo(checkpointer))

    yield

    if _conexao_da_memoria is not None:
        _conexao_da_memoria.close()


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
    # Liga o ciclo de auto-correção: quando a resposta admite não ter achado na
    # base, o fluxo amplia a busca e tenta uma segunda vez. Desligado por
    # padrão — o ciclo custa uma busca e duas idas ao modelo a mais, e quem
    # consome a API decide se vale.
    auto_corrigir: bool = False
    # Qual conversa é esta. Vira a chave sob a qual o estado do grafo é gravado,
    # e é o que separa o que uma pessoa disse do que outra disse. Quem gera o
    # identificador é quem conversa, não a API: o mesmo cliente mantendo a mesma
    # chave é o que faz o atendimento lembrar do turno anterior.
    #
    # Vazio é aceito de propósito — chamada solta à API não é conversa, e recebe
    # uma chave descartável.
    conversa_id: str = ""


class DocumentoNovo(BaseModel):
    """Corpo do POST /api/base/documentos.

    O conteúdo vai como texto no JSON, e não como upload multipart, porque um
    `.md` é texto: quem envia lê o arquivo e manda o conteúdo, sem precisar de
    codificação binária nem de dependência a mais para interpretá-la.

    A extensão não é validada aqui por `pattern`: a mensagem que o Pydantic gera
    para um padrão que não bate é a expressão regular em inglês, e ela apareceria
    crua na tela. A conferência fica na rota, com uma frase que dá para ler.
    """

    arquivo: str = Field(min_length=1)
    conteudo: str = Field(min_length=1)


@app.get("/api/saude")
def saude():
    return {"status": "ok"}


MOTIVO_KB_AUSENTE = (
    "Defina KNOWLEDGE_BASE_ID no .env com o id de um Knowledge Base do Amazon "
    "Bedrock. Sem ele, use o modo de busca na base, que roda no pgvector local."
)


def _disponibilidade(modo: str) -> dict:
    """Se o modo pode ser usado agora, e por que não, quando não pode.

    Só o modo gerenciado tem como estar indisponível: os outros três dependem de
    código e de dados que vêm no próprio repositório.
    """
    if modo == "rag_gerenciado" and not retrieval_gerenciado.esta_configurado():
        return {"disponivel": False, "indisponivel_porque": MOTIVO_KB_AUSENTE}
    return {"disponivel": True, "indisponivel_porque": None}


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
        # Cada modo diz se está disponível nesta instalação. O gerenciado depende
        # de um recurso provisionado numa conta AWS: quem clona o repositório sem
        # isso vê o modo na tela, desabilitado e com o motivo — em vez de clicar e
        # receber um erro que não explica nada.
        "modos": [
            {
                "id": chave,
                "nome": modo["nome"],
                "descricao": modo["descricao"],
                **_disponibilidade(chave),
            }
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
    if not _disponibilidade(pergunta_do_cliente.modo)["disponivel"]:
        raise HTTPException(status_code=400, detail=MOTIVO_KB_AUSENTE)

    atendimento = assistente.responder(
        pergunta_do_cliente.pergunta,
        pergunta_do_cliente.perfil,
        pergunta_do_cliente.modelo,
        pergunta_do_cliente.temperatura,
        pergunta_do_cliente.modo,
        pergunta_do_cliente.auto_corrigir,
        pergunta_do_cliente.conversa_id,
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
        "fontes": atendimento.fontes,
        # Quantas vezes o fluxo ampliou a busca para chegar nesta resposta. Zero
        # no caminho normal; a interface só o mostra quando houve volta.
        "tentativas": atendimento.tentativas,
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
    """Os documentos da base, com quantos chunks cada um tem indexado.

    A pasta de `.md` e a base vetorial são duas coisas, e podem divergir: um
    documento recém-copiado para a pasta aparece aqui com zero chunks até alguém
    indexar. Mostrar as duas contagens lado a lado é o que torna essa diferença
    visível em vez de surpreendente.
    """
    chunks_por_arquivo = retrieval.contar_por_arquivo()
    itens = [
        {**item, "chunks": chunks_por_arquivo.get(item["arquivo"], 0)}
        for item in documentos.listar()
    ]
    return {
        "itens": itens,
        "total": len(itens),
        "chunks_indexados": sum(chunks_por_arquivo.values()),
    }


@app.post("/api/base/documentos")
def enviar_documento(documento: DocumentoNovo):
    """Grava um `.md` na base e o indexa na hora.

    Indexar aqui, e não por script, é o que faz um documento novo passar a valer
    na resposta seguinte. As etapas voltam como lista para quem enviou acompanhar
    o que aconteceu: recebido, partido em chunks, indexado.

    Remove os chunks antigos antes de indexar. Sem isso, uma versão nova mais
    curta que a anterior deixaria para trás os chunks das posições que já não
    existem, e a busca continuaria encontrando texto que foi apagado.
    """
    if not documento.arquivo.lower().endswith(".md"):
        raise HTTPException(
            status_code=400,
            detail="A base aceita apenas arquivos .md (Markdown).",
        )

    arquivo, titulo = documentos.gravar(documento.arquivo, documento.conteudo)
    retrieval.remover(arquivo)
    chunks = retrieval.indexar([(arquivo, titulo, documento.conteudo)])

    return {
        "arquivo": arquivo,
        "titulo": titulo,
        "chunks": chunks,
        "etapas": [
            f"documento recebido: {arquivo} ({len(documento.conteudo)} caracteres)",
            f"partido em {chunks} chunk(s) de até {retrieval.CHUNK_SIZE} caracteres",
            f"indexado com {retrieval.EMBEDDING_MODEL}",
        ],
    }


@app.delete("/api/base/documentos/{arquivo}")
def remover_documento(arquivo: str):
    """Apaga o `.md` e os chunks dele. As duas coisas, ou a base fica incoerente.

    Os vetores são apagados pelo nome que `documentos.remover` devolveu, e não
    pelo que veio na URL: é o mesmo nome que a indexação gravou no metadata.
    """
    nome = documentos.remover(arquivo)
    if nome is None:
        raise HTTPException(status_code=404, detail="Documento não encontrado.")

    return {"arquivo": nome, "chunks_removidos": retrieval.remover(nome)}


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
