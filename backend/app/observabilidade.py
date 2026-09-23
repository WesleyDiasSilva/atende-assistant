# -*- coding: utf-8 -*-
"""O LangFuse: a segunda plataforma de tracing, e a que precisa de código.

O LangSmith não aparece neste arquivo, e a ausência é a lição: ele é do mesmo
fabricante do LangChain, o SDK já vem junto do langchain-core e se liga sozinho
quando as variáveis `LANGSMITH_*` estão no ambiente. O LangFuse é de terceiros.
O LangChain não sabe que ele existe, e a única porta de entrada que sobra é a
que o LangChain oferece a qualquer um: um **callback** — um objeto que recebe o
aviso de "começou", "terminou" e "falhou" de cada etapa da execução.

O callback entra no `config` do `invoke()` do grafo, e não em cada chamada ao
modelo. O config se propaga: todo node, toda ida ao Bedrock e toda ferramenta
dentro daquela execução herdam o mesmo callback, e aparecem como uma árvore só,
na mesma trace.

**Fail-open.** Sem as duas chaves no ambiente, este módulo não cria nada e o
atendimento roda como antes. Observabilidade que derruba o atendimento quando o
fornecedor está fora do ar é pior do que não ter observabilidade.
"""
import logging
import os

logger = logging.getLogger(__name__)


def langfuse_configurado() -> bool:
    """As duas chaves do projeto no LangFuse estão no ambiente."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def handler_langfuse():
    """Um callback do LangFuse novo, ou `None` quando ele não está configurado.

    Um por pergunta, e não um global: o handler anota em si mesmo o id da última
    trace que gravou (`last_trace_id`). Com um só para a API inteira, duas
    perguntas simultâneas escreveriam no mesmo campo, e a nota de uma iria parar
    na trace da outra. Criar é barato — a conexão com o LangFuse é do cliente do
    SDK, que é um só por processo.

    O import fica aqui dentro de propósito: quem não configurou o LangFuse não
    paga nem o carregamento do SDK. E qualquer falha na criação vira aviso no
    log, não exceção — a pergunta segue sem tracing.

    O `CallbackHandler()` sem argumentos lê `LANGFUSE_PUBLIC_KEY`,
    `LANGFUSE_SECRET_KEY` e `LANGFUSE_HOST` do ambiente. É o SDK 4.x: no 2.x
    as chaves iam no construtor do handler, e o import era outro
    (`langfuse.callback`).
    """
    if not langfuse_configurado():
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception as erro:  # o tracing não pode derrubar o atendimento
        logger.warning("[observabilidade] LangFuse não pôde ser iniciado: %s", erro)
        return None


def config_observado(config: dict, **metadados) -> tuple[dict, object | None]:
    """Acrescenta ao config do grafo o callback e os metadados da trace.

    Recebe o config que já tem o `thread_id` e devolve outro, junto com o handler
    usado (ou `None`). Quando o LangFuse está ligado, o config ganha:

    - `callbacks`: o handler, que o config propaga para dentro de cada node.
    - `metadata` com prefixo `langfuse_`: o handler lê essas chaves como
      atributos da trace. `langfuse_session_id` agrupa as traces de uma mesma
      conversa na tela de Sessions — e o valor é o próprio `thread_id`, a mesma
      chave com que o checkpointer grava o estado. Uma conversa, uma chave, nos
      dois lugares.

    Os demais metadados (modo, modelo, perfil) vão para a trace como estão, e
    viram filtro na tela. O LangSmith também os recebe: o `metadata` do config é
    do LangChain, não do LangFuse, e qualquer tracer conectado o lê.
    """
    thread_id = config["configurable"]["thread_id"]
    metadata = {**metadados, "thread_id": thread_id}

    # O nome da execução raiz: é o que aparece na lista de traces das duas
    # plataformas, no lugar do genérico "LangGraph".
    novo = {**config, "run_name": "atendimento", "metadata": metadata}

    handler = handler_langfuse()
    if handler is None:
        return novo, None

    metadata["langfuse_session_id"] = thread_id
    # O LangFuse não usa o `run_name` como nome da trace: pede a chave própria.
    metadata["langfuse_trace_name"] = "atendimento"
    metadata["langfuse_tags"] = [f"{chave}:{valor}" for chave, valor in metadados.items()]
    return {**novo, "callbacks": [handler]}, handler


def logar_estado() -> None:
    """Uma linha no boot dizendo o que está ligado. Nunca imprime chave."""
    langsmith = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2")
    logger.info(
        "[observabilidade] LangSmith: %s · LangFuse: %s",
        "ligado" if (langsmith or "").lower() == "true" else "desligado",
        "ligado" if langfuse_configurado() else "desligado",
    )
