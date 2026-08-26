# -*- coding: utf-8 -*-
"""Configuração dos controles que o usuário troca na interface.

Fica separado da chain de propósito: aqui é só catálogo e faixa de valores,
sem nenhuma lógica de LangChain.
"""
import os

REGIAO_AWS = os.getenv("AWS_REGION", "us-east-1")

# Os identificadores abaixo são inference profiles do Bedrock, e o prefixo
# (global. ou us.) é obrigatório: o Bedrock recusa o ID nu com
# ValidationException. Os três foram verificados nesta conta — acordo aceito,
# autorização concedida e parâmetro `temperature` aceito.
MODELOS = {
    "rapido": {
        "nome": "Rápido e econômico",
        "model_id": os.getenv(
            "MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001-v1:0"
        ),
        "custo_relativo": "menor custo",
    },
    "equilibrado": {
        "nome": "Equilibrado",
        "model_id": "global.anthropic.claude-sonnet-4-6",
        "custo_relativo": "custo intermediário",
    },
    "capaz": {
        "nome": "Mais capaz",
        "model_id": "global.anthropic.claude-opus-4-6-v1",
        "custo_relativo": "maior custo",
    },
}

# A temperatura controla o quanto o modelo varia entre chamadas. Em 0 ele escolhe
# sempre o token mais provável, o que deixa a resposta estável; perto de 1 ele
# arrisca mais, e a mesma pergunta pode voltar diferente.
#
# Cuidado ao trocar de modelo: nos modelos da família 5 e no Opus 4.7/4.8 o
# parâmetro `temperature` foi removido da API e a chamada volta com erro 400.
TEMPERATURA_PADRAO = 0.0
TEMPERATURA_MINIMA = 0.0
TEMPERATURA_MAXIMA = 1.0
TEMPERATURA_PASSO = 0.1

# Quanto da base de conhecimento o atendente recebe junto com a pergunta, e por
# qual caminho. Ao contrário do perfil, isto muda *o que ele sabe* — e é o único
# controle da interface capaz de mudar o conteúdo de uma resposta.
#
# Os três primeiros modos são um dial de quantidade de contexto: nada, tudo, ou só
# o que a pergunta pede. O quarto muda outra coisa: a quantidade é a mesma do
# terceiro, mas quem faz a busca é um serviço gerenciado em vez do nosso código.
#
# Ficam ligáveis lado a lado de propósito. A mesma pergunta nos quatro modos
# mostra a diferença sem precisar de explicação — e as duas últimas colunas
# isolam exatamente uma variável: o retriever.
MODOS_DE_CONHECIMENTO = {
    "sem_conhecimento": {
        "nome": "Sem conhecimento",
        "descricao": "Só o prompt. O atendente não vê nenhum documento.",
    },
    "stuffing": {
        "nome": "Base inteira no prompt",
        "descricao": "Todos os documentos vão no contexto, a cada pergunta.",
    },
    "rag": {
        "nome": "Busca na base (RAG)",
        "descricao": "Só os trechos que a pergunta recupera vão no contexto.",
    },
    "rag_gerenciado": {
        "nome": "Busca gerenciada (Knowledge Base)",
        "descricao": "A mesma busca, feita por um serviço da AWS em vez de pelo nosso código.",
    },
}

MODO_PADRAO = "sem_conhecimento"

# O `TOP_K` da segunda tentativa, quando a auto-correção decide ampliar a busca.
# O `TOP_K` normal mora em `retrieval.py`, junto do resto da configuração da
# busca; este fica aqui porque não é um parâmetro do retriever: é uma decisão de
# fluxo — quanto a mais vale a pena olhar quando a primeira passada não achou.
#
# Ampliar não é grátis: mais trechos é mais token de entrada, e ruído a mais no
# contexto pode piorar a resposta em vez de melhorá-la. Dez contra quatro é a
# folga que ainda cabe no prompt desta base sem afogar o trecho certo.
TOP_K_AMPLIADO = int(os.getenv("TOP_K_AMPLIADO", "10"))

# Cada perfil injeta uma instrução de tom diferente na mensagem de sistema.
# Muda como o atendente escreve, não o que ele sabe.
PERFIS_DE_ATENDIMENTO = {
    "padrao": {
        "nome": "Padrão da marca",
        "instrucao_de_tom": (
            "Escreva com simpatia e cuidado. Cumprimente, reconheça o que a pessoa "
            "perguntou e ofereça ajuda adicional no fim. Use frases completas."
        ),
    },
    "objetivo": {
        "nome": "Objetivo",
        "instrucao_de_tom": (
            "Responda direto ao ponto, sem saudação e sem oferta de ajuda adicional. "
            "Uma ou duas frases, apenas o que foi perguntado."
        ),
    },
}
