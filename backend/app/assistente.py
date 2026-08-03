# -*- coding: utf-8 -*-
"""A chain do atendente.

Aqui vai entrar a chain do LangChain: prompt | model | parser.
Por enquanto devolve um texto fixo, só para provar que a interface, a API e o
container conversam de ponta a ponta.
"""
from app.config import MODELOS, PERFIS_DE_ATENDIMENTO, TEMPERATURA_PADRAO


def responder(
    pergunta: str,
    perfil: str,
    modelo: str,
    temperatura: float = TEMPERATURA_PADRAO,
) -> str:
    """Eco dos três controles, para conferir que a interface chega até aqui."""
    return (
        "A chain ainda não foi implementada. "
        f"Recebi a pergunta “{pergunta}” "
        f"com o perfil “{PERFIS_DE_ATENDIMENTO[perfil]['nome']}”, "
        f"o modelo “{MODELOS[modelo]['nome']}” "
        f"e temperatura {temperatura}."
    )
