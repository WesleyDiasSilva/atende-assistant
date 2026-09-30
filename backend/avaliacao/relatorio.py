# -*- coding: utf-8 -*-
"""O relatório da rodada, em HTML, reescrito a cada caso.

O terminal é a saída que vale: é o que uma integração contínua lê, e é onde sai
o código de saída. Este relatório é a mesma rodada para quem precisa **ver**: um
cartão por caso, que fica verde ou vermelho quando o caso termina, o caminho que
a pergunta percorreu, o motivo do juiz por extenso e, no fim, o delta contra a
rodada anterior.

Arquivo estático, sem servidor e sem dependência: enquanto a rodada corre, a
página se recarrega sozinha a cada dois segundos e rola até o caso em execução;
na última escrita a recarga sai. Abrir no navegador é abrir o arquivo.

A escrita é atômica (arquivo temporário + `replace`): o navegador que recarrega
no meio de uma escrita lê a versão anterior inteira, nunca metade de uma nova.
"""
from __future__ import annotations

import html
import os
from datetime import datetime
from pathlib import Path

PASTA = Path(__file__).resolve().parent / "relatorio"
ARQUIVO = PASTA / "rodada.html"

# Ordem e aparência de cada situação de caso. `esperado` é o vermelho
# documentado: continua sendo vermelho, mas não é notícia, e por isso não grita.
SITUACOES = {
    "pendente": ("na fila", "pendente"),
    "rodando": ("rodando", "rodando"),
    "passou": ("passou", "passou"),
    "falhou": ("falhou", "falhou"),
    "esperado": ("falhou · esperado", "esperado"),
    "surpresa": ("passou · esperava vermelho", "falhou"),
    "infra": ("não avaliado · infraestrutura", "infra"),
}


def _e(texto) -> str:
    return html.escape(str(texto), quote=True)


class RelatorioAoVivo:
    """Acompanha uma rodada e reescreve o HTML a cada mudança."""

    def __init__(self, casos: list[dict], titulo: str) -> None:
        self.titulo = titulo
        self.inicio = datetime.now()
        self.ordem = [c["id"] for c in casos]
        self.casos = {c["id"]: c for c in casos}
        self.cartoes: dict[str, dict] = {cid: {"situacao": "pendente"} for cid in self.ordem}
        self.atual: str | None = None
        self.delta: dict | None = None
        self.final = False
        self.aviso: str | None = None
        PASTA.mkdir(parents=True, exist_ok=True)
        self._gravar()

    # --- Eventos da rodada ---------------------------------------------------

    def rodando(self, caso_id: str) -> None:
        self.atual = caso_id
        self.cartoes[caso_id] = {"situacao": "rodando"}
        self._gravar()

    def registrar(self, caso_id: str, situacao: str, vereditos=(), estado=None,
                  infraestrutura: str | None = None, rotulo: str | None = None) -> None:
        self.cartoes[caso_id] = {
            "situacao": situacao,
            "vereditos": list(vereditos),
            "estado": estado,
            "infraestrutura": infraestrutura,
            "rotulo": rotulo,
        }
        self._gravar()

    def concluir(self, delta: dict | None, aviso: str | None = None) -> None:
        self.atual = None
        self.delta = delta
        self.aviso = aviso
        self.final = True
        self._gravar()

    # --- Montagem ------------------------------------------------------------

    def _contagem(self) -> dict[str, int]:
        conta = {chave: 0 for chave in SITUACOES}
        for cartao in self.cartoes.values():
            conta[cartao["situacao"]] += 1
        return conta

    def _gravar(self) -> None:
        temporario = ARQUIVO.with_suffix(".tmp")
        temporario.write_text(self._html(), encoding="utf-8")
        os.replace(temporario, ARQUIVO)

    def _html(self) -> str:
        conta = self._contagem()
        medidos = conta["passou"] + conta["falhou"] + conta["esperado"] + conta["surpresa"]
        aprovados = conta["passou"] + conta["surpresa"]
        feitos = len(self.ordem) - conta["pendente"] - conta["rodando"]
        progresso = int(100 * feitos / len(self.ordem)) if self.ordem else 100
        recarga = "" if self.final else '<meta http-equiv="refresh" content="2">'
        estado_da_rodada = (
            "rodada concluída"
            if self.final
            else f"rodando {feitos + 1} de {len(self.ordem)}" if self.atual else "preparando"
        )
        cartoes = "\n".join(self._cartao(cid) for cid in self.ordem)
        return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
{recarga}
<title>Suíte de avaliação</title>
<style>{CSS}</style>
</head>
<body>
<header class="topo">
  <div>
    <div class="rotulo">Suíte de avaliação</div>
    <h1>{_e(self.titulo)}</h1>
    <div class="sub">{_e(estado_da_rodada)} · iniciada às {self.inicio:%H:%M:%S}</div>
  </div>
  <div class="placar">
    <div class="numero">{aprovados}<span>/{medidos}</span></div>
    <div class="legenda">casos passaram</div>
  </div>
</header>
<div class="barra"><div style="width:{progresso}%"></div></div>
<section class="contadores">
  {self._contador("passou", conta["passou"] + conta["surpresa"])}
  {self._contador("falhou", conta["falhou"])}
  {self._contador("esperado", conta["esperado"])}
  {self._contador("infra", conta["infra"])}
</section>
{self._painel_delta()}
<main class="grade">
{cartoes}
</main>
<script>
  var alvo = document.getElementById("atual") || document.getElementById("delta");
  if (alvo) alvo.scrollIntoView({{block: "center"}});
</script>
</body>
</html>
"""

    def _contador(self, chave: str, valor: int) -> str:
        nomes = {"passou": "passaram", "falhou": "falharam", "esperado": "vermelho esperado", "infra": "não avaliados"}
        return f'<div class="contador {chave}"><b>{valor}</b><span>{nomes[chave]}</span></div>'

    def _painel_delta(self) -> str:
        if not self.final:
            return ""
        partes = []
        if self.aviso:
            partes.append(f'<div class="aviso">{_e(self.aviso)}</div>')
        d = self.delta
        if d is None:
            partes.append('<div class="delta-num">sem rodada anterior para comparar</div>')
        else:
            regr = d["regressoes"]
            classe = "caiu" if regr else "estavel"
            partes.append(
                f'<div class="delta-num {classe}">{d["antes"]}/{d["total"]}'
                f' <span class="seta">→</span> {d["agora"]}/{d["total"]}</div>'
                f'<div class="delta-regr {classe}">{len(regr)} regress{"ão" if len(regr) == 1 else "ões"}</div>'
            )
            if regr:
                partes.append('<div class="nomes"><span>regrediram</span>' + "".join(
                    f"<code>{_e(c)}</code>" for c in regr) + "</div>")
            if d["recuperacoes"]:
                partes.append('<div class="nomes recuperou"><span>recuperaram</span>' + "".join(
                    f"<code>{_e(c)}</code>" for c in d["recuperacoes"]) + "</div>")
            if d.get("vermelho_esperado_mudou"):
                partes.append('<div class="nomes esperado-mudou"><span>vermelho esperado mudou</span>' + "".join(
                    f"<code>{_e(c)}</code>" for c in d["vermelho_esperado_mudou"]) + "</div>")
        return f'<section class="delta" id="delta"><div class="rotulo">Delta contra a rodada anterior</div>{"".join(partes)}</section>'

    def _cartao(self, caso_id: str) -> str:
        caso = self.casos[caso_id]
        cartao = self.cartoes[caso_id]
        situacao = cartao["situacao"]
        rotulo, classe = SITUACOES[situacao]
        rotulo = cartao.get("rotulo") or rotulo
        ancora = ' id="atual"' if caso_id == self.atual else ""
        corpo = [f'<p class="descricao">{_e(caso.get("descricao", ""))}</p>',
                 f'<p class="pergunta">“{_e(caso["pergunta"])}”</p>']
        estado = cartao.get("estado")
        if estado is not None and situacao not in ("pendente", "rodando"):
            corpo.append(self._caminho(caso, estado))
        if cartao.get("infraestrutura"):
            corpo.append(f'<div class="infra-motivo">{_e(cartao["infraestrutura"][:220])}</div>')
        for v in cartao.get("vereditos", []):
            corpo.append(self._criterio(v))
        if estado is not None and situacao not in ("pendente", "rodando", "infra"):
            corpo.append(self._groundedness(estado.get("groundedness")))
        if caso.get("esperado_vermelho") and situacao in ("esperado", "surpresa"):
            corpo.append(f'<div class="documentado"><b>vermelho documentado</b> {_e(caso.get("motivo_do_vermelho", ""))}</div>')
        return (
            f'<article class="cartao {classe}"{ancora}>'
            f'<div class="cabeca"><code>{_e(caso_id)}</code><span class="selo {classe}">{_e(rotulo)}</span></div>'
            + "".join(corpo) + "</article>"
        )

    def _caminho(self, caso: dict, estado: dict) -> str:
        esperados = set(caso.get("criterios", {}).get("rota") or [])
        chips = []
        for node in estado.get("trajetoria") or []:
            destaque = " esperado" if node in esperados else ""
            chips.append(f'<span class="chip{destaque}">{_e(node)}</span>')
        return '<div class="caminho">' + '<span class="liga">→</span>'.join(chips) + "</div>"

    def _criterio(self, v) -> str:
        if not v.avaliado:
            marca, classe = "·", "nao-medido"
        else:
            marca, classe = ("✓", "ok") if v.ok else ("✗", "nok")
        if v.criterio == "juiz":
            return (
                f'<div class="criterio juiz {classe}"><span class="marca">{marca}</span>'
                f'<span class="nome">juiz</span><blockquote>{_e(v.detalhe)}</blockquote></div>'
            )
        return (
            f'<div class="criterio {classe}"><span class="marca">{marca}</span>'
            f'<span class="nome">{_e(v.criterio)}</span><span class="detalhe">{_e(v.detalhe)}</span></div>'
        )

    def _groundedness(self, valor) -> str:
        if valor is None:
            return '<div class="ground"><span class="nome">groundedness</span><span class="valor">—</span><em>reportado, nunca decide</em></div>'
        largura = max(0, min(100, int(float(valor) * 100)))
        return (
            f'<div class="ground"><span class="nome">groundedness</span>'
            f'<span class="trilho"><span style="width:{largura}%"></span></span>'
            f'<span class="valor">{float(valor):.2f}</span><em>reportado, nunca decide</em></div>'
        )


CSS = """
:root { --magenta:#D0095F; --wash:#FBE5EF; --texto:#1A1A1A; --cinza:#6B6B6B; --fundo:#FFFFFF;
        --caixa:#F4F4F4; --verde:#1E7F4B; --verde-wash:#E5F4EC; --vermelho:#C0262D; --vermelho-wash:#FBE7E8; }
* { box-sizing:border-box; }
body { margin:0; background:var(--fundo); color:var(--texto); font-family:Arial, Helvetica, sans-serif; font-size:20px; }
.topo { display:flex; justify-content:space-between; align-items:flex-end; padding:36px 48px 20px; border-left:14px solid var(--magenta); }
.rotulo { font-size:15px; font-weight:700; letter-spacing:.24em; text-transform:uppercase; color:var(--magenta); }
h1 { margin:6px 0 4px; font-size:44px; font-weight:800; letter-spacing:-.02em; }
.sub { color:var(--cinza); font-size:18px; }
.placar { text-align:right; }
.placar .numero { font-size:92px; font-weight:800; line-height:1; color:var(--magenta); }
.placar .numero span { color:#B9B9B9; font-size:60px; }
.placar .legenda { color:var(--cinza); font-size:18px; }
.barra { height:8px; background:var(--caixa); margin:0 48px; }
.barra div { height:8px; background:var(--magenta); transition:width .3s; }
.contadores { display:flex; gap:16px; padding:20px 48px 8px; }
.contador { flex:1; background:var(--caixa); padding:14px 18px; display:flex; align-items:baseline; gap:12px; }
.contador b { font-size:40px; }
.contador span { color:var(--cinza); font-size:17px; }
.contador.passou b { color:var(--verde); } .contador.falhou b { color:var(--vermelho); }
.contador.esperado b { color:#8A5A5C; } .contador.infra b { color:var(--cinza); }
.delta { margin:20px 48px 8px; padding:26px 32px; background:var(--wash); border-left:10px solid var(--magenta); }
.delta-num { font-size:72px; font-weight:800; margin-top:6px; }
.delta-num .seta { color:var(--magenta); }
.delta-num.caiu, .delta-regr.caiu { color:var(--vermelho); }
.delta-regr { font-size:32px; font-weight:700; margin:4px 0 12px; }
.delta-regr.estavel { color:var(--verde); }
.nomes { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-top:8px; }
.nomes span { color:var(--cinza); font-size:17px; margin-right:6px; }
.nomes code { background:#fff; padding:6px 12px; font-size:19px; border:2px solid var(--vermelho); color:var(--vermelho); }
.nomes.recuperou code { border-color:var(--verde); color:var(--verde); }
.nomes.esperado-mudou code { border:2px dashed #8A5A5C; color:#8A5A5C; }
.aviso { background:#fff; padding:10px 14px; margin-bottom:12px; color:#8A5A00; border-left:6px solid #D8A300; }
.grade { display:grid; grid-template-columns:repeat(auto-fill, minmax(560px, 1fr)); gap:18px; padding:20px 48px 60px; }
.cartao { background:var(--caixa); padding:18px 22px; border-top:8px solid #D6D6D6; }
.cartao.passou { background:var(--verde-wash); border-top-color:var(--verde); }
.cartao.falhou { background:var(--vermelho-wash); border-top-color:var(--vermelho); }
.cartao.esperado { background:#F6EFEF; border-top:8px dashed #B07A7D; }
.cartao.infra { background:#F1F1F1; border-top:8px dotted #9A9A9A; }
.cartao.rodando { background:#fff; border-top-color:var(--magenta); outline:4px solid var(--magenta); animation:pulso 1.2s infinite; }
.cartao.pendente { opacity:.55; }
@keyframes pulso { 50% { outline-color:#F3A9C9; } }
.cabeca { display:flex; justify-content:space-between; align-items:center; gap:12px; }
.cabeca code { font-size:21px; font-weight:700; }
.selo { font-size:15px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; padding:5px 10px; background:#fff; white-space:nowrap; }
.selo.passou { color:var(--verde); } .selo.falhou { color:var(--vermelho); } .selo.esperado { color:#8A5A5C; }
.selo.rodando { color:#fff; background:var(--magenta); } .selo.pendente, .selo.infra { color:var(--cinza); }
.descricao { margin:10px 0 4px; color:#444; font-size:17px; line-height:1.35; }
.pergunta { margin:6px 0 10px; font-size:20px; font-style:italic; }
.caminho { display:flex; flex-wrap:wrap; align-items:center; gap:4px; margin:8px 0 12px; }
.chip { background:#fff; border:2px solid #CFCFCF; padding:4px 9px; font-size:16px; font-family:Menlo, Consolas, monospace; }
.chip.esperado { border-color:var(--magenta); color:var(--magenta); font-weight:700; }
.liga { color:#A0A0A0; font-size:15px; }
.criterio { display:grid; grid-template-columns:26px 150px 1fr; align-items:baseline; padding:5px 0; border-top:1px solid rgba(0,0,0,.07); font-size:18px; }
.criterio .marca { font-weight:800; }
.criterio.ok .marca { color:var(--verde); } .criterio.nok .marca { color:var(--vermelho); } .criterio.nao-medido .marca { color:var(--cinza); }
.criterio .nome { font-family:Menlo, Consolas, monospace; font-size:16px; color:#555; }
.criterio .detalhe { line-height:1.3; word-break:break-word; }
.criterio.juiz blockquote { margin:0; padding:8px 12px; background:#fff; border-left:5px solid var(--magenta); line-height:1.35; font-size:18px; }
.criterio.juiz.nok blockquote { border-left-color:var(--vermelho); }
.ground { display:flex; align-items:center; gap:12px; margin-top:10px; color:var(--cinza); font-size:16px; }
.ground .nome { font-family:Menlo, Consolas, monospace; }
.ground .trilho { flex:0 0 150px; height:10px; background:#fff; border:1px solid #D0D0D0; }
.ground .trilho span { display:block; height:100%; background:#9A9A9A; }
.ground em { font-style:normal; white-space:nowrap; font-size:14px; letter-spacing:.06em; text-transform:uppercase; }
.infra-motivo { font-family:Menlo, Consolas, monospace; font-size:15px; color:#555; background:#fff; padding:8px 10px; margin:6px 0; word-break:break-word; }
.documentado { margin-top:10px; font-size:16px; color:#6D4B4D; line-height:1.35; }
.documentado b { text-transform:uppercase; font-size:13px; letter-spacing:.08em; margin-right:6px; }
"""
