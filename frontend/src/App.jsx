import { useEffect, useRef, useState } from 'react'

const PERGUNTAS_DE_EXEMPLO = [
  'Meu pedido chegou descongelado, o que eu faço?',
  'Como funciona a troca de um produto?',
  'Vocês entregam no mesmo dia?',
]

const formatarSegundos = (ms) => `${(ms / 1000).toFixed(1).replace('.', ',')} s`
const formatarTemperatura = (valor) => valor.toFixed(1).replace('.', ',')

export default function App() {
  // O catálogo de modelos, perfis e faixa de temperatura vem da API — a
  // interface não conhece nenhum desses valores de antemão.
  const [configuracao, setConfiguracao] = useState(null)
  const [estadoDaApi, setEstadoDaApi] = useState('verificando')

  const [modeloEscolhido, setModeloEscolhido] = useState('rapido')
  const [perfilEscolhido, setPerfilEscolhido] = useState('padrao')
  const [temperatura, setTemperatura] = useState(0)

  const [textoDigitado, setTextoDigitado] = useState('')
  const [conversa, setConversa] = useState([])
  const [aguardandoResposta, setAguardandoResposta] = useState(false)

  const fimDaConversa = useRef(null)
  const campoDeTexto = useRef(null)

  useEffect(() => {
    fetch('/api/saude')
      .then((r) => setEstadoDaApi(r.ok ? 'ok' : 'erro'))
      .catch(() => setEstadoDaApi('erro'))

    fetch('/api/configuracao')
      .then((r) => r.json())
      .then((dados) => {
        setConfiguracao(dados)
        if (dados.temperatura) setTemperatura(dados.temperatura.padrao)
      })
      .catch(() => setConfiguracao({ modelos: [], perfis: [] }))
  }, [])

  // Mantém a última mensagem visível conforme a conversa cresce.
  useEffect(() => {
    fimDaConversa.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [conversa, aguardandoResposta])

  const modeloAtual = configuracao?.modelos?.find((m) => m.id === modeloEscolhido)
  const perfilAtual = configuracao?.perfis?.find((p) => p.id === perfilEscolhido)
  const faixaDeTemperatura =
    configuracao?.temperatura ?? { minima: 0, maxima: 1, passo: 0.1 }

  async function enviarPergunta(textoAlternativo) {
    const pergunta = (textoAlternativo ?? textoDigitado).trim()
    if (!pergunta || aguardandoResposta) return

    setTextoDigitado('')
    setConversa((atual) => [...atual, { autor: 'cliente', texto: pergunta }])
    setAguardandoResposta(true)

    // Mede o tempo real da chamada, que aparece no rodapé da resposta.
    const inicio = performance.now()
    try {
      const resposta = await fetch('/api/responder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pergunta,
          perfil: perfilEscolhido,
          modelo: modeloEscolhido,
          temperatura,
        }),
      })
      const dados = await resposta.json()
      setConversa((atual) => [
        ...atual,
        {
          autor: 'atendente',
          texto: resposta.ok
            ? dados.resposta
            : dados.detail?.[0]?.msg || dados.detail || 'Não foi possível responder.',
          houveErro: !resposta.ok,
          duracaoEmMs: performance.now() - inicio,
          nomeDoModelo: modeloAtual?.nome,
          nomeDoPerfil: perfilAtual?.nome,
          temperaturaUsada: temperatura,
        },
      ])
    } catch {
      setConversa((atual) => [
        ...atual,
        {
          autor: 'atendente',
          texto: 'Falha de conexão com a API.',
          houveErro: true,
          duracaoEmMs: performance.now() - inicio,
        },
      ])
    } finally {
      setAguardandoResposta(false)
      campoDeTexto.current?.focus()
    }
  }

  return (
    <div className="app">
      <header className="topo">
        <div className="marca">
          <span className="marca-selo" aria-hidden="true" />
          <div>
            <strong>Atendente de Pedidos</strong>
            <span>Console de atendimento</span>
          </div>
        </div>
        <div className={`estado ${estadoDaApi}`}>
          <span className="ponto" aria-hidden="true" />
          {estadoDaApi === 'ok'
            ? 'API conectada'
            : estadoDaApi === 'erro'
              ? 'API indisponível'
              : 'verificando…'}
        </div>
      </header>

      <main className="corpo">
        <section className="conversa">
          <div className="rolagem">
            {conversa.length === 0 ? (
              <div className="inicio">
                <h1>Como posso ajudar o cliente?</h1>
                <p>
                  Escreva a mensagem que o cliente enviou. A resposta usa o modelo, o
                  perfil e a temperatura selecionados ao lado.
                </p>
                <div className="chips">
                  {PERGUNTAS_DE_EXEMPLO.map((exemplo) => (
                    <button
                      key={exemplo}
                      onClick={() => enviarPergunta(exemplo)}
                      disabled={aguardandoResposta}
                    >
                      {exemplo}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <ol className="mensagens">
                {conversa.map((mensagem, indice) => (
                  <li key={indice} className={`item ${mensagem.autor}`}>
                    <span className="autor">
                      {mensagem.autor === 'cliente' ? 'Cliente' : 'Atendente'}
                    </span>
                    <div className={`bolha ${mensagem.houveErro ? 'erro' : ''}`}>
                      {mensagem.texto}
                    </div>
                    {/* Registra a configuração de cada turno: ao trocar um controle
                        e repetir a pergunta, a diferença fica documentada na tela. */}
                    {mensagem.autor === 'atendente' && !mensagem.houveErro && (
                      <span className="rodape-msg">
                        {mensagem.nomeDoModelo} · {mensagem.nomeDoPerfil} · temp{' '}
                        <span className="num">
                          {formatarTemperatura(mensagem.temperaturaUsada)}
                        </span>{' '}
                        ·{' '}
                        <span className="num">
                          {formatarSegundos(mensagem.duracaoEmMs)}
                        </span>
                      </span>
                    )}
                  </li>
                ))}
                {aguardandoResposta && (
                  <li className="item atendente">
                    <span className="autor">Atendente</span>
                    <div className="bolha digitando" aria-label="digitando">
                      <i /><i /><i />
                    </div>
                  </li>
                )}
              </ol>
            )}
            <div ref={fimDaConversa} />
          </div>

          <form
            className="composer"
            onSubmit={(evento) => {
              evento.preventDefault()
              enviarPergunta()
            }}
          >
            <input
              ref={campoDeTexto}
              value={textoDigitado}
              onChange={(evento) => setTextoDigitado(evento.target.value)}
              placeholder="Mensagem do cliente…"
              disabled={aguardandoResposta}
              autoFocus
            />
            <button type="submit" disabled={aguardandoResposta || !textoDigitado.trim()}>
              Enviar
            </button>
          </form>
        </section>

        <aside className="painel">
          <h2 className="secao">Configuração</h2>

          <div className="campo">
            <label>Modelo</label>
            <div className="segmentado">
              {configuracao?.modelos?.map((modelo) => (
                <button
                  key={modelo.id}
                  type="button"
                  className={modeloEscolhido === modelo.id ? 'ativo' : ''}
                  onClick={() => setModeloEscolhido(modelo.id)}
                >
                  <strong>{modelo.nome}</strong>
                  <span>{modelo.custo_relativo}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="campo">
            <label>Perfil de atendimento</label>
            <div className="segmentado">
              {configuracao?.perfis?.map((perfil) => (
                <button
                  key={perfil.id}
                  type="button"
                  className={perfilEscolhido === perfil.id ? 'ativo' : ''}
                  onClick={() => setPerfilEscolhido(perfil.id)}
                >
                  <strong>{perfil.nome}</strong>
                </button>
              ))}
            </div>
            <p className="dica">Muda como o atendente escreve, não o que ele sabe.</p>
          </div>

          <div className="campo">
            <label htmlFor="temperatura">
              Temperatura
              <output className="num" htmlFor="temperatura">
                {formatarTemperatura(temperatura)}
              </output>
            </label>
            <input
              id="temperatura"
              type="range"
              min={faixaDeTemperatura.minima}
              max={faixaDeTemperatura.maxima}
              step={faixaDeTemperatura.passo}
              value={temperatura}
              onChange={(evento) => setTemperatura(Number(evento.target.value))}
            />
            <div className="escala num">
              <span>{formatarTemperatura(faixaDeTemperatura.minima)}</span>
              <span>{formatarTemperatura(faixaDeTemperatura.maxima)}</span>
            </div>
            <p className="dica">
              {temperatura === 0
                ? 'Em 0 o modelo escolhe sempre o token mais provável, então a resposta tende a se repetir.'
                : 'Acima de 0 a mesma pergunta pode devolver respostas diferentes.'}
            </p>
          </div>

          {/* Mostra a chain que está de fato sendo executada, com o identificador
              completo do modelo. Ajuda a conferir o que mudou ao trocar um controle. */}
          <h2 className="secao">Chain ativa</h2>
          <div className="inspetor">
            <div className="pipe">
              <span>prompt</span>
              <em>|</em>
              <span>model</span>
              <em>|</em>
              <span>parser</span>
            </div>
            {modeloAtual && (
              <>
                <code>{modeloAtual.model_id}</code>
                <code className="num">temperature={formatarTemperatura(temperatura)}</code>
              </>
            )}
          </div>
        </aside>
      </main>
    </div>
  )
}
