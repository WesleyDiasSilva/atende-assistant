import { useEffect, useRef, useState } from 'react'

import BaseDeConhecimento from './BaseDeConhecimento'

const PERGUNTAS_DE_EXEMPLO = [
  'Meu pedido chegou descongelado, o que eu faço?',
  'Como funciona a troca de um produto?',
  'Vocês entregam no mesmo dia?',
]

// Os valores vêm do enum do schema; aqui ficam só os rótulos de exibição.
const NOME_DO_TIPO = {
  status_pedido: 'Status do pedido',
  troca: 'Troca',
  duvida_produto: 'Dúvida de produto',
  outro: 'Outro',
}

// O passo que cada modo acrescenta antes do prompt, no inspetor da chain.
const NOME_DO_PASSO_DE_CONTEXTO = {
  stuffing: 'base',
  rag: 'busca',
  rag_gerenciado: 'busca AWS',
}

const formatarSegundos = (ms) => `${(ms / 1000).toFixed(1).replace('.', ',')} s`
const formatarTokens = (valor) => valor.toLocaleString('pt-BR')
const formatarTemperatura = (valor) => valor.toFixed(1).replace('.', ',')
const formatarHorario = (iso) =>
  new Date(iso).toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' })

export default function App() {
  // O catálogo de modelos, perfis, modos de conhecimento e faixa de temperatura
  // vem da API — a interface não conhece nenhum desses valores de antemão.
  const [configuracao, setConfiguracao] = useState(null)
  const [estadoDaApi, setEstadoDaApi] = useState('verificando')

  const [modeloEscolhido, setModeloEscolhido] = useState('rapido')
  const [perfilEscolhido, setPerfilEscolhido] = useState('padrao')
  // Quanto da base de conhecimento o atendente recebe. É o único controle que
  // muda o que ele sabe, e por isso o único que pode mudar o conteúdo da resposta.
  const [modoEscolhido, setModoEscolhido] = useState('sem_conhecimento')
  const [temperatura, setTemperatura] = useState(0)
  // Liga o ciclo de auto-correção: quando a resposta admite não ter achado na
  // base, o fluxo amplia a busca e tenta uma segunda vez, com a mesma pergunta.
  const [autoCorrigir, setAutoCorrigir] = useState(false)
  // Se o atendente usa a conversa anterior para responder este turno.
  const [memoriaAtiva, setMemoriaAtiva] = useState(true)
  // Pede que a próxima execução pare num node, uma vez. Serve para ver o que o
  // estado gravado permite: continuar de onde parou em vez de recomeçar.
  const [interromperEm, setInterromperEm] = useState('')

  // Qual conversa é esta. Vai em cada pergunta e é a chave sob a qual o backend
  // grava o estado do fluxo — é ela que faz o atendente lembrar do turno
  // anterior, e é ela que separa esta conversa de qualquer outra.
  //
  // Guardada no navegador de propósito: recarregar a página não pode virar
  // conversa nova. O identificador é de quem conversa, não da aba aberta.
  const [conversaId, setConversaId] = useState(() => {
    const gravada = localStorage.getItem('conversa_id')
    if (gravada) return gravada
    const nova = crypto.randomUUID()
    localStorage.setItem('conversa_id', nova)
    return nova
  })

  // Começar de novo é trocar a chave, não apagar nada: a conversa anterior
  // continua gravada no banco, sob a chave antiga.
  const comecarNovaConversa = () => {
    const nova = crypto.randomUUID()
    localStorage.setItem('conversa_id', nova)
    setConversaId(nova)
    setConversa([])
  }

  const [textoDigitado, setTextoDigitado] = useState('')
  const [conversa, setConversa] = useState([])
  const [aguardandoResposta, setAguardandoResposta] = useState(false)

  const [abaAtiva, setAbaAtiva] = useState('conversa')
  const [metricas, setMetricas] = useState(null)
  const [solicitacoes, setSolicitacoes] = useState(null)

  // Recarrega ao abrir o painel: os números mudam a cada mensagem respondida na
  // outra aba, então buscar uma vez só deixaria a tela desatualizada.
  useEffect(() => {
    if (abaAtiva !== 'dashboard') return
    fetch('/api/metricas')
      .then((r) => r.json())
      .then(setMetricas)
      .catch(() => setMetricas({ total: 0, por_tipo: {}, ultimos: [] }))
  }, [abaAtiva])

  const carregarSolicitacoes = () =>
    fetch('/api/solicitacoes')
      .then((r) => r.json())
      .then(setSolicitacoes)
      .catch(() => setSolicitacoes({ itens: [], abertas: 0 }))

  useEffect(() => {
    if (abaAtiva !== 'solicitacoes') return
    carregarSolicitacoes()
  }, [abaAtiva])

  // Fechar um item é decisão de quem atende; a tela recarrega para refletir.
  const resolverSolicitacao = async (protocolo) => {
    await fetch(`/api/solicitacoes/${protocolo}/resolver`, { method: 'POST' })
    carregarSolicitacoes()
  }

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
        if (dados.modo_padrao) setModoEscolhido(dados.modo_padrao)
      })
      .catch(() => setConfiguracao({ modelos: [], perfis: [], modos: [] }))
  }, [])

  // Mantém a última mensagem visível conforme a conversa cresce.
  useEffect(() => {
    fimDaConversa.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [conversa, aguardandoResposta])

  const modeloAtual = configuracao?.modelos?.find((m) => m.id === modeloEscolhido)
  const perfilAtual = configuracao?.perfis?.find((p) => p.id === perfilEscolhido)
  const modoAtual = configuracao?.modos?.find((m) => m.id === modoEscolhido)
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
          modo: modoEscolhido,
          temperatura,
          auto_corrigir: autoCorrigir,
          conversa_id: conversaId,
          memoria_ativa: memoriaAtiva,
          interromper_em: interromperEm,
        }),
      })
      const dados = await resposta.json()

      // A execução parou no meio: não há resposta para mostrar, há um turno
      // pendente. O interruptor volta a zero porque ele é de um disparo só —
      // deixá-lo ligado faria a retomada parecer não ter funcionado.
      if (resposta.ok && dados.interrompido) {
        setInterromperEm('')
        setConversa((atual) => [
          ...atual,
          {
            autor: 'atendente',
            interrompido: true,
            node: dados.node,
            texto: dados.detalhe,
            duracaoEmMs: performance.now() - inicio,
          },
        ])
        return
      }

      setConversa((atual) => [
        ...atual,
        {
          autor: 'atendente',
          texto: resposta.ok
            ? dados.resposta
            : dados.detail?.[0]?.msg || dados.detail || 'Não foi possível responder.',
          // Campo do schema, lido pelo nome. É o mesmo valor que alimenta a
          // contagem por assunto do painel.
          tipo: resposta.ok ? dados.tipo : null,
          houveErro: !resposta.ok,
          duracaoEmMs: performance.now() - inicio,
          nomeDoModelo: modeloAtual?.nome,
          nomeDoPerfil: perfilAtual?.nome,
          nomeDoModo: modoAtual?.nome,
          // O que o contexto custou nesta pergunta. É o número que compara os
          // modos: a mesma resposta pode sair por um décimo dos tokens.
          tokensDeEntrada: resposta.ok ? dados.tokens_de_entrada : null,
          // Os documentos que a busca trouxe. Vêm do nosso código, não do texto
          // do modelo — é isso que permite conferir a resposta contra a base.
          fontes: resposta.ok ? dados.fontes : null,
          // Quantas vezes o fluxo ampliou a busca antes de responder. Vem do
          // nosso código: quem contou as voltas foi o grafo, não o modelo.
          tentativas: resposta.ok ? dados.tentativas : null,
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

  // Continua a execução que ficou pendente. O backend lê o estado gravado e roda
  // só o que faltava — daí a resposta chegar com o custo do que já havia sido
  // feito antes da parada, e não com um custo novo.
  async function retomarExecucao(indice) {
    if (aguardandoResposta) return
    setAguardandoResposta(true)
    const inicio = performance.now()
    try {
      const resposta = await fetch(`/api/conversas/${conversaId}/retomar`, {
        method: 'POST',
      })
      const dados = await resposta.json()
      setConversa((atual) => {
        const proximo = [...atual]
        if (proximo[indice]) proximo[indice] = { ...proximo[indice], retomado: true }
        proximo.push({
          autor: 'atendente',
          texto: resposta.ok
            ? dados.resposta
            : dados.detail || 'Não foi possível retomar a execução.',
          tipo: resposta.ok ? dados.tipo : null,
          houveErro: !resposta.ok,
          duracaoEmMs: performance.now() - inicio,
          nomeDoModelo: modeloAtual?.nome,
          nomeDoPerfil: perfilAtual?.nome,
          nomeDoModo: modoAtual?.nome,
          tokensDeEntrada: resposta.ok ? dados.tokens_de_entrada : null,
          fontes: resposta.ok ? dados.fontes : null,
          tentativas: resposta.ok ? dados.tentativas : null,
          // De onde a execução continuou. Só aparece em turno retomado.
          retomadoDe: resposta.ok ? dados.retomado_de : null,
          temperaturaUsada: temperatura,
        })
        return proximo
      })
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
        {/* A navegação separa o atendimento em si do que o sistema apurou a
            partir dos campos da resposta. */}
        <nav className="abas">
          <button
            type="button"
            className={abaAtiva === 'conversa' ? 'ativo' : ''}
            onClick={() => setAbaAtiva('conversa')}
          >
            Conversa
          </button>
          <button
            type="button"
            className={abaAtiva === 'dashboard' ? 'ativo' : ''}
            onClick={() => setAbaAtiva('dashboard')}
          >
            Dashboard
          </button>
          <button
            type="button"
            className={abaAtiva === 'base' ? 'ativo' : ''}
            onClick={() => setAbaAtiva('base')}
          >
            Base de conhecimento
          </button>
          <button
            type="button"
            className={abaAtiva === 'solicitacoes' ? 'ativo' : ''}
            onClick={() => setAbaAtiva('solicitacoes')}
          >
            Solicitações
            {solicitacoes?.abertas > 0 && (
              <span className="contador">{solicitacoes.abertas}</span>
            )}
          </button>
        </nav>

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
        {abaAtiva === 'base' ? (
          <BaseDeConhecimento />
        ) : abaAtiva === 'solicitacoes' ? (
          <section className="dashboard">
            <h2 className="secao">Fila de atendimento</h2>
            {!solicitacoes ? (
              <p className="vazio">Carregando…</p>
            ) : solicitacoes.itens.length === 0 ? (
              <p className="vazio">
                Nenhuma solicitação na fila. Ela recebe os casos que o atendimento
                encaminha para uma pessoa resolver.
              </p>
            ) : (
              <ul className="fila">
                {solicitacoes.itens.map((item) => (
                  <li
                    key={item.protocolo}
                    className={item.situacao === 'resolvida' ? 'resolvida' : ''}
                  >
                    <div className="linha-topo">
                      <span className="protocolo num">{item.protocolo}</span>
                      {/* A origem diz por qual caminho o caso entrou na fila. */}
                      <span className={`origem ${item.origem}`}>
                        {item.origem === 'troca' ? 'Troca aberta' : 'Encaminhado'}
                      </span>
                      <span className="chip-tipo">
                        {NOME_DO_TIPO[item.assunto] || item.assunto}
                      </span>
                      <span className="quando num">{formatarHorario(item.quando)}</span>
                      {item.situacao === 'aberta' ? (
                        <button
                          type="button"
                          className="resolver"
                          onClick={() => resolverSolicitacao(item.protocolo)}
                        >
                          Resolver
                        </button>
                      ) : (
                        <span className="selo-resolvida">Resolvida</span>
                      )}
                    </div>
                    <p className="motivo">{item.motivo}</p>
                    {item.pergunta && <p className="origem-msg">“{item.pergunta}”</p>}
                  </li>
                ))}
              </ul>
            )}
          </section>
        ) : abaAtiva === 'dashboard' ? (
          <section className="dashboard">
            <h2 className="secao">Atendimentos por assunto</h2>
            {!metricas ? (
              <p className="vazio">Carregando…</p>
            ) : metricas.total === 0 ? (
              <p className="vazio">
                Nenhum atendimento registrado ainda. Responda uma mensagem na aba
                Conversa para o painel começar a contar.
              </p>
            ) : (
              <>
                <div className="cartoes">
                  {Object.entries(metricas.por_tipo).map(([tipo, quantidade]) => (
                    <div key={tipo} className="cartao">
                      <span className="rotulo">{NOME_DO_TIPO[tipo] || tipo}</span>
                      <strong className="num">{quantidade}</strong>
                      {/* Proporção sobre o total, para comparar assuntos sem
                          depender só do número absoluto. */}
                      <div className="barra">
                        <div
                          className="preenchimento"
                          style={{
                            width: `${metricas.total ? (quantidade / metricas.total) * 100 : 0}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                <h2 className="secao">Últimos atendimentos</h2>
                <ul className="lista-atendimentos">
                  {metricas.ultimos.map((item, indice) => (
                    <li key={indice}>
                      <span className="chip-tipo">
                        {NOME_DO_TIPO[item.tipo] || item.tipo}
                      </span>
                      <span className="pergunta">{item.pergunta}</span>
                      <span className="quando num">{formatarHorario(item.quando)}</span>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </section>
        ) : (
          <>
        <section className="conversa">
          <div className="rolagem">
            {conversa.length === 0 ? (
              <div className="inicio">
                <h1>Como posso ajudar o cliente?</h1>
                <p>
                  Escreva a mensagem que o cliente enviou. A resposta usa os
                  controles selecionados ao lado — e o de <strong>conhecimento</strong>{' '}
                  é o único que muda o que o atendente sabe.
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
                    {/* O assunto vem em campo próprio da resposta: a interface o
                        exibe sem precisar interpretar o texto. */}
                    {mensagem.tipo && (
                      <span className="chip-tipo">{NOME_DO_TIPO[mensagem.tipo] || mensagem.tipo}</span>
                    )}
                    {mensagem.interrompido ? (
                      /* Turno pendente, e não turno errado: o que ele oferece é
                         continuar, porque o passo anterior está gravado. */
                      <div className="bolha interrompida">
                        {mensagem.texto}
                        {!mensagem.retomado && (
                          <button
                            type="button"
                            className="retomar"
                            onClick={() => retomarExecucao(indice)}
                            disabled={aguardandoResposta}
                          >
                            Retomar execução
                          </button>
                        )}
                      </div>
                    ) : (
                      <div className={`bolha ${mensagem.houveErro ? 'erro' : ''}`}>
                        {mensagem.texto}
                      </div>
                    )}
                    {/* Fora da bolha, como campo próprio: quem lê a resposta
                        consegue abrir o documento e conferir.

                        O rótulo diz "recuperados", e não "fontes", porque é o que
                        a busca trouxe — não necessariamente o que a resposta usou.
                        Quando o atendente diz que não encontrou na base, estes
                        trechos apareceram e não serviram; sem o rótulo, a lista
                        pareceria contradizer a própria resposta. */}
                    {mensagem.fontes?.length > 0 && (
                      <div className="recuperados">
                        <span className="rotulo-recuperados">
                          Trechos recuperados
                        </span>
                        <ul className="fontes">
                          {mensagem.fontes.map((arquivo) => (
                            <li key={arquivo}>{arquivo}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {/* Registra a configuração de cada turno: ao trocar um controle
                        e repetir a pergunta, a diferença fica documentada na tela. */}
                    {mensagem.autor === 'atendente' &&
                      !mensagem.houveErro &&
                      !mensagem.interrompido && (
                      <span className="rodape-msg">
                        {mensagem.nomeDoModelo} · {mensagem.nomeDoPerfil} ·{' '}
                        {mensagem.nomeDoModo} · temp{' '}
                        <span className="num">
                          {formatarTemperatura(mensagem.temperaturaUsada)}
                        </span>{' '}
                        ·{' '}
                        <span className="num">
                          {formatarSegundos(mensagem.duracaoEmMs)}
                        </span>
                        {mensagem.tokensDeEntrada > 0 && (
                          <>
                            {' · '}
                            <span className="num">
                              {formatarTokens(mensagem.tokensDeEntrada)}
                            </span>{' '}
                            tokens de entrada
                          </>
                        )}
                        {/* Só aparece quando houve volta: no caminho normal a
                            contagem é zero e a menção seria ruído. */}
                        {mensagem.tentativas > 0 && (
                          <>
                            {' · '}
                            <span className="num">{mensagem.tentativas}</span>{' '}
                            {mensagem.tentativas === 1
                              ? 'busca ampliada'
                              : 'buscas ampliadas'}
                          </>
                        )}
                        {/* Só em turno retomado: diz de que node a execução
                            continuou, em vez de deixar parecer que rodou inteira. */}
                        {mensagem.retomadoDe?.length > 0 && (
                          <>
                            {' · '}retomado de{' '}
                            <span className="num">
                              {mensagem.retomadoDe.join(', ')}
                            </span>
                          </>
                        )}
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

          {/* Entra logo depois do modelo, e antes de perfil e temperatura, porque
              é o único destes controles que mexe no conteúdo da resposta — os
              outros dois mexem na redação. */}
          <div className="campo">
            <label>Conhecimento</label>
            <div className="segmentado">
              {configuracao?.modos?.map((modo) => (
                <button
                  key={modo.id}
                  type="button"
                  className={modoEscolhido === modo.id ? 'ativo' : ''}
                  // Um modo pode depender de recurso que esta instalação não tem.
                  // O botão continua na tela, desabilitado, com o motivo no title:
                  // esconder o modo faria parecer que ele não existe.
                  disabled={modo.disponivel === false}
                  title={modo.indisponivel_porque || undefined}
                  onClick={() => setModoEscolhido(modo.id)}
                >
                  <strong>{modo.nome}</strong>
                  <span>
                    {modo.disponivel === false
                      ? modo.indisponivel_porque
                      : modo.descricao}
                  </span>
                </button>
              ))}
            </div>
            <p className="dica">
              Muda o que o atendente sabe. A mesma pergunta pode voltar com outra
              resposta.
            </p>
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

          {/* Não muda o que o atendente sabe nem como ele escreve: muda o que
              ele leva do turno anterior para este. Vale em todos os modos,
              porque o histórico é anterior à decisão de contexto. */}
          <div className="campo">
            <label>Memória da conversa</label>
            <div className="segmentado">
              <button
                type="button"
                className={memoriaAtiva ? '' : 'ativo'}
                onClick={() => setMemoriaAtiva(false)}
              >
                <strong>Desligada</strong>
                <span>Cada pergunta é a primeira. Nada do turno anterior entra.</span>
              </button>
              <button
                type="button"
                className={memoriaAtiva ? 'ativo' : ''}
                onClick={() => setMemoriaAtiva(true)}
              >
                <strong>Ligada</strong>
                <span>A conversa anterior entra junto com a pergunta de agora.</span>
              </button>
            </div>
            <p className="dica">
              Desligada, o turno não lê o histórico <strong>e não entra nele</strong>
              . O que já estava gravado continua gravado — desligar não apaga.
            </p>
            <div className="conversa-atual">
              <code>{conversaId.slice(0, 8)}</code>
              <button type="button" className="nova-conversa" onClick={comecarNovaConversa}>
                Nova conversa
              </button>
            </div>
          </div>

          {/* Entra por último porque não muda o que o atendente sabe nem como
              ele escreve: muda o que o fluxo faz quando a primeira resposta não
              se sustentou na base. Só tem efeito nos modos de busca — é lá que
              existe um alcance para ampliar. */}
          <div className="campo">
            <label>Auto-correção</label>
            <div className="segmentado">
              <button
                type="button"
                className={autoCorrigir ? '' : 'ativo'}
                onClick={() => setAutoCorrigir(false)}
              >
                <strong>Desligada</strong>
                <span>Uma passada só. Não achou, não achou.</span>
              </button>
              <button
                type="button"
                className={autoCorrigir ? 'ativo' : ''}
                onClick={() => setAutoCorrigir(true)}
              >
                <strong>Ligada</strong>
                <span>Não achou na base? Amplia a busca e tenta de novo.</span>
              </button>
            </div>
            <p className="dica">
              A pergunta do cliente não é reescrita — o que muda é quantos trechos
              a busca devolve na segunda tentativa. Se ainda assim não achar, o
              caso vai para a fila humana.
            </p>
          </div>

          {/* Não é um controle de produto: é o que permite observar, com o código
              que está no ar, o que o estado gravado garante quando um passo
              falha. Vale um disparo e volta a zero sozinho. */}
          <div className="campo">
            <label>Interromper execução</label>
            <div className="segmentado">
              <button
                type="button"
                className={interromperEm ? '' : 'ativo'}
                onClick={() => setInterromperEm('')}
              >
                <strong>Não</strong>
                <span>A execução vai até o fim.</span>
              </button>
              <button
                type="button"
                className={interromperEm === 'formalizar' ? 'ativo' : ''}
                onClick={() => setInterromperEm('formalizar')}
              >
                <strong>Parar em formalizar</strong>
                <span>Depois da busca e da conversa, antes do formato.</span>
              </button>
            </div>
            <p className="dica">
              A próxima pergunta para nesse node. A resposta traz o botão de
              retomar, e a retomada roda <strong>só o que faltava</strong> — a
              busca e as idas ao modelo anteriores não se repetem.
            </p>
          </div>

          {/* Mostra a chain que está de fato sendo executada, com o identificador
              completo do modelo. Ajuda a conferir o que mudou ao trocar um controle. */}
          <h2 className="secao">Chain ativa</h2>
          <div className="inspetor">
            <div className="pipe">
              {/* O modo de conhecimento acrescenta um passo antes do prompt: no
                  stuffing a base inteira, nos dois modos de busca só o que o
                  retriever trouxe — o nosso ou o da AWS. */}
              {modoEscolhido !== 'sem_conhecimento' && (
                <>
                  <span>{NOME_DO_PASSO_DE_CONTEXTO[modoEscolhido]}</span>
                  <em>|</em>
                </>
              )}
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
          </>
        )}
      </main>
    </div>
  )
}
