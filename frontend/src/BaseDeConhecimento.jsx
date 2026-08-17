import { useEffect, useState } from 'react'

// A aba da base de conhecimento: o que está no disco, o que está indexado, e o
// caminho para acrescentar um documento novo.
//
// Fica em arquivo próprio porque é uma tela inteira com estado próprio — e
// porque quem lê o App.jsx não precisa passar por ela para entender a conversa.
export default function BaseDeConhecimento() {
  const [base, setBase] = useState(null)
  const [etapas, setEtapas] = useState([])
  const [enviando, setEnviando] = useState(false)
  const [erro, setErro] = useState(null)

  const carregar = () =>
    fetch('/api/base/documentos')
      .then((r) => r.json())
      .then(setBase)
      .catch(() => setBase({ itens: [], total: 0, chunks_indexados: 0 }))

  useEffect(() => {
    carregar()
  }, [])

  // O .md é lido como texto no navegador e enviado como texto no JSON. Não há
  // nada binário num documento de política.
  async function enviarArquivo(arquivo) {
    if (!arquivo) return
    setEnviando(true)
    setEtapas([])
    setErro(null)
    try {
      const resposta = await fetch('/api/base/documentos', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          arquivo: arquivo.name,
          conteudo: await arquivo.text(),
        }),
      })
      const dados = await resposta.json()
      if (!resposta.ok) throw new Error(dados.detail?.[0]?.msg || dados.detail)
      setEtapas(dados.etapas)
      carregar()
    } catch (falha) {
      setErro(falha.message || 'Não foi possível enviar o documento.')
    } finally {
      setEnviando(false)
    }
  }

  async function removerDocumento(nomeDoArquivo) {
    setErro(null)
    try {
      // O nome vai codificado: um arquivo enviado como "Politica de Frete.md" tem
      // espaço no nome, e espaço cru na URL não chega inteiro ao servidor.
      const resposta = await fetch(
        `/api/base/documentos/${encodeURIComponent(nomeDoArquivo)}`,
        { method: 'DELETE' },
      )
      if (!resposta.ok) throw new Error((await resposta.json()).detail)
    } catch (falha) {
      setErro(falha.message || 'Não foi possível remover o documento.')
    }
    carregar()
  }

  return (
    <section className="dashboard">
      <h2 className="secao">Acrescentar documento</h2>
      <div className="upload">
        <label className="botao-arquivo">
          <input
            type="file"
            accept=".md"
            disabled={enviando}
            onChange={(evento) => {
              enviarArquivo(evento.target.files[0])
              // Limpa o campo para o mesmo arquivo poder ser reenviado depois
              // de uma edição — sem isso o onChange não dispara de novo.
              evento.target.value = ''
            }}
          />
          {enviando ? 'Indexando…' : 'Escolher um arquivo .md'}
        </label>
        <p className="dica">
          O documento é gravado na base e indexado na hora: ele já vale na próxima
          pergunta feita no modo de busca.
        </p>
      </div>

      {/* As etapas da indexação, na ordem em que aconteceram. É o que mostra que
          indexar tem passos, e não é um botão que "liga" o documento. */}
      {etapas.length > 0 && (
        <ol className="etapas">
          {etapas.map((etapa) => (
            <li key={etapa}>{etapa}</li>
          ))}
        </ol>
      )}
      {erro && <p className="vazio erro-texto">{erro}</p>}

      <h2 className="secao">
        Documentos na base
        {base && (
          <span className="contagem num">
            {base.total} documento(s) · {base.chunks_indexados} chunk(s) indexados
          </span>
        )}
      </h2>

      {!base ? (
        <p className="vazio">Carregando…</p>
      ) : base.itens.length === 0 ? (
        <p className="vazio">
          A base está vazia. Envie um arquivo .md para o atendente ter o que ler.
        </p>
      ) : (
        <ul className="documentos">
          {base.itens.map((item) => (
            <li key={item.arquivo}>
              <span className="titulo-doc">{item.titulo}</span>
              <code>{item.arquivo}</code>
              {/* Zero chunks significa que o arquivo está no disco mas não foi
                  indexado: a busca não o encontra. */}
              <span className={`chunks num ${item.chunks === 0 ? 'ausente' : ''}`}>
                {item.chunks === 0 ? 'não indexado' : `${item.chunks} chunk(s)`}
              </span>
              <button type="button" onClick={() => removerDocumento(item.arquivo)}>
                Remover
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
