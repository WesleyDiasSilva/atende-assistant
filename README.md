# atende-assistant

Assistente de atendimento ao cliente para um cenário de e-commerce de produtos
congelados (pedidos e políticas), construído de forma incremental.

Neste estado o assistente responde num formato fixo, classifica o assunto de
cada mensagem, consulta pedidos por ferramenta e abre solicitações de troca —
dentro da política, que é verificada no código. O que ele não resolve vai para
uma fila de atendimento humano.

Ele também tem acesso aos documentos da empresa, e **quanto** desse acesso ele
tem é um controle da interface: nenhum documento, todos eles, ou só os trechos
que a pergunta recupera de uma base vetorial. Os três modos existem lado a lado
porque a diferença entre eles — na resposta e no custo — é o assunto.

## Estrutura

```
backend/
  app/
    main.py         API (FastAPI): saude, configuracao, responder,
                    metricas, solicitacoes e base de conhecimento
    assistente.py   o ciclo de ferramenta, o modo de conhecimento e a saída
                    estruturada
    retrieval.py    embeddings, chunking, indexação e busca (pgvector)
    documentos.py   leitura e escrita dos .md da base
    db.py           conexão com o Postgres
    tools.py        ferramentas: consulta de pedido e abertura de troca
    regras.py       política de troca — a decisão de negócio
    schemas.py      os contratos da resposta (Pydantic)
    dados.py        persistência em arquivo
    config.py       catálogo de modelos, perfis, modos e temperatura
    log.py          configuração de log da API e dos scripts
  scripts/
    indexar_base.py indexa dados/base/ na base vetorial
    limpar_base.py  apaga os vetores, preservando os .md
  dados/
    base/               os documentos da empresa (.md), a base de conhecimento
    exemplos/           documentos .md que NÃO estão na base — servem para
                        acrescentar pela interface e ver a indexação acontecer
    pedidos.json        cadastro de pedidos usado pelas ferramentas
    atendimentos.jsonl  histórico das mensagens respondidas (gerado)
    solicitacoes.jsonl  fila de trabalho humano (gerado)
  requirements.txt  dependências pinadas
  Dockerfile        imagem baseada em python:3.12-slim
db/init.sql         habilita a extensão pgvector na criação do banco
frontend/           interface em React + Vite (imagem node:22-slim)
docker-compose.yml  serviços db, backend e frontend
.env.example        modelo das variáveis de ambiente
```

## Configuração

A autenticação usa uma **chave de API do Bedrock** (bearer token), não par de
chaves IAM. Para gerar:

1. Acesse o console da AWS na região desejada (ex.: `us-east-1`).
2. Abra **Amazon Bedrock** → menu lateral **API keys**.
3. Gere uma chave de curta ou longa duração e copie o valor exibido — ele não
   é mostrado novamente.
4. Confirme em **Model access** que os modelos escolhidos estão habilitados na
   conta. Estar listado como inference profile **não** significa ter acesso.

A chave é lida pelo `boto3` automaticamente a partir da variável de ambiente
`AWS_BEARER_TOKEN_BEDROCK`. Nenhum código passa credenciais explicitamente —
inclusive a chain, que usa o mesmo resolvedor por baixo.

Variáveis usadas:

| Variável | Descrição |
| --- | --- |
| `AWS_BEARER_TOKEN_BEDROCK` | Chave de API do Bedrock |
| `AWS_REGION` | Região de invocação (padrão sugerido: `us-east-1`) |
| `MODEL_ID` | Inference profile do modelo rápido, usado pelo catálogo em `config.py` |
| `DB_HOST` / `DB_PORT` | Onde o Postgres atende. No `.env` valem para quem roda o uvicorn na máquina; dentro do compose o próprio `docker-compose.yml` os sobrescreve para `db:5432` |
| `DB_USER` / `DB_PASSWORD` / `DB_NAME` | Credenciais e banco |
| `EMBEDDING_MODEL` | Modelo que transforma texto em vetor |
| `TOP_K` | Quantos trechos a busca devolve por pergunta |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Tamanho do pedaço e a sobreposição, em caracteres |

`DB_PORT` é a porta publicada **no host**, com 5434 como padrão: 5432 costuma
estar ocupada por um Postgres local e 5433 por outro projeto. Se as três
estiverem em uso, troque a variável — nada no código depende do número.

Trocar `EMBEDDING_MODEL`, `CHUNK_SIZE` ou `CHUNK_OVERLAP` invalida o que já está
indexado: rode `limpar_base` e `indexar_base` depois de mexer em qualquer um dos
três.

O suporte a `AWS_BEARER_TOKEN_BEDROCK` no `boto3`/`botocore` existe a partir da
versão `1.39.0`. O piso do `requirements.txt` é mais alto que isso porque
`langchain-aws` exige `boto3>=1.43.32`.

## Execução

### A) Docker

```bash
cp .env.example .env
# preencha AWS_BEARER_TOKEN_BEDROCK no .env
docker compose up --build

# em outro terminal, uma vez: indexa a base de conhecimento
docker compose exec backend python -m scripts.indexar_base
```

- Interface: http://localhost:5173
- API: http://localhost:8000/api/saude

Sem o passo de indexação a base vetorial fica vazia, e o modo de busca não
encontra nada. A aba **Base de conhecimento** mostra quantos chunks cada
documento tem indexado — zero significa "está no disco, mas a busca não o vê".

### B) Sem Docker

Python 3.12 é a versão recomendada (a mesma da imagem) e 3.10 é o mínimo —
`langchain-aws` exige `>=3.10`. Node 22 para o frontend.

O Postgres com pgvector continua vindo do compose: `docker compose up -d db`.

```bash
# backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env .env          # ou exporte as variáveis no shell
python -m scripts.indexar_base   # uma vez, para popular a base vetorial
uvicorn app.main:app --reload

# frontend, em outro terminal
cd frontend
npm install
npm run dev
```

## A base de conhecimento

Os documentos da empresa são os `.md` de `backend/dados/base/`. Eles chegam ao
atendente por dois caminhos, e a diferença entre os dois é o assunto do projeto:

**No prompt inteiro (stuffing).** Todos os documentos entram no contexto a cada
pergunta. Simples, e não precisa de banco nenhum — mas paga a base completa em
tokens toda vez, mesmo quando a resposta está num parágrafo só.

**Por busca (RAG).** Cada documento é partido em pedaços de até
`CHUNK_SIZE` caracteres, cada pedaço vira um vetor pelo `EMBEDDING_MODEL`, e os
vetores ficam no Postgres. A pergunta é transformada em vetor pelo **mesmo**
modelo, e só os `TOP_K` pedaços mais próximos entram no contexto.

O mesmo modelo nos dois lados não é detalhe: cada modelo projeta o texto num
espaço próprio, e distância entre vetores de espaços diferentes não mede
semelhança nenhuma.

```bash
python -m scripts.indexar_base   # indexa dados/base/ (idempotente)
python -m scripts.limpar_base    # apaga os vetores, mantém os .md
```

A aba **Base de conhecimento** lista os documentos com a contagem de chunks de
cada um, e permite acrescentar um `.md` — que é gravado e indexado na hora, com
as etapas aparecendo na tela.

### Quem recusa é o prompt, não o score

A busca devolve um score de distância junto de cada trecho, e ele **não** serve
para decidir se a pergunta tem resposta na base. Medido nesta base, as faixas de
score de pergunta coberta e não coberta se sobrepõem: uma pergunta que a base não
cobre pontua melhor que várias que ela cobre.

O "não encontrei na base" vem da instrução que acompanha o contexto, em
`assistente.py`. Ela chega junto com os documentos, e não antes deles — é por isso
que o modo sem conhecimento não recusa: sem documento, não há o que instruir.

## Os quatro controles da interface

**Conhecimento** — quanto da base entra no contexto: nada, tudo, ou os trechos
recuperados. É o único controle que muda *o que* o atendente sabe. O rodapé de
cada resposta mostra os tokens de entrada, que é o custo daquele contexto.

**Modelo** — troca qual modelo responde. Os três configurados foram verificados
nesta conta, com acesso liberado e `temperature` aceito. O custo é indicado em
ordem relativa; a latência real de cada resposta aparece no rodapé da mensagem.

**Perfil de atendimento** — troca o tom da resposta. Muda *como* o atendente
escreve, não *o que* ele sabe.

**Temperatura** — de 0 a 1. Em 0 o modelo escolhe sempre o token mais provável e
a resposta tende a se repetir; acima de 0 a mesma pergunta pode voltar diferente.
O determinismo em 0 é confiável no modelo rápido; nos maiores a substância se
mantém, mas a redação pode variar.

> **Ao trocar os modelos em `backend/app/config.py`:** nos modelos da família 5
> (`claude-sonnet-5`, `claude-opus-5`, `claude-fable-5`) e no Opus 4.7/4.8 o
> parâmetro `temperature` foi removido da API e a chamada retorna erro 400. Se
> trocar por um desses, tire o `temperature` da chain junto.

## Erros comuns

**Token inválido.** Se a chave estiver definida mas incorreta, expirada ou
revogada, o Bedrock responde
`AccessDeniedException: Authentication failed`. Gere uma nova chave no console e
confirme que não há espaços ou quebras de linha no valor copiado.

Sem a variável definida, o `boto3` cai em outra credencial da máquina e o erro
vira `ExpiredTokenException`.

**`MODEL_ID` sem o prefixo do inference profile.** Vários modelos só podem ser
invocados através de um inference profile e recusam o ID base. Por exemplo,
`anthropic.claude-haiku-4-5-20251001-v1:0` falha, enquanto
`global.anthropic.claude-haiku-4-5-20251001-v1:0` funciona. O prefixo varia
conforme o escopo (`global.`, `us.`, `eu.`, `apac.`). O sintoma é
`ValidationException`.

**Modelo sem acesso liberado.** `AccessDeniedException` citando o nome do modelo
significa que ele existe na região mas a conta não tem acesso. Libere em
**Model access** no console.

**Região errada.** O inference profile e o acesso ao modelo são resolvidos por
região. Uma `AWS_REGION` que não oferece o modelo, ou na qual o acesso não foi
habilitado, produz `ResourceNotFoundException` ou `AccessDeniedException`.

**Interface abre mas o envio falha.** O backend não subiu. Confira
`docker compose logs backend`.

**`banco indisponivel no boot` no log do backend.** A API sobe de propósito
mesmo assim: as rotas que não dependem do banco continuam funcionando, e só o
modo de busca falha. Rodando na máquina, a causa quase sempre é `DB_HOST` e
`DB_PORT` no `.env` apontando para o serviço do compose (`db:5432`) em vez da
porta publicada no host.

**O modo de busca não encontra nada.** A base vetorial está vazia — rode
`python -m scripts.indexar_base`. A aba **Base de conhecimento** confirma:
`não indexado` em cada documento significa arquivo no disco sem vetor no banco.

**Porta do banco já em uso.** `bind: address already in use` ao subir o `db`
significa que outro Postgres publica na mesma porta. Troque `DB_PORT` no `.env`.
