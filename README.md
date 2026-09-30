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
    assistente.py   o trabalho: montar o contexto, executar a ferramenta que
                    o modelo pediu, contar token, traduzir erro do Bedrock
    grafo.py        a topologia: os nodes, as arestas e as decisões do fluxo
    retrieval.py    embeddings, chunking, indexação e busca (pgvector)
    documentos.py   leitura e escrita dos .md da base
    db.py           conexão com o Postgres
    tools.py        ferramentas: consulta de pedido e abertura de troca
    regras.py       política de troca — a decisão de negócio
    schemas.py      os contratos da resposta (Pydantic)
    dados.py        persistência em arquivo
    config.py       catálogo de modelos, perfis, modos e temperatura
    log.py          configuração de log da API e dos scripts
  avaliacao/
    casos.json      os casos: pergunta, contexto, critérios e o estado
                    esperado da base — dado, não código
    rodar.py        o runner: executa o grafo por caso, aplica os critérios,
                    compara com a rodada anterior e sai com código de porta
    regua.py        critérios determinísticos (caminho, escopo, ferramenta...)
    juiz.py         critério julgado por modelo, veredito binário com motivo
    sondas.py       callback que observa ferramentas e falhas da execução
    isolamento.py   fila de solicitações temporária durante cada caso
    plataforma.py   vereditos como scores na trace do LangFuse
    relatorio.py    relatório HTML da rodada, reescrito a cada caso
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
| `TOP_K_AMPLIADO` | Quantos trechos a busca devolve quando a auto-correção amplia o alcance |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Tamanho do pedaço e a sobreposição, em caracteres |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` / `LANGSMITH_ENDPOINT` | Tracing no LangSmith. Opcional; `false` ou sem chave, nada é enviado |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | Tracing no LangFuse. Opcional; sem as duas chaves, o callback não é criado |

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

Os documentos da empresa são os `.md` de `backend/dados/base/`. Além do modo em
que nenhum deles chega ao atendente, há dois caminhos pelos quais eles chegam — e
a diferença entre esses dois é o assunto do projeto:

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
cada um, permite acrescentar um `.md` — gravado e indexado na hora, com as etapas
aparecendo na tela — e remover um documento, que apaga o arquivo e os vetores
dele juntos.

### Quem recusa é o prompt, não o score

A busca devolve um score de distância junto de cada trecho, e ele **não** serve
para decidir se a pergunta tem resposta na base. Medido nesta base, as faixas de
score de pergunta coberta e não coberta se sobrepõem: uma pergunta que a base não
cobre pontua melhor que várias que ela cobre.

O "não encontrei na base" vem da instrução que acompanha o contexto, em
`assistente.py`. Ela chega junto com os documentos, e não antes deles: no modo sem
conhecimento não há base contra a qual conferir, então o modelo responde com o que
aprendeu sobre lojas em geral — às vezes um número específico e errado, às vezes
uma resposta vaga, nunca o que esta loja escreveu.

## O fluxo como grafo

O caminho de uma pergunta é declarado em `backend/app/grafo.py`, como um
`StateGraph`: cada etapa é um **node** (uma função que recebe o estado e devolve
o que mudou nele) e cada decisão é uma **aresta condicional** (uma função que só
escolhe o próximo node). O estado é o único canal entre eles.

Antes isso era o corpo de `responder()` — um `if` de modo, um `for` de ferramenta
e um `return` no fim. Funcionava; o que não havia era onde *ler* o fluxo. O que
se ganha declarando não é desempenho, é topologia: o grafo se desenha, e
capacidade nova entra como node em vez de `if` mais fundo.

```bash
docker compose exec backend python desenhar_grafo.py
```

Sai o Mermaid da topologia atual, emitido pelo próprio grafo compilado — é
documentação que não pode ficar desatualizada, porque ela é o código.

Duas coisas que o grafo trouxe e que não existiam antes:

**Triagem na entrada.** Uma chamada curta classifica a mensagem em "assunto da
loja", "pergunta sobre esta conversa" ou "fora do escopo". O que está fora recebe
uma recusa educada e termina ali, sem busca vetorial, sem ciclo de ferramenta e
sem chamada de formato: uma ida ao modelo em vez de três. A classificação é
*fail-open* — na dúvida, segue como se estivesse no escopo. Um falso "fora"
calaria um cliente legítimo; um falso "no escopo" só custa o fluxo normal, que já
sabe recusar o que não está na base.

**Rota conversacional.** "E o nome do cliente?" não tem o que recuperar na base:
o que ela pede já foi dito num turno anterior. Essa mensagem vai para um node que
responde lendo o histórico, sem busca e sem ferramenta — e que, sem conversa
gravada, devolve um texto fixo sem chamar o modelo.

O desempate está escrito no prompt da triagem, e é ele que protege a consulta:
entre "atendimento" e "conversacional", vence "atendimento". Repetir *onde está o
meu pedido 81030?* é pedir o dado outra vez, e o dado pode ter mudado desde a
resposta anterior — a pergunta volta à fonte em vez de ser respondida do
histórico.

**Ciclo de auto-correção.** Quando a resposta admite não ter achado a informação
na base, o fluxo pode **ampliar a busca** — de `TOP_K` para `TOP_K_AMPLIADO`
trechos — e tentar uma segunda vez. Se ainda não achar, o caso vai para a fila
humana com o motivo.

O que o ciclo **não** faz é reescrever a pergunta. A pergunta do cliente é o que
ela é: trocar as palavras dele por outras muda o que foi perguntado, e uma
resposta certa para uma pergunta que ninguém fez é pior do que um "não
encontrei". A hipótese testada na segunda passada é outra — o trecho certo pode
estar na base, só não entre os primeiros colocados.

O ciclo tem teto (uma ampliação) porque ciclo sem teto num grafo é o mesmo
problema do laço sem teto numa função: ele não termina.

**Bifurcação e junção.** *Onde está o meu pedido 81030 e qual a política de
troca?* são duas perguntas de naturezas diferentes: uma só o sistema responde, a
outra só a base. Uma rota única atende uma das duas e responde metade com toda a
confiança.

A triagem reconhece esse caso e manda a pergunta para `bifurcar`, que a separa em
duas metades. De lá saem duas arestas, e o LangGraph executa os dois ramos **no
mesmo passo**: `ramo_do_pedido` consulta a ferramenta, `ramo_da_regra` busca nos
documentos. `juntar` recebe aresta dos dois e por isso só é agendado quando os
dois concluíram — a espera é a topologia, não um contador no código.

A junção é uma ida ao modelo, e não uma concatenação: as duas metades podem se
condicionar. Um pedido cancelado muda o que a política de troca permite, e grudar
os dois parágrafos entregaria ao cliente uma regra que não vale para o caso dele.

Cada ramo recebe só a sua metade. A busca do ramo da regra não leva o número do
pedido junto — número não descreve assunto, e deslocaria o ranking.

## O caminho percorrido

Cada resposta traz o campo `trajetoria`: os nodes por onde a pergunta passou, na
ordem em que concluíram. A interface o mostra abaixo da bolha, junto dos trechos
recuperados. Duas respostas parecidas podem ter vindo por caminhos diferentes —
uma do histórico, outra da base — e é o caminho que distingue as duas.

O campo tem **mais de um escritor no mesmo turno**: todo node acrescenta o
próprio nome. Num campo de sobrescrita o último apagaria os anteriores, e sobraria
um nome só. Por isso ele é declarado com um **reducer**
(`Annotated[list[str], acumular_trajetoria]`), como o `historico` — a diferença é
que o rastro recomeça a cada turno, e o histórico não.

O que ele registra é nome de node e ordem. Nada mais.

É no caso composto que o rastro mostra mais: os dois ramos aparecem lado a lado,
na ordem em que terminaram — que não é a ordem em que começaram, porque eles
começaram juntos.

## Duas formas de escrita no mesmo estado

O fan-out coloca as duas lado a lado, e elas pedem tratamentos opostos.

O **rastro acumula**: os dois ramos escrevem em `trajetoria` no mesmo passo, e o
que se quer é a soma. Isso é reducer — e sem ele o LangGraph nem aceitaria os
dois escritores concorrentes.

As **respostas parciais não acumulam**: `resposta_do_pedido` e
`resposta_da_regra` são coisas diferentes, uma vinda do sistema e a outra da
base. Cada uma tem um escritor só, e vai num campo próprio, sem reducer. Somá-las
numa lista entregaria a `juntar` duas respostas sem etiqueta, e ele precisa saber
qual é qual: uma é dado daquele cliente, a outra é política da loja.

Reducer resolve acúmulo. Composição pede critério, e o critério mora num node.

## Os cinco controles da interface

Na mesma ordem em que aparecem no painel.

**Modelo** — troca qual modelo responde. Os três configurados foram verificados
nesta conta, com acesso liberado e `temperature` aceito. O custo é indicado em
ordem relativa; a latência real de cada resposta aparece no rodapé da mensagem.

**Conhecimento** — quanto da base entra no contexto: nada, tudo, ou os trechos
recuperados. É o único destes quatro que muda *o que* o atendente sabe; os outros
três mudam *como* ele responde.

O rodapé de cada resposta mostra os **tokens de entrada da primeira ida ao
modelo** — o tamanho do prompt, que é o que o modo de conhecimento controla. Não
é o custo total da pergunta: cada volta do ciclo de ferramenta e a chamada final
que exige o formato reenviam a conversa inteira. O número serve para comparar os
modos entre si, e a diferença entre eles é grande o bastante para o argumento.

**Perfil de atendimento** — troca o tom da resposta. Muda *como* o atendente
escreve, não *o que* ele sabe.

**Temperatura** — de 0 a 1. Em 0 o modelo escolhe sempre o token mais provável e
a resposta tende a se repetir; acima de 0 a mesma pergunta pode voltar diferente.
O determinismo em 0 é confiável no modelo rápido; nos maiores a substância se
mantém, mas a redação pode variar.

**Auto-correção** — liga o ciclo que amplia a busca quando a resposta não se
sustentou na base. Não muda o que o atendente sabe nem como ele escreve: muda o
que o fluxo *faz* diante de uma resposta que não se sustentou. Só tem efeito nos
modos de busca — é lá que existe um alcance para ampliar. Desligada por padrão,
porque o ciclo custa uma busca e duas idas ao modelo a mais.

> **Ao trocar os modelos em `backend/app/config.py`:** nos modelos da família 5
> (`claude-sonnet-5`, `claude-opus-5`, `claude-fable-5`) e no Opus 4.7/4.8 o
> parâmetro `temperature` foi removido da API e a chamada retorna erro 400. Se
> trocar por um desses, tire o `temperature` da chain junto.

## Observabilidade

Cada pergunta pode virar uma **trace** — a árvore de tudo o que a execução fez:
os nodes do grafo, cada ida ao Bedrock com tokens e custo, cada ferramenta.
Duas plataformas, ligadas lado a lado, e as duas **fail-open**: sem chave, o
atendimento responde igual e nada sai da máquina.

- **LangSmith é config, não código.** O SDK vem com o `langchain-core`; basta
  `LANGSMITH_TRACING=true` e `LANGSMITH_API_KEY` no `.env` e recriar o
  conteiner (`docker compose up -d backend` — o `--reload` relê código, não
  variável de ambiente).
- **LangFuse é um callback.** `backend/app/observabilidade.py` cria um
  `CallbackHandler` (SDK 4.x) por pergunta e o põe em `config["callbacks"]` do
  `invoke()` do grafo; o config se propaga para todos os nodes. O
  `langfuse_session_id` é o `thread_id` da conversa: os turnos de uma conversa
  aparecem juntos em **Sessions**. Modo, modelo e perfil vão como tags.
- **Groundedness.** O node `avaliar_groundedness` (depois de `formalizar`)
  embute a resposta e os trechos recuperados com o mesmo Titan da busca e guarda
  o maior cosseno. Só nos modos de busca — nos outros o valor é `null`. Depois do
  `invoke()`, o score vai para a trace da própria execução
  (`create_score(trace_id=handler.last_trace_id, name="groundedness")`) e
  aparece no rodapé da resposta na interface. Score baixo não é resposta errada:
  uma resposta que veio da ferramenta de pedido não se parece com a base, e o
  número diz exatamente isso.

## Avaliação

Uma suíte que responde "passou ou não passou" para um conjunto fixo de perguntas,
a cada mudança. Vive em `backend/avaliacao/`, fora de `app/`: ela importa o
grafo, invoca e observa — não acrescenta uma linha ao fluxo que mede.

```bash
docker compose exec backend python -m avaliacao.rodar             # todos os casos
docker compose exec backend python -m avaliacao.rodar --rapido    # o conjunto curto
docker compose exec backend python -m avaliacao.rodar --caso regra-prazo-troca
docker compose exec backend python -m avaliacao.rodar --caso regra-prazo-troca --repeticoes 5
```

Sempre `python -m avaliacao.rodar`, de dentro do conteiner: `python
avaliacao/rodar.py` não encontra o pacote `app`.

**Casos são dado.** Cada caso em `casos.json` tem pergunta, histórico quando
precisa de conversa anterior, entradas (modo, auto-correção, memória) e
critérios. A `descricao` registra *por que* aquela é a expectativa.

**Duas camadas de critério.** Tudo o que tem gabarito é medido em código
(`regua.py`), sem chamada a modelo:

| Critério | De onde vem o valor |
|---|---|
| `rota` | a trajetória que cada node escreve, comparada como **subsequência** |
| `nao_passa_por` | a mesma trajetória: nodes que não podem aparecer |
| `escopo` | a classificação da triagem (`Literal` fechado) |
| `fontes` | o documento esperado está entre os **recuperados** pela busca |
| `recusa` | `precisa_de_humano` com trecho recuperado — falha da base não conta |
| `texto_fixo` | os textos que o sistema escreve sem modelo |
| `ferramenta` | nome e argumentos executados, lidos pelo callback |
| `regra` | a saída de `regras.impedimento_para_troca` chegou ao fluxo |
| `tentativas` | o contador do ciclo de ampliação |
| `ramos` | as duas respostas parciais da pergunta composta |
| `bifurcacao` | o número do pedido fica fora da metade da regra |
| `contem` | o texto afirma o que o gabarito diz |
| `tipo` | o assunto declarado (`Enum` fechado) |

O que não tem gabarito vai para o **juiz** (`juiz.py`): um modelo declarado no
próprio módulo, temperatura zero, que recebe pergunta, contexto e resposta — e
não o resultado da régua. O critério é uma pergunta fechada que descreve um
defeito; a saída é `aprovado` ou `reprovado` com o motivo.

O `groundedness` aparece em cada caso e **nunca decide**: é um cosseno, cego à
negação, e dá nota baixa à recusa correta.

**Vermelho esperado.** Um caso com `esperado_vermelho` documenta um defeito
conhecido (`motivo_do_vermelho`). Reprovar é o esperado; passar é que é notícia.

**Delta.** Cada rodada compara o resultado por caso com a anterior, gravada em
`avaliacao/.ultima-rodada.json` (fora do git), e imprime `antes → agora` e quem
regrediu. Só entram na comparação casos medidos nas duas rodadas. Um vermelho
esperado que passou (ou voltou a reprovar) não é recuperação nem regressão: sai
numa linha própria, `vermelho esperado mudou`.

**Código de saída.** `0` quando tudo está como os casos declaram; `1` quando há
notícia (um verde que reprovou, ou um vermelho esperado que passou); `2` quando
parte da rodada não pôde ser medida, ou quando a base não está no estado
esperado.

**O que a suíte isola.** A fila de solicitações vira um arquivo temporário por
caso — abrir troca grava, e a segunda troca do mesmo pedido é recusada pela
regra. Uma chamada ao modelo que falhou durante o caso (limite de requisições,
credencial) é refeita; persistindo, o caso sai como **não avaliado
(infraestrutura)**, fora do placar e do delta. `AVALIACAO_PAUSA` controla a
pausa entre casos (padrão 2 s): o Bedrock limita requisições por minuto, e uma rodada é uma rajada.

**Pré-condição da base.** `base_esperada` em `casos.json` declara os documentos
e o total de chunks. Base diferente interrompe a rodada com a divergência; para
medir assim mesmo, `--ignorar-base`. Documento apagado pelo painel sai também do
disco: volta com `git checkout -- backend/dados/base/<arquivo>` (ou reenviado
pelo painel) e `python -m scripts.indexar_base`. `TOP_K` pode ser trocado só para a rodada:
`docker compose exec -e TOP_K=1 backend python -m avaliacao.rodar`.

**Relatório.** A rodada é escrita em `backend/avaliacao/relatorio/rodada.html` a
cada caso; aberto no navegador, ele se atualiza sozinho enquanto a rodada corre.

**Plataforma.** Com o LangFuse configurado, cada caso vira uma trace com o id do
caso como nome, a rodada como sessão, e os vereditos como scores `regua` e `juiz`
(o motivo vai no comentário).

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
