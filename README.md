# atende-assistant

Assistente de atendimento ao cliente para um cenário de e-commerce de produtos
congelados (pedidos e políticas), construído de forma incremental.

Neste estado o assistente responde num formato fixo, classifica o assunto de
cada mensagem, consulta pedidos por ferramenta e abre solicitações de troca —
dentro da política, que é verificada no código. O que ele não resolve vai para
uma fila de atendimento humano. Ele ainda não tem acesso aos documentos da
empresa, então não é fonte confiável sobre políticas e prazos que não estejam
no cadastro do pedido.

## Estrutura

```
backend/
  app/
    main.py         API (FastAPI): saude, configuracao, responder,
                    metricas e solicitacoes
    assistente.py   o ciclo de chamada de ferramenta e a saída estruturada
    tools.py        ferramentas: consulta de pedido e abertura de troca
    regras.py       política de troca — a decisão de negócio
    schemas.py      o contrato da resposta (Pydantic)
    dados.py        persistência em arquivo
    config.py       catálogo de modelos, perfis e faixa de temperatura
  dados/
    pedidos.json        cadastro de pedidos usado pelas ferramentas
    atendimentos.jsonl  histórico das mensagens respondidas (gerado)
    solicitacoes.jsonl  fila de trabalho humano (gerado)
  requirements.txt  dependências pinadas
  Dockerfile        imagem baseada em python:3.12-slim
frontend/           interface em React + Vite (imagem node:22-slim)
docker-compose.yml  serviços backend e frontend
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

O suporte a `AWS_BEARER_TOKEN_BEDROCK` no `boto3`/`botocore` existe a partir da
versão `1.39.0`. O piso do `requirements.txt` é mais alto que isso porque
`langchain-aws` exige `boto3>=1.43.32`.

## Execução

### A) Docker

```bash
cp .env.example .env
# preencha AWS_BEARER_TOKEN_BEDROCK no .env
docker compose up --build
```

- Interface: http://localhost:5173
- API: http://localhost:8000/api/saude

### B) Sem Docker

Python 3.12 é a versão recomendada (a mesma da imagem) e 3.10 é o mínimo —
`langchain-aws` exige `>=3.10`. Node 22 para o frontend.

```bash
# backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env .env          # ou exporte as variáveis no shell
uvicorn app.main:app --reload

# frontend, em outro terminal
cd frontend
npm install
npm run dev
```

## Os três controles da interface

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
