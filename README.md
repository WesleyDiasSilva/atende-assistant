# atende-assistant

Assistente de atendimento ao cliente para um cenário de e-commerce (pedidos e
políticas), construído de forma incremental.

Este estado inicial contém apenas o programa mínimo: uma chamada direta a um
modelo de linguagem no Amazon Bedrock, usando `boto3` e a operação `converse()`.
O programa envia uma pergunta fixa e imprime o texto retornado pelo modelo.
Não há ainda prompt de sistema, histórico de conversa, ferramentas ou base de
conhecimento.

## Arquivos

| Arquivo | Função |
| --- | --- |
| `chamada.py` | Programa único: monta o client, chama `converse()` e imprime a resposta |
| `requirements.txt` | Dependências pinadas (`boto3`, `python-dotenv`) |
| `.env.example` | Modelo das variáveis de ambiente |
| `Dockerfile` | Imagem baseada em `python:3.12-slim` |
| `docker-compose.yml` | Serviço `app` para execução via Docker |

## Configuração

A autenticação usa uma **chave de API do Bedrock** (bearer token), não par de
chaves IAM. Para gerar:

1. Acesse o console da AWS na região desejada (ex.: `us-east-1`).
2. Abra **Amazon Bedrock** → menu lateral **API keys**.
3. Gere uma chave de curta ou longa duração e copie o valor exibido — ele não
   é mostrado novamente.
4. Confirme em **Model access** que o modelo escolhido está habilitado na conta.

A chave é lida pelo `boto3` automaticamente a partir da variável de ambiente
`AWS_BEARER_TOKEN_BEDROCK`. Nenhum código passa credenciais explicitamente.

Variáveis usadas:

| Variável | Descrição |
| --- | --- |
| `AWS_BEARER_TOKEN_BEDROCK` | Chave de API do Bedrock |
| `AWS_REGION` | Região de invocação (padrão sugerido: `us-east-1`) |
| `MODEL_ID` | Identificador do modelo ou do inference profile |

O suporte a `AWS_BEARER_TOKEN_BEDROCK` no `boto3`/`botocore` existe a partir da
versão `1.39.0`; o `requirements.txt` pina uma versão superior a essa.

## Execução

### A) Docker

```bash
cp .env.example .env
# preencha AWS_BEARER_TOKEN_BEDROCK no .env
docker compose run --rm app
```

### B) Python direto

Python 3.12 é a versão recomendada (a mesma da imagem Docker) e 3.10 é o
mínimo: o `boto3` encerra o suporte a versões anteriores, que deixam de receber
atualizações de serviço, correções e patches de segurança.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# preencha AWS_BEARER_TOKEN_BEDROCK no .env
python chamada.py
```

O `.env` é carregado por `python-dotenv`. Alternativamente, exporte as
variáveis diretamente no shell.

## Erros comuns

**Token ausente ou inválido.** Se `AWS_BEARER_TOKEN_BEDROCK` não estiver
definida, o programa aborta antes de qualquer chamada de rede. Se a chave
estiver definida mas incorreta, expirada ou revogada, o Bedrock responde com
erro de autenticação (`UnrecognizedClientException` / `InvalidSignatureException`).
Gere uma nova chave no console e confirme que não há espaços ou quebras de linha
no valor copiado.

**`MODEL_ID` sem o prefixo do inference profile.** Vários modelos só podem ser
invocados através de um inference profile e recusam o ID base. Por exemplo,
`anthropic.claude-haiku-4-5-20251001-v1:0` falha, enquanto
`global.anthropic.claude-haiku-4-5-20251001-v1:0` funciona. O prefixo varia
conforme o escopo (`global.`, `us.`, `eu.`, `apac.`). O sintoma é
`ValidationException` ou `ResourceNotFoundException`.

**Região errada.** O inference profile e o acesso ao modelo são resolvidos por
região. Uma `AWS_REGION` que não oferece o modelo, ou na qual o acesso não foi
habilitado, produz `ResourceNotFoundException` ou `AccessDeniedException`.
Verifique se o prefixo do `MODEL_ID` é compatível com a região configurada e se
a chave de API foi criada para essa mesma região.
