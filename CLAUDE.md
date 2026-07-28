# atende-assistant

Assistente de atendimento ao cliente (pedidos e políticas de e-commerce),
construído de forma incremental.

## Convenção de branches

O repositório organiza a evolução do código em pares de branches
`aulaXX-inicio` e `aulaXX-fim`, com numeração sequencial (`aula01`, `aula02`,
...):

- `aulaXX-inicio` marca o estado do código antes das mudanças de `aulaXX`.
- `aulaXX-fim` marca o estado do código depois das mudanças de `aulaXX`.
- `aulaXX-inicio` é sempre idêntica a `aula(XX-1)-fim` — o mesmo commit, sem
  nenhuma diferença de conteúdo.
- Nunca implementar em uma branch `-inicio`: ela é apenas um marcador de ponto
  de partida. A implementação acontece em `main`, e a branch `-fim` aponta para
  o commit resultante.
- `main` acompanha sempre a `-fim` mais recente.

## Estado atual

- `aula01-inicio`: estrutura do projeto, sem a lógica de chamada.
- `aula01-fim`: chamada mínima ao modelo via Bedrock implementada.
