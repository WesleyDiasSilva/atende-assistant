-- Habilita a extensao de vetores (pgvector) no banco do projeto.
--
-- O Postgres executa os arquivos de /docker-entrypoint-initdb.d/ apenas na
-- primeira inicializacao do volume de dados. Volume que ja existe nunca ve este
-- script: quem cobre esse caso e o garantir_extensao_vector() do app/db.py.
CREATE EXTENSION IF NOT EXISTS vector;
