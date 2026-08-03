FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o diretorio inteiro, e nao um arquivo nomeado: assim modulos novos e a
# pasta dados/ entram na imagem sem precisar editar o Dockerfile.
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
