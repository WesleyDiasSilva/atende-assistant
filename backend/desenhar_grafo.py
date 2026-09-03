from dotenv import load_dotenv
load_dotenv()
# Sem checkpointer de proposito: desenhar a topologia nao exige banco de pe.
from app.grafo import compilar_grafo
print(compilar_grafo().get_graph().draw_mermaid())
