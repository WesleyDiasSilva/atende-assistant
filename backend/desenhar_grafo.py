from dotenv import load_dotenv
load_dotenv()
from app.grafo import GRAFO
print(GRAFO.get_graph().draw_mermaid())
