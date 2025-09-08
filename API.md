## API do Sistema Cortical Generativo

Base URL padrão (dev): `http://127.0.0.1:8000`

- Documentação interativa (gerada pela FastAPI):
  - Swagger UI: `/docs`
  - ReDoc: `/redoc`

### Variáveis de ambiente relevantes
- `GEMINI_API_KEY` (ou `GOOGLE_API_KEY` como fallback) — obrigatório para o LLM.
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` — opcionais (Neo4j, default off).
- `NEO4J_LOG_INTERACTIONS` — `on` para logar interações, default `off`.

---

### Endpoints

#### Saúde e Status
- GET `/status`
  - Retorna se o sistema está pronto.
  - Exemplo:
```bash
curl "http://127.0.0.1:8000/status" | cat
```

- GET `/ping`
  - Retorna `{ ok: true|false }`.

#### Estatísticas e Exportação
- GET `/stats`
  - Resumo de fragmentos, rede sináptica e aprendizado.

- GET `/export?max_items=<int>`
  - Exporta amostra da memória (itens e metadados).

- GET `/api/synapses?min_weight=<float>&only_consolidated=<bool>&include_nodes=<bool>`
  - Lista arestas (sinapses) e nós (fragmentos) opcionalmente.

#### Chat e Memória de Sessão
- POST `/api/chat`
  - Corpo (JSON):
```json
{ "message": "sua mensagem", "session_id": "opcional" }
```
  - Resposta:
```json
{ "reply": "texto", "meta": { "session_id": "...", "history_size": 3, "intent": {"intent":"ask","confidence":0.7} } }
```
  - Observações:
    - Mantém histórico por `session_id` (janela de 12 mensagens, in-memory).
    - Detecta intenção (LLM + heurística) e ajusta política de escrita (`off/volatile/auto`).

- GET `/api/chat/history?session_id=<id>`
  - Retorna histórico da sessão.

- POST `/api/chat/clear`
  - Corpo: `{ "session_id": "..." }`. Limpa o histórico da sessão.

#### Intenção
- POST `/api/intent`
  - Corpo: `{ "message": "...", "session_id": "opcional" }`
  - Resposta: `{ "intent": "ask|learn|...", "confidence": 0.0-1.0, "entities": [] }`

#### Consulta complexa (pipeline LLM + córtex)
- POST `/query`
  - Corpo: `{ "question": "...", "pre_queries": ["opcional"], "include_analysis": false }`
  - Resposta: `{ "answer": "...", "reflection_data": { ... }, "synaptic_summary": { ... } }`

#### Aprendizado e Esquecimento
- POST `/learn`
  - Corpo:
```json
{ "topic": "Transformers", "complexity": "um resumo conciso para iniciante", "generate_queries": true }
```
  - Integra novo conhecimento ao córtex (via LLM + reflexão).

- POST `/forget`
  - Corpo: `{ "pattern": "texto", "max_remove": 5 }`
  - Remove itens cujo texto contenha o padrão.

#### Neo4j (opcional, default off)
- POST `/neo4j/sync`
  - Sincroniza grafo de fragmentos/sinapses com Neo4j, se variáveis de ambiente estiverem configuradas.

---

### Exemplos rápidos

Chat com sessão:
```bash
curl -X POST "http://127.0.0.1:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"Explique metaplasticidade.","session_id":"demo"}' | cat
```

Ensinar o córtex:
```bash
curl -X POST "http://127.0.0.1:8000/learn" \
  -H "Content-Type: application/json" \
  -d '{"topic":"Transformers em PLN","complexity":"um resumo claro para iniciante"}' | cat
```

Detectar intenção:
```bash
curl -X POST "http://127.0.0.1:8000/api/intent" \
  -H "Content-Type: application/json" \
  -d '{"message":"ensine-me sobre embeddings"}' | cat
```

Exportar memória:
```bash
curl "http://127.0.0.1:8000/export?max_items=200" | cat
```

Esquecer por padrão de texto:
```bash
curl -X POST "http://127.0.0.1:8000/forget" \
  -H "Content-Type: application/json" \
  -d '{"pattern":"Transformers","max_remove":3}' | cat
```

---

### Notas
- Memória de sessão e córtex são mantidos em RAM; persistência opcional pode ser adicionada (ex.: SQLite/Redis/Neo4j).
- `write_mode` é inferido pela intenção no `/api/chat` e não precisa ser enviado pelo cliente.


