
import asyncio
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import os

# Carrega variáveis do arquivo .env (se existir)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
from pydantic import BaseModel
import uvicorn

# Importe as classes do seu arquivo principal
# É crucial que o main.py esteja no mesmo diretório ou em um caminho acessível
from main import EnhancedSynapticCorticalSystem, GenerativeCorticalSystem

# --- Modelos de Dados Pydantic ---
# Usamos Pydantic para validação automática dos dados de entrada da API.
class QueryRequest(BaseModel):
    """Modelo para o corpo da requisição de uma consulta."""
    question: str
    pre_queries: list[str] | None = None
    include_analysis: bool = False

class ApiResponse(BaseModel):
    """Modelo para a resposta da API."""
    answer: str
    reflection_data: dict | None = None
    synaptic_summary: dict | None = None

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    write_mode: str | None = None  # "off" | "volatile" | "auto"

class ChatResponse(BaseModel):
    reply: str
    meta: dict | None = None

class ChatHistoryRequest(BaseModel):
    session_id: str

class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, str]]

class IntentRequest(BaseModel):
    message: str
    session_id: str | None = None

class IntentResponse(BaseModel):
    intent: str
    confidence: float
    entities: List[str] | None = None

class SynapseEdge(BaseModel):
    u: int
    v: int
    w: float
    volatile: bool
    usage: int
    last_update: int | None = None
    weights: Dict[str, float] | None = None

class SynapseListResponse(BaseModel):
    total: int
    edges: List[SynapseEdge]
    nodes: Dict[int, Dict[str, Any]] | None = None

class LearnRequest(BaseModel):
    """Modelo para o corpo da requisição de aprendizado."""
    topic: str
    complexity: str = "um resumo conciso para um iniciante"
    generate_queries: bool = True

class LearnResponse(BaseModel):
    """Modelo para a resposta do endpoint de aprendizado."""
    message: str

class Neo4jSyncResponse(BaseModel):
    ok: bool
    detail: str | None = None

# --- Inicialização do Aplicativo FastAPI ---
app = FastAPI(
    title="API do Sistema Cortical Generativo",
    description="Uma interface para interagir com um sistema de memória cortical e um LLM.",
    version="1.0.0"
)

# CORS (libera acesso do app Vite React em desenvolvimento)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajuste para domínios específicos em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Variáveis Globais ---
# Essas variáveis manterão nosso sistema treinado em memória enquanto a API estiver rodando.
generative_system_instance: GenerativeCorticalSystem | None = None
cortical_system_instance: EnhancedSynapticCorticalSystem | None = None
# Instâncias por sessão (personalização por usuário)
session_cortex: Dict[str, EnhancedSynapticCorticalSystem] = {}
session_generative: Dict[str, GenerativeCorticalSystem] = {}
is_ready = False
llm_ready = False

# --- Memória de Conversa (Sessão) ---
# Armazena histórico por sessão em memória de processo
session_histories: Dict[str, List[Dict[str, str]]] = {}
MAX_HISTORY_MESSAGES = 12  # janela deslizante por sessão

def _sanitize_text(text: str, max_len: int = 2000) -> str:
    text = (text or "").trim() if hasattr(text, 'trim') else (text or "").strip()
    if len(text) > max_len:
        return text[:max_len]
    return text

def _get_session_id(sid: str | None) -> str:
    return sid.strip() if sid and sid.strip() else "default"

def _add_history(session_id: str, role: str, content: str) -> None:
    history = session_histories.setdefault(session_id, [])
    history.append({"role": role, "content": _sanitize_text(content)})
    if len(history) > MAX_HISTORY_MESSAGES:
        del history[: len(history) - MAX_HISTORY_MESSAGES]

def _build_pre_queries_from_history(session_id: str) -> List[str]:
    history = session_histories.get(session_id, [])
    lines: List[str] = []
    for h in history[-6:]:  # últimas 6 mensagens (user/assistant)
        prefix = "Usuário" if h.get("role") == "user" else "Assistente"
        lines.append(f"{prefix}: {h.get('content','')}")
    return lines

# --- Analisador de Intenção ---
INTENT_LABELS = [
    "ask", "learn", "learn_about", "smalltalk", "greeting", "forget", "stats", "export", "unknown"
]

def _heuristic_intent(message: str) -> Dict[str, Any]:
    text = (message or "").strip()
    lowered = text.lower()
    entities: List[str] = []
    # Cumprimentos e smalltalk
    if any(x in lowered for x in ["oi", "ola", "olá", "opa", "bom dia", "boa tarde", "boa noite"]):
        return {"intent": "greeting", "confidence": 0.85, "entities": entities}
    if any(x in lowered for x in ["kkk", "rs", "haha", "tudo bem", "como vai"]):
        return {"intent": "smalltalk", "confidence": 0.8, "entities": entities}
    # Forget
    if any(x in lowered for x in ["esquece", "esquecer", "forget"]):
        return {"intent": "forget", "confidence": 0.7, "entities": entities}
    # Stats / export
    if any(x in lowered for x in ["estat", "stats", "sinapse", "grafo"]):
        return {"intent": "stats", "confidence": 0.6, "entities": entities}
    if any(x in lowered for x in ["export", "baixar", "dump", "json"]):
        return {"intent": "export", "confidence": 0.6, "entities": entities}
    # Learn-about (conteúdo novo a ser aprendido)
    if any(phrase in lowered for phrase in ["aprenda sobre", "aprender sobre", "aprenda a respeito", "aprender a respeito", "estude sobre", "pesquise sobre"]):
        # extrai possível entidade após 'sobre'
        try:
            after = lowered.split("sobre", 1)[1].strip()
            if after:
                entities.append(after[:80])
        except Exception:
            pass
        return {"intent": "learn_about", "confidence": 0.9, "entities": entities}
    # Learn / ensinar genérico
    if any(x in lowered for x in ["ensina", "ensinar", "aprender", "learn", "estude", "estudar", "ensine"]):
        return {"intent": "learn", "confidence": 0.7, "entities": entities}
    # Pergunta
    if "?" in text or any(x in lowered.split()[:3] for x in ["como", "o que", "qual", "quando", "por que", "porque"]):
        return {"intent": "ask", "confidence": 0.65, "entities": entities}
    return {"intent": "ask", "confidence": 0.5, "entities": entities}

def _extract_json(s: str) -> Dict[str, Any] | None:
    try:
        s = (s or "").strip()
        # tenta bloco ```json ... ```
        if "```" in s:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start:end+1])
        # tenta JSON bruto
        if s.startswith("{") and s.endswith("}"):
            return json.loads(s)
        # fallback: substring
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end+1])
    except Exception:
        return None
    return None

def _analyze_intent_sync(message: str, session_id: str) -> Dict[str, Any]:
    # LLM primeiro, heurístico como fallback
    if llm_ready and generative_system_instance is not None:
        try:
            prompt = (
                "Classifique a intenção da mensagem do usuário em uma destas categorias: "
                + ", ".join(INTENT_LABELS)
                + ".\nRetorne APENAS um bloco final ```json com: {\"intent\":\"...\", \"confidence\":0.0, \"entities\":[\"...\"]}. Nada após o bloco.\n\n"
                + f"Mensagem: '{message}'"
            )
            # Chamada direta ao LLM (sem RAG) para evitar poluir o córtex
            answer = generative_system_instance.reasoning_model.generate_content(prompt).text
            parsed = _extract_last_json_block(answer)
            if isinstance(parsed, dict) and isinstance(parsed.get("intent"), str):
                intent = parsed.get("intent").strip().lower()
                if intent not in INTENT_LABELS:
                    intent = "unknown"
                conf = float(parsed.get("confidence", 0.7))
                ents = parsed.get("entities") or []
                if not isinstance(ents, list):
                    ents = []
                return {"intent": intent, "confidence": conf, "entities": ents}
        except Exception:
            pass
    # Fallback
    return _heuristic_intent(message)

def _extract_last_json_block(text: str) -> Dict[str, Any] | None:
    try:
        s = (text or "").strip()
        last_fence = s.rfind("```json")
        if last_fence == -1:
            return _extract_json(s)
        closing = s.find("```", last_fence + 6)
        if closing == -1:
            return None
        block = s[last_fence + 6:closing].strip()
        return json.loads(block)
    except Exception:
        return None

def _curriculum_prompt(topic: str) -> str:
    t = (topic or "").strip()
    return (
        f"Você é um tutor. Gere um mini currículo estruturado sobre '{t}' (português):\n"
        f"1) Explique em 4-6 frases curtas e práticas (para iniciantes).\n"
        f"2) Ao final, RETORNE APENAS um bloco ```json com o formato:\n"
        f"{{\n  \"teach_seed\": [\n    {{\"kind\":\"fact\",\"content\":{{\"text\":\"...\"}},\"confidence\":0.9}}\n  ],\n  \"next_prompts\": [\"...\"]\n}}\n"
        f"Nada após o bloco JSON."
    )

# --- Auto-learn a partir do conteúdo do usuário ---
def _extract_user_facts(raw: str) -> List[str]:
    lines = [l.strip(" \t-•*\u2022") for l in (raw or "").splitlines()]
    facts: List[str] = []
    for l in lines:
        if not l:
            continue
        if l.lower().startswith(("fato:", "definição:", "definicao:", "conceito:", "nota:")):
            facts.append(l.split(":", 1)[1].strip() or l)
            continue
        # bullets e listas numeradas
        if l[0:2].isdigit() or any(l.startswith(p) for p in ("- ", "* ", "• ")):
            facts.append(l.lstrip("0123456789.) "))
            continue
        # frases longas podem ser ignoradas aqui
    # normaliza e limita
    facts = [f for f in (s.strip() for s in facts) if len(f) >= 8]
    # dedup simples
    seen = set()
    uniq: List[str] = []
    for f in facts:
        key = f.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(f)
    return uniq[:5]

session_auto_learn_gate: Dict[str, List[int]] = {}

def _rate_limited_auto_learn(session_id: str, max_per_5min: int = 3) -> bool:
    import time
    now = int(time.time())
    window_start = now - 300
    arr = session_auto_learn_gate.setdefault(session_id, [])
    arr[:] = [t for t in arr if t >= window_start]
    if len(arr) >= max_per_5min:
        return False
    arr.append(now)
    return True

# --- Eventos de Ciclo de Vida do FastAPI ---
@app.on_event("startup")
async def startup_event():
    """
    Esta função é executada uma vez quando a API é iniciada.
    Ela treina o sistema cortical e o prepara para uso.
    """
    global generative_system_instance, cortical_system_instance, is_ready, llm_ready
    
    print("Iniciando o treinamento do sistema cortical... Este processo pode levar alguns minutos.")
    
    # O loop de eventos do asyncio pode interferir com algumas bibliotecas que não são
    # totalmente compatíveis com async. Executamos a função de treinamento em um 
    # thread separado para evitar bloqueios.
    # Garante mapeamento alternativo GOOGLE_API_KEY -> GEMINI_API_KEY, se necessário
    if not os.getenv("GEMINI_API_KEY") and os.getenv("GOOGLE_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY") or ""

    # Cold start: não carregamos corpus global; criamos um córtex vazio e inicializamos o LLM se possível
    global cortical_system_instance, generative_system_instance, llm_ready
    cortical_system_instance = EnhancedSynapticCorticalSystem(n_subspaces=4, max_hierarchy_levels=3)
    try:
        # Garante mapeamento alternativo GOOGLE_API_KEY -> GEMINI_API_KEY, se necessário
        if not os.getenv("GEMINI_API_KEY") and os.getenv("GOOGLE_API_KEY"):
            os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY") or ""
        generative_system_instance = GenerativeCorticalSystem(cortical_system_instance)
        llm_ready = True
        print("LLM inicializado em modo cold start (córtex vazio global).")
    except Exception as e:
        generative_system_instance = None
        llm_ready = False
        print(f"LLM indisponível no startup (seguirá com fallback RAG por sessão). Detalhe: {e}")

    is_ready = True
    print("="*80)
    print(">>> Cold start: memórias vazias por sessão serão criadas sob demanda. <<<")
    print("="*80)

def train_cortical_system():
    """Função síncrona que executa o treinamento do sistema."""
    global generative_system_instance, cortical_system_instance, llm_ready
    
    # O código abaixo é adaptado da função demo_enhanced_synaptic_system() do seu main.py
    # Ele apenas treina o sistema e o armazena na variável global.
    
    corpus = [
        # O mesmo corpus usado no seu main.py
        "Algoritmos de machine learning revolucionam a análise de dados complexos.",
        "Deep learning permite reconhecimento avançado de padrões visuais.",
        "Redes neurais artificiais simulam o processamento cerebral humano.",
        "Inteligência artificial automatiza processos empresariais complexos.",
        "Big data requer ferramentas de processamento distribuído eficiente.",
        "Cloud computing oferece escalabilidade infinita para aplicações modernas.",
        "Processamento de linguagem natural facilita interação homem-máquina.",
        "Plasticidade sináptica permite adaptação contínua do sistema nervoso.",
        "Metaplasticidade regula a capacidade de mudança das conexões neurais.",
        "Potenciação a longo prazo fortalece sinapses com uso repetido.",
        "Depressão a longo prazo enfraquece conexões pouco utilizadas.",
        "Homeostase sináptica mantém equilíbrio da atividade neural.",
        "Neurogênese adulta adiciona novos neurônios ao sistema existente.",
        "Florestas tropicais são pulmões vitais do planeta Terra.",
        "Biodiversidade garante equilíbrio dos ecossistemas naturais complexos.",
        "Mudanças climáticas afetam habitats ao redor do mundo inteiro.",
        "Conservação ambiental protege espécies em extinção crítica.",
        "Energias renováveis reduzem pegada de carbono significativamente.",
        "Ecossistemas marinhos enfrentam poluição plástica crescente.",
        "Agricultura sustentável preserva solos para futuras gerações.",
        "Gestão eficaz coordena equipes multidisciplinares produtivas.",
        "Liderança inspiradora motiva colaboradores criativos e dedicados.",
        "Planejamento estratégico define metas organizacionais claras e mensuráveis.",
        "Inovação empresarial gera vantagem competitiva sustentável e duradoura.",
        "Cultura organizacional influencia performance dos times significativamente.",
        "Transformação digital acelera processos internos das empresas.",
        "Empreendedorismo social combina lucro com impacto positivo.",
        "Exercícios regulares fortalecem sistema cardiovascular humano eficientemente.",
        "Alimentação balanceada fornece nutrientes essenciais ao organismo.",
        "Sono reparador restaura funções cognitivas cerebrais importantes.",
        "Medicina preventiva reduz riscos de doenças crônicas graves.",
        "Bem-estar mental impacta qualidade de vida profundamente.",
        "Meditação mindfulness diminui estresse e ansiedade diários.",
        "Atividade física regular previne doenças metabólicas comuns.",
        "Pedagogia moderna adapta-se às necessidades individuais dos estudantes.",
        "Tecnologia educacional personaliza experiências de aprendizado únicas.",
        "Metodologias ativas engajam estudantes no processo educativo.",
        "Educação continuada desenvolve competências profissionais essenciais.",
        "Interdisciplinaridade conecta diferentes áreas do conhecimento humano.",
        "Ensino híbrido combina presencial e digital eficientemente.",
        "Gamificação torna aprendizagem mais envolvente e divertida.",
        "Inteligência artificial bio-inspirada imita plasticidade neural adaptativa.",
        "Aprendizado contínuo fortalece conexões entre conhecimentos diversos.",
        "Adaptabilidade organizacional espelha flexibilidade sináptica cerebral.",
        "Redes colaborativas emergem através de conexões repetidas e fortalecidas.",
        "Consolidação de memórias requer repetição e reforço contextual.",
        "Sistemas auto-organizados desenvolvem padrões através de uso frequente."
    ]
    
    # 1. Inicializa e processa o corpus com o sistema cortical
    # Mantido para compatibilidade, mas não usado no cold start
    cortical_system = EnhancedSynapticCorticalSystem(n_subspaces=4, max_hierarchy_levels=3)
    cortical_system_instance = cortical_system
    
    # 2. Realiza uma série de consultas para "aquecer" e estabilizar a rede sináptica
    queries = [
        "Como redes neurais artificiais simulam plasticidade cerebral?",
        "Que papel tem a metaplasticidade no aprendizado adaptativo?",
        "Gestão de equipes requer adaptabilidade constante e flexibilidade.",
        "Plasticidade sináptica permite adaptação contínua do sistema nervoso.",
        "Como algoritmos de machine learning imitam processos neurais?",
        "Liderança adaptativa espelha flexibilidade das conexões cerebrais.",
        "Sustentabilidade ambiental requer inovação tecnológica constante.",
        "Educação personalizada utiliza inteligência artificial avançada.",
        "Bem-estar mental impacta performance organizacional significativamente.",
        "Redes neurais artificiais simulam processamento cerebral humano.",
        "Metaplasticidade regula capacidade de mudança das conexões neurais.",
        "Sistemas auto-organizados desenvolvem padrões através de uso frequente.",
        "Aprendizado contínuo fortalece conexões entre conhecimentos diversos.",
        "Inteligência artificial bio-inspirada imita plasticidade neural adaptativa.",
        "Adaptabilidade organizacional espelha flexibilidade sináptica cerebral.",
        "Como metodologias ágeis aplicam princípios de plasticidade neural?",
        "Gamificação educacional utiliza reforço similar à potenciação sináptica.",
        "Transformação digital requer adaptabilidade inspirada em neuroplasticidade."
    ]
    
    for query in queries:
        cortical_system.enhanced_query_processing(query)
    
    # 3. Inicializa o sistema generativo com o córtex já treinado
    try:
        generative_system_instance = GenerativeCorticalSystem(cortical_system)
        llm_ready = True
    except (ImportError, NameError) as e:
        print(f"ERRO CRÍTICO: Não foi possível inicializar o GenerativeCorticalSystem. {e}")
        print("Verifique se a biblioteca 'google-generativeai' está instalada e a API Key configurada.")
        generative_system_instance = None
        llm_ready = False
    except Exception as e:
        print(f"ERRO CRÍTICO durante a inicialização do sistema generativo: {e}")
        generative_system_instance = None
        llm_ready = False


# --- Endpoints da API ---

@app.get("/status")
def get_status():
    """Endpoint para verificar se a API está pronta para receber consultas."""
    if is_ready:
        return {"status": "ready", "message": "O sistema está treinado e pronto."}
    return {"status": "loading", "message": "O sistema está sendo treinado. Por favor, aguarde."}

@app.get("/ping")
def ping():
    if is_ready and generative_system_instance is not None:
        return {"ok": True}
    return {"ok": False}


@app.get("/stats")
def get_stats():
    """Resumo rápido das estatísticas do córtex e rede sináptica."""
    if not is_ready or generative_system_instance is None:
        raise HTTPException(status_code=503, detail="Sistema não pronto.")
    try:
        cortex = generative_system_instance.cortical_memory
        analysis = cortex.comprehensive_analysis()
        return {
            "fragment_stats": analysis.get('fragment_stats', {}),
            "network_stats": analysis.get('network_stats', {}),
            "synaptic_stats": analysis.get('synaptic_stats', {}),
            "learning_stats": analysis.get('learning_stats', {}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao coletar estatísticas: {e}")

@app.get("/export")
def export_state(max_items: int = 500):
    if not is_ready or generative_system_instance is None:
        raise HTTPException(status_code=503, detail="Sistema não pronto.")
    try:
        cortex = generative_system_instance.cortical_memory
        data = cortex.export_memory(max_items=max_items)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao exportar memória: {e}")

class ForgetRequest(BaseModel):
    pattern: str
    max_remove: int | None = 10

@app.post("/forget")
def forget_items(req: ForgetRequest):
    if not is_ready or generative_system_instance is None:
        raise HTTPException(status_code=503, detail="Sistema não pronto.")
    try:
        cortex = generative_system_instance.cortical_memory
        result = cortex.forget_items_by_text(req.pattern, max_remove=req.max_remove or 10)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao esquecer itens: {e}")


@app.post("/query", response_model=ApiResponse)
async def process_query(request: QueryRequest):
    """
    Endpoint principal para processar uma consulta do usuário.
    Recebe uma pergunta e retorna a resposta gerada pelo sistema.
    """
    if not is_ready or generative_system_instance is None:
        raise HTTPException(
            status_code=503, 
            detail="O sistema ainda não está pronto. Por favor, tente novamente em alguns instantes."
        )

    try:
        # A lógica de processamento da consulta é assíncrona por natureza no FastAPI
        # Executamos a função síncrona do seu sistema em um executor para não bloquear o loop de eventos.
        loop = asyncio.get_event_loop()
        answer, reflection_data = await loop.run_in_executor(
            None, 
            generative_system_instance.process_complex_query, 
            request.question, 
            request.pre_queries
        )
        summary = None
        if request.include_analysis:
            # coleta um resumo sináptico do córtex
            summary = generative_system_instance.cortical_memory._get_synaptic_network_stats()
        
        return ApiResponse(answer=answer, reflection_data=reflection_data if request.include_analysis else None, synaptic_summary=summary)
        
    except Exception as e:
        # Captura exceções que possam ocorrer durante o processamento
        print(f"Erro ao processar a consulta: {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno: {e}")

@app.post("/learn", response_model=LearnResponse)
async def learn_topic(request: LearnRequest):
    """
    Endpoint para instruir o sistema a aprender sobre um novo tópico.
    O sistema usará seu LLM interno para gerar conhecimento e o integrará à sua memória cortical.
    """
    if not is_ready or generative_system_instance is None:
        raise HTTPException(
            status_code=503, 
            detail="O sistema ainda não está pronto. Por favor, tente novamente em alguns instantes."
        )

    try:
        loop = asyncio.get_event_loop()
        # Executa a função de aprendizado em um thread separado para não bloquear a API
        # Usa o LLM via o próprio sistema para gerar e integrar conhecimento
        prompt = f"Gere 3 a 5 sentenças claras sobre '{request.topic}' com {request.complexity}. Saída em lista simples."
        # Reutiliza o pipeline interno chamando process_complex_query para gatilhar reflexão e escrita
        question = prompt if request.generate_queries else f"Explique: {request.topic}"
        answer, reflection = await loop.run_in_executor(
            None,
            generative_system_instance.process_complex_query,
            question,
            None
        )
        return LearnResponse(message=f"Aprendizado integrado. Resumo: {answer[:200]}...")

    except Exception as e:
        print(f"Erro ao aprender sobre o tópico '{request.topic}': {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno durante o aprendizado: {e}")


@app.post("/neo4j/sync", response_model=Neo4jSyncResponse)
def neo4j_sync():
    if not is_ready or generative_system_instance is None:
        raise HTTPException(status_code=503, detail="Sistema não pronto.")
    try:
        cortex = generative_system_instance.cortical_memory
        ok = cortex.push_to_neo4j()
        if ok:
            return Neo4jSyncResponse(ok=True, detail="Sincronização concluída")
        return Neo4jSyncResponse(ok=False, detail="Sincronização falhou (ver logs)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao sincronizar com Neo4j: {e}")


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):
    if not is_ready:
        raise HTTPException(status_code=503, detail="Sistema carregando.")
    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message requerido")
    try:
        loop = asyncio.get_event_loop()
        session_id = _get_session_id(req.session_id)
        # Inicializa córtex/LLM da sessão sob demanda
        if session_id not in session_cortex:
            session_cortex[session_id] = EnhancedSynapticCorticalSystem(n_subspaces=4, max_hierarchy_levels=3)
        if llm_ready and session_id not in session_generative and generative_system_instance is not None:
            try:
                session_generative[session_id] = GenerativeCorticalSystem(session_cortex[session_id])
            except Exception:
                session_generative[session_id] = None  # type: ignore

        # Atualiza histórico com a mensagem do usuário
        _add_history(session_id, "user", msg)
        # Auto-aprendizado de fatos do usuário (volátil, rate-limited)
        try:
            facts = _extract_user_facts(msg)
            if facts and _rate_limited_auto_learn(session_id):
                cortex_local = session_cortex.get(session_id)
                if cortex_local is not None:
                    md = [{"source": "user_fact", "volatile": True} for _ in facts]
                    # limite duro de 3 por interação
                    cortex_local.add_new_knowledge(facts[:3], metadata_list=md[:3], max_to_add=3)
        except Exception:
            pass
        # Análise de intenção (LLM + heurística)
        intent = await loop.run_in_executor(None, _analyze_intent_sync, msg, session_id)
        if intent.get("intent") in ("greeting", "smalltalk"):
            return ChatResponse(reply="Olá! Posso ajudar com perguntas ou tarefas quando quiser.", meta={"memory_impact": "none", "session_id": session_id, "history_size": len(session_histories.get(session_id, [])), "intent": intent})

        # 1) Se LLM estiver pronto, usa o pipeline generativo para resposta em linguagem natural
        # Preferir gerador por sessão; se não houver, usar global
        gen = session_generative.get(session_id) if llm_ready else None
        if llm_ready and (gen is not None or generative_system_instance is not None):
            preq = _build_pre_queries_from_history(session_id)
            # Política de escrita baseada em intenção
            it = (intent.get("intent") or "ask").lower()
            write_mode = "auto"
            if it in ("greeting", "smalltalk", "stats", "export"):
                write_mode = "off"
            elif it in ("learn",):
                write_mode = "volatile"  # aprende mas marca como volátil até reforçar
            # Override por solicitação explícita do cliente
            if isinstance(req.write_mode, str):
                wm = req.write_mode.strip().lower()
                if wm in ("off", "volatile", "auto"):
                    write_mode = wm

            # Atalho especial: se intenção for "learn", dispara prompt de ensino explícito
            if it in ("learn", "learn_about"):
                learn_prompt = _curriculum_prompt(msg)
                answer, reflection = await loop.run_in_executor(
                    None,
                    (gen or generative_system_instance).process_complex_query,
                    learn_prompt,
                    [],
                    write_mode
                )
            else:
                answer, reflection = await loop.run_in_executor(
                    None,
                    (gen or generative_system_instance).process_complex_query,
                    msg,
                    preq,
                    write_mode
                )
            def _clean(ans: str) -> str:
                if not ans:
                    return ""
                # Remove cabeçalhos comuns e qualquer seção de análise
                lowered = ans.lower()
                cut_markers = ["análise metacognitiva", "analise metacognitiva", "analysis", "```json"]
                for mk in cut_markers:
                    idx = lowered.find(mk)
                    if idx != -1:
                        ans = ans[:idx]
                        break
                # Remove prefixos como "**Sua Resposta (em linguagem natural):**"
                ans = ans.replace("**Sua Resposta (em linguagem natural):**", "").strip()
                return ans.strip()
            # Parser de JSON à prova de barulho (último bloco ```json)
            parsed = _extract_last_json_block(answer)
            cleaned = _clean(answer)
            if not cleaned:
                cleaned = answer or ""
            _add_history(session_id, "assistant", cleaned)
            meta = { 'reflection': reflection, 'session_id': session_id, 'history_size': len(session_histories.get(session_id, [])), 'write_mode': write_mode, 'intent': intent }
            if it in ("learn", "learn_about"):
                meta['learned'] = True
                if isinstance(parsed, dict):
                    if isinstance(parsed.get('teach_seed'), list):
                        meta['learned_count'] = min(len(parsed.get('teach_seed')), 10)
                        meta['learn_preview'] = [str(x.get('content', {}).get('text', ''))[:100] for x in parsed.get('teach_seed')[:3] if isinstance(x, dict)]
                    if isinstance(parsed.get('next_prompts'), list):
                        # limita a 5 sugestões
                        meta['next_prompts'] = [str(p)[:140] for p in parsed.get('next_prompts')[:5]]
            return ChatResponse(reply=cleaned, meta=meta)

        # 2) Fallback: usa o córtex diretamente (RAG) e sintetiza um pequeno parágrafo
        cortex = (session_cortex.get(session_id) or (generative_system_instance.cortical_memory if generative_system_instance else cortical_system_instance))
        if cortex is None:
            raise HTTPException(status_code=503, detail="Córtex indisponível.")
        preq = _build_pre_queries_from_history(session_id)
        hist_context = "\n".join(preq)
        augmented = f"{hist_context}\nPergunta atual: {msg}" if hist_context else msg
        result = await loop.run_in_executor(None, cortex.enhanced_query_processing, augmented)
        top = result.get('top_items', [])
        if top:
            # Síntese simples: concatena 1-2 evidências em uma resposta curta
            snippets = [text for _, text, _, _ in top[:2]]
            reply = (
                f"Com base no meu conhecimento: {snippets[0]}" +
                (f" Também: {snippets[1]}" if len(snippets) > 1 else "")
            )
        else:
            reply = "Não encontrei contexto suficiente. Tente reformular a pergunta."
        meta = {
            'active_fragments': result.get('active_fragments', 0),
            'synapses': result.get('synaptic_network', {}).get('total_synapses', 0),
            'session_id': session_id,
            'intent': intent
        }
        _add_history(session_id, "assistant", reply)
        meta['history_size'] = len(session_histories.get(session_id, []))
        return ChatResponse(reply=reply, meta=meta)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no chat: {e}")


@app.get("/api/chat/history", response_model=ChatHistoryResponse)
def api_chat_history(session_id: str):
    sid = _get_session_id(session_id)
    hist = session_histories.get(sid, [])
    return ChatHistoryResponse(session_id=sid, messages=hist)


@app.post("/api/chat/clear", response_model=ChatHistoryResponse)
def api_chat_clear(req: ChatHistoryRequest):
    sid = _get_session_id(req.session_id)
    session_histories[sid] = []
    return ChatHistoryResponse(session_id=sid, messages=[])


@app.post("/api/intent", response_model=IntentResponse)
async def api_intent(req: IntentRequest):
    if not is_ready:
        raise HTTPException(status_code=503, detail="Sistema carregando.")
    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message requerido")
    loop = asyncio.get_event_loop()
    intent = await loop.run_in_executor(None, _analyze_intent_sync, msg, _get_session_id(req.session_id))
    return IntentResponse(intent=intent.get("intent", "unknown"), confidence=float(intent.get("confidence", 0.6)), entities=intent.get("entities") or [])


@app.get("/api/synapses", response_model=SynapseListResponse)
def list_synapses(min_weight: float = 0.0, only_consolidated: bool = False, include_nodes: bool = True):
    if not is_ready or generative_system_instance is None:
        raise HTTPException(status_code=503, detail="Sistema não pronto.")
    try:
        cortex = generative_system_instance.cortical_memory
        edges: List[SynapseEdge] = []
        for u, v, d in cortex.fragment_graph.edges(data=True):
            w = float(d.get('w', 0.0))
            volatile = bool(d.get('volatile', False))
            if w < min_weight:
                continue
            if only_consolidated and volatile:
                continue
            weights = None
            if 'weights' in d and isinstance(d['weights'], dict):
                # converte chaves para string para JSON
                weights = {str(k): float(val) for k, val in d['weights'].items()}
            edges.append(SynapseEdge(
                u=int(u), v=int(v), w=w, volatile=volatile,
                usage=int(d.get('usage_count', 0)),
                last_update=int(d.get('last_update', 0)),
                weights=weights
            ))

        nodes = None
        if include_nodes:
            nodes = {}
            for fid, data in cortex.fragments.items():
                nodes[int(fid)] = {
                    'usage': float(data.get('usage', 0.0)),
                    'subspace': int(data.get('subspace', -1)),
                    'label': data.get('label', ''),
                    'centrality': float(data.get('centrality', 0.0)) if 'centrality' in data else 0.0,
                    'plasticity_state': data.get('plasticity_state', 'normal')
                }

        return SynapseListResponse(total=len(edges), edges=edges, nodes=nodes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao listar sinapses: {e}")

# --- Execução da API ---
# Esta parte permite que você execute o arquivo diretamente com `python api.py`
if __name__ == "__main__":
    # O uvicorn é o servidor ASGI que executa o nosso aplicativo FastAPI.
    # host="0.0.0.0" torna a API acessível de outras máquinas na mesma rede.
    # reload=True reinicia o servidor automaticamente quando você salva o arquivo (ótimo para desenvolvimento).
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
