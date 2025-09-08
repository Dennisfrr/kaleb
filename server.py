import threading
import json
from typing import List

from flask import Flask, request, jsonify, send_from_directory, make_response

# Importa o sistema do arquivo principal
from main import EnhancedSynapticCorticalSystem


app = Flask(__name__, static_folder='webchat', static_url_path='/webchat')
_lock = threading.Lock()


# Inicializa o sistema globalmente no startup
def _build_corpus() -> List[str]:
    # Corpus reduzido e neutro para startup rápido (pode substituir pelo seu dataset)
    return [
        "Algoritmos de machine learning revolucionam a análise de dados complexos.",
        "Deep learning permite reconhecimento avançado de padrões visuais.",
        "Redes neurais artificiais simulam o processamento cerebral humano.",
        "Inteligência artificial automatiza processos empresariais complexos.",
        "Plasticidade sináptica permite adaptação contínua do sistema nervoso.",
        "Metaplasticidade regula a capacidade de mudança das conexões neurais.",
        "Homeostase sináptica mantém equilíbrio da atividade neural.",
        "Sustentabilidade ambiental requer inovação tecnológica constante.",
        "Educação personalizada utiliza inteligência artificial avançada.",
        "Liderança adaptativa espelha flexibilidade das conexões cerebrais."
    ]


system = EnhancedSynapticCorticalSystem(n_subspaces=4, max_hierarchy_levels=3)
_ = system.process_corpus_optimized(_build_corpus())


def _cors(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    resp.headers['Access-Control-Allow-Methods'] = 'POST,GET,OPTIONS'
    return resp


@app.route('/')
def root():
    return send_from_directory('webchat', 'index.html')


@app.route('/api/healthz')
def healthz():
    return _cors(make_response({'status': 'ok'}, 200))


@app.route('/api/readyz')
def readyz():
    ok = bool(system and len(system.fragments) > 0)
    code = 200 if ok else 503
    return _cors(make_response({'ready': ok}, code))


@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def api_chat():
    if request.method == 'OPTIONS':
        return _cors(make_response('', 204))
    try:
        payload = request.get_json(silent=True) or {}
        message = str(payload.get('message', '')).strip()
        if not message:
            return _cors(make_response({'error': 'message requerido'}, 400))

        with _lock:
            result = system.enhanced_query_processing(message)

        # Síntese simples da resposta com base nos top items
        top = result.get('top_items', [])
        if top:
            lines = []
            for iid, text, score, explanation in top[:2]:
                lines.append(f"- [{score:.2f}] {text}")
            reply = "\n".join(lines)
        else:
            reply = "Não encontrei contexto suficiente. Tente reformular a pergunta."

        data = {
            'reply': reply,
            'meta': {
                'active_fragments': result.get('active_fragments', 0),
                'synapses': result.get('synaptic_network', {}).get('total_synapses', 0)
            }
        }
        return _cors(make_response(jsonify(data), 200))
    except Exception as e:
        return _cors(make_response({'error': str(e)}, 500))


@app.route('/api/commit', methods=['POST'])
def api_commit():
    # Executa manutenção da janela de commit manualmente
    try:
        with _lock:
            # Preferimos operar no subgrafo sujo se houver
            system._apply_synaptic_decay_and_ltd()
            if system._dirty_nodes:
                system._enforce_synaptic_limit_subset(system._dirty_nodes)
            else:
                system._enforce_synaptic_limit()
            # Atualiza centralidade incremental (usa últimos seeds conhecidos: nós sujos)
            seeds = list(system._dirty_nodes)[:20]
            system._update_fragment_centrality_incremental(seeds=seeds or list(system.fragments.keys())[:20])
            # Atualiza/poda fast paths
            system._update_fast_paths()
            # Limpa sujeira
            system._dirty_nodes.clear()
            system._dirty_edges.clear()
        return _cors(make_response({'status': 'committed'}, 200))
    except Exception as e:
        return _cors(make_response({'error': str(e)}, 500))


if __name__ == '__main__':
    # Executa servidor Flask
    # Acesse http://localhost:5000/ para abrir o chat
    app.run(host='0.0.0.0', port=5000, debug=False)


