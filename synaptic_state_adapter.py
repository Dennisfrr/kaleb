from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
import networkx as nx


@dataclass
class DecodeHints:
    temperature: float = 0.7
    top_p: float = 0.9
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class SynapticStateAdapter:
    def __init__(self, cortical_system, max_fragments: int = 10, max_paths: int = 4,
                 alpha: float = 1.0, beta: float = 0.6, gamma: float = 0.3,
                 softmax_tau: float = 0.7, diversity_strength: float = 1.0,
                 tau_decay: float = 6.0, history_window: int = 12, act_threshold: float = 0.3,
                 surprise_strength: float = 0.5, max_path_hops: int = 3, num_paths: int = 6,
                 # PID + Sigmóide + bounds
                 Kp: float = 0.10, Ki: float = 0.02, Kd: float = 0.05,
                 T_min: float = 0.3, T_max: float = 0.95,
                 sig_a: float = 4.0, sig_b: float = 1.0,
                 top_p_min: float = 0.85, top_p_max: float = 0.97,
                 # UCB
                 ucb_kappa: float = 0.8, ucb_weight: float = 0.4,
                 # Energia-Entropia
                 en_ent_blend: float = 0.5):
        self.cx = cortical_system
        self.max_fragments = max_fragments
        self.max_paths = max_paths
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.softmax_tau = float(softmax_tau)
        self.diversity_strength = float(diversity_strength)
        self._rng = np.random.default_rng(42)
        self.tau_decay = float(tau_decay)
        self.history_window = int(history_window)
        self.act_threshold = float(act_threshold)
        self.surprise_strength = float(surprise_strength)
        self.max_path_hops = int(max_path_hops)
        self.num_paths = int(num_paths)
        # PID
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self._pid_integral = 0.0
        self._pid_prev_error = 0.0
        # Sigmóide e limites
        self.T_min = float(T_min)
        self.T_max = float(T_max)
        self.sig_a = float(sig_a)
        self.sig_b = float(sig_b)
        self.top_p_min = float(top_p_min)
        self.top_p_max = float(top_p_max)
        # UCB
        self.ucb_kappa = float(ucb_kappa)
        self.ucb_weight = float(ucb_weight)
        # Energia/Entropia blend fator em [0,1]
        self.en_ent_blend = float(np.clip(en_ent_blend, 0.0, 1.0))

    def _min_max_norm(self, arr: np.ndarray) -> np.ndarray:
        a_min = float(np.min(arr)) if arr.size > 0 else 0.0
        a_max = float(np.max(arr)) if arr.size > 0 else 1.0
        if a_max - a_min < 1e-12:
            return np.zeros_like(arr, dtype=float)
        return (arr - a_min) / (a_max - a_min)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _select_diverse(self, candidates: List[int], feat_map: Dict[int, np.ndarray], k: int) -> List[int]:
        if not candidates:
            return []
        if len(candidates) <= k:
            return list(candidates)
        # Greedy farthest-point (aprox. DPP): começa com melhor candidato, depois maximiza distância mínima
        selected: List[int] = [candidates[0]]
        remaining = set(candidates[1:])
        while len(selected) < k and remaining:
            best_id = None
            best_score = -1.0
            for cid in list(remaining):
                fv = feat_map[cid]
                # distância mínima a já selecionados (1 - cosine)
                dists = [1.0 - self._cosine(fv, feat_map[sid]) for sid in selected]
                score = min(dists) if dists else 1.0
                # pequeno ruído para desempate
                score += 1e-6 * self._rng.random()
                if score > best_score:
                    best_score = score
                    best_id = cid
            if best_id is None:
                break
            selected.append(best_id)
            remaining.discard(best_id)
        return selected

    def _ucb_array(self, ids: List[int]) -> np.ndarray:
        hist = getattr(self.cx, 'activation_history', [])
        if not hist:
            return np.zeros(len(ids), dtype=float)
        window = hist[-self.history_window:]
        t = max(1, len(window))
        ucb_vals: List[float] = []
        for fid in ids:
            acts = [float(h.get(fid, 0.0)) for h in window]
            mu = float(np.mean(acts)) if acts else 0.0
            n_i = float(np.sum([1.0 for a in acts if a >= self.act_threshold]))
            ucb = mu + self.ucb_kappa * np.sqrt(np.log(t + 1.0) / max(1.0, n_i + 1.0))
            ucb_vals.append(ucb)
        arr = np.asarray(ucb_vals, dtype=float)
        # normaliza min-max para combinar ao score composto
        a_min, a_max = float(np.min(arr)), float(np.max(arr))
        if a_max - a_min < 1e-12:
            return np.zeros_like(arr)
        return (arr - a_min) / (a_max - a_min)

    def _pick_hot_subgraph(self) -> Tuple[List[int], List[Tuple[int, int, float]]]:
        G = self.cx.fragment_graph
        if G.number_of_nodes() == 0:
            return [], []

        # Coleta métricas
        ids: List[int] = []
        use_list: List[float] = []
        cent_list: List[float] = []
        nov_list: List[float] = []
        sub_list: List[float] = []
        for fid, data in G.nodes(data=True):
            ids.append(int(fid))
            use_list.append(float(data.get("usage", 0.0)))
            cent_list.append(float(data.get("centrality", 0.0)))
            nov_list.append(float(data.get("novelty_score", 0.0)) if "novelty_score" in data else 0.0)
            sub_list.append(float(data.get("subspace", 0)))

        use_arr = np.asarray(use_list, dtype=float)
        cent_arr = np.asarray(cent_list, dtype=float)
        nov_arr = np.asarray(nov_list, dtype=float)
        sub_arr = np.asarray(sub_list, dtype=float)

        # Normalização min-max por métrica
        u_norm = self._min_max_norm(use_arr)
        c_norm = self._min_max_norm(cent_arr)
        n_norm = self._min_max_norm(nov_arr)
        s_norm = self._min_max_norm(sub_arr)  # usado apenas no vetor de features

        # Score composto + UCB normalizado
        ucb_norm = self._ucb_array(ids)
        score = self.alpha * u_norm + self.beta * c_norm + self.gamma * n_norm + self.ucb_weight * ucb_norm
        # Softmax com temperatura
        tau = max(1e-6, float(self.softmax_tau))
        logits = score / tau
        logits = logits - float(np.max(logits))  # estabilidade numérica
        expv = np.exp(logits)
        p = expv / np.sum(expv)

        # Amostragem sem reposição proporcional a p
        k = int(min(self.max_fragments, len(ids)))
        pool_size = int(min(max(2 * k, k + 2), len(ids)))
        cand_idx = self._rng.choice(len(ids), size=pool_size, replace=False, p=p)
        candidates = [ids[i] for i in cand_idx]

        # Vetor de features para diversidade: [u_norm, c_norm, n_norm, s_norm]
        feat_map: Dict[int, np.ndarray] = {}
        for i, fid in enumerate(ids):
            feat_map[fid] = np.asarray([u_norm[i], c_norm[i], n_norm[i], s_norm[i]], dtype=float)

        # Seleção diversa (greedy DPP-like)
        # Ordena candidatos por score decrescente para seed inicial consistente
        candidates.sort(key=lambda fid: -score[ids.index(fid)])
        diverse = self._select_diverse(candidates, feat_map, k=k)

        fids = diverse

        # ---- Seleção de trajetórias com pesos ajustados ----
        def _coactivation_rate(u: int, v: int) -> float:
            hist = getattr(self.cx, 'activation_history', [])
            if not hist:
                return 0.0
            window = hist[-self.history_window:]
            total = len(window)
            if total == 0:
                return 0.0
            both = 0
            for acts in window:
                au = float(acts.get(u, 0.0))
                av = float(acts.get(v, 0.0))
                if au >= self.act_threshold and av >= self.act_threshold:
                    both += 1
            return float(both / total)

        def _adjusted_weight(u: int, v: int) -> float:
            if not G.has_edge(u, v):
                return 0.0
            base = float(self.cx._edge_weight_effective(u, v, hot=False))
            edata = G[u][v]
            last_upd = float(edata.get('last_update', 0))
            t_now = float(getattr(self.cx, 'interaction_count', 0))
            dt = max(0.0, t_now - last_upd)
            decay = float(np.exp(-dt / max(self.tau_decay, 1e-6)))
            coact = _coactivation_rate(u, v)
            surprise = 1.0 + self.surprise_strength * (1.0 - coact)
            return base * decay * surprise

        # Subgrafo induzido pelos nós escolhidos
        sub = G.subgraph(fids).copy()
        if sub.number_of_edges() == 0:
            return fids, []

        # Define custo como -log(w_hat)
        for u, v in sub.edges():
            w_hat = max(_adjusted_weight(int(u), int(v)), 1e-6)
            sub[u][v]['w_hat'] = w_hat
            sub[u][v]['cost'] = -float(np.log(w_hat))

        # Seleciona caminhos de menor energia entre pares de nós
        paths: List[Tuple[List[int], float]] = []
        fid_list = list(sub.nodes())
        for i in range(len(fid_list)):
            for j in range(i + 1, len(fid_list)):
                s, t = fid_list[i], fid_list[j]
                try:
                    path = nx.shortest_path(sub, source=s, target=t, weight='cost')
                    if len(path) - 1 <= self.max_path_hops:
                        # score = soma dos logs => - total_cost
                        total_cost = 0.0
                        for a, b in zip(path[:-1], path[1:]):
                            total_cost += float(sub[a][b]['cost'])
                        paths.append((path, -total_cost))
                except Exception:
                    continue

        if not paths:
            # fallback: retorna top arestas por w_hat
            edges: List[Tuple[int, int, float]] = []
            for u, v in sub.edges():
                edges.append((int(u), int(v), float(sub[u][v].get('w_hat', 0.0))))
            edges.sort(key=lambda e: -e[2])
            return fids, edges[: self.max_paths]

        # Ordena caminhos por energia (maior produto de w_hat)
        paths.sort(key=lambda x: -x[1])
        chosen_paths = paths[: self.num_paths]

        # Extrai arestas dos melhores caminhos e resume por maior w_hat
        best_edges: Dict[Tuple[int, int], float] = {}
        for path, _score in chosen_paths:
            for a, b in zip(path[:-1], path[1:]):
                u, v = int(a), int(b)
                val = float(sub[u][v]['w_hat'])
                key = (min(u, v), max(u, v))
                if key not in best_edges or val > best_edges[key]:
                    best_edges[key] = val

        edges_ranked = [(u, v, w) for (u, v), w in best_edges.items()]
        edges_ranked.sort(key=lambda e: -e[2])
        return fids, edges_ranked[: self.max_paths]

    # --- Thompson Sampling para seleção de regra de plasticidade ---
    def _thompson_pick_rule(self, rules: List[str]) -> str:
        # Estado Beta(α, β) por regra
        if not hasattr(self.cx, '_rule_alpha'):
            self.cx._rule_alpha = {r: 1.0 for r in rules}
            self.cx._rule_beta = {r: 1.0 for r in rules}
        samples = {}
        for r in rules:
            a = float(self.cx._rule_alpha.get(r, 1.0))
            b = float(self.cx._rule_beta.get(r, 1.0))
            # Amostra Beta por reparam via Gamma
            x = self._rng.gamma(a, 1.0)
            y = self._rng.gamma(b, 1.0)
            s = float(x / max(1e-9, (x + y)))
            samples[r] = s
        # Escolhe a maior amostra
        best = max(samples.items(), key=lambda kv: kv[1])[0]
        return best

    def _update_rule_posterior(self, rule: str, reward_signal: float):
        if not hasattr(self.cx, '_rule_alpha'):
            return
        r = float(np.clip(reward_signal, 0.0, 1.0))
        self.cx._rule_alpha[rule] = float(self.cx._rule_alpha.get(rule, 1.0) + r)
        self.cx._rule_beta[rule] = float(self.cx._rule_beta.get(rule, 1.0) + (1.0 - r))

    def _summarize_fragments(self, fids: List[int]) -> List[str]:
        texts: List[str] = []
        for fid in fids:
            node = self.cx.fragments.get(fid, {})
            label = node.get("label", f"frag_{fid}")
            sub = node.get("subspace", 0)
            spec = node.get("specialization_score", 0.0)
            state = node.get("plasticity_state", "normal")
            texts.append(f"[{label}|s{sub}|spec={spec:.2f}|{state}]")
        return texts

    def _homeostasis_to_hints(self, selected_fids: List[int] | None = None) -> DecodeHints:
        G = self.cx.fragment_graph
        if G.number_of_nodes() == 0:
            return DecodeHints()

        # Média do peso sináptico de saída por nó (mais estável que soma global crua)
        out_strengths: List[float] = []
        for u in G.nodes():
            s = 0.0
            for _, v, d in G.edges(u, data=True):
                s += float(d.get("w", d.get("w_q", 0) * d.get("w_scale", 0.0)))
            out_strengths.append(s)
        mean_out = float(np.mean(out_strengths)) if out_strengths else 0.0

        target = float(getattr(self.cx, "max_total_synaptic_weight", 3.0))
        target = max(target, 1e-6)
        error = mean_out - target
        # PID incremental
        self._pid_integral += error
        d_error = error - self._pid_prev_error
        self._pid_prev_error = error
        delta_T = -(self.Kp * error + self.Ki * self._pid_integral + self.Kd * d_error)

        # Sigmóide adaptativa baseada em r = mean_out/target
        r = mean_out / target
        T_sig = self.T_min + (self.T_max - self.T_min) / (1.0 + float(np.exp(-self.sig_a * (r - self.sig_b))))

        # Combinação: sigmóide + ajuste PID
        temp = float(np.clip(T_sig + delta_T, self.T_min, self.T_max))

        # Meta-acoplamento: multiplicador por confiança Bayesiana (posterior_mean)
        # τ(t) = clip( τ_PID(t) * (1 + λ * (1 - c̄)), T_min, T_max )
        try:
            posterior_mean = None
            # Tenta obter do próprio córtex
            posterior_mean = float(getattr(self.cx, "posterior_mean", None))
            if posterior_mean is None or not np.isfinite(posterior_mean):
                # Alternativa: parâmetros internos do filtro beta (quando expostos)
                alpha = float(getattr(self.cx, "_conf_alpha", 0.0))
                beta = float(getattr(self.cx, "_conf_beta", 0.0))
                if (alpha + beta) > 0:
                    posterior_mean = float(alpha / (alpha + beta))
            if posterior_mean is not None and np.isfinite(posterior_mean):
                lambda_conf = 0.6  # ganho do multiplicador de confiança
                temp = float(np.clip(temp * (1.0 + lambda_conf * (1.0 - posterior_mean)), self.T_min, self.T_max))
        except Exception:
            pass

        # Acoplamento Energia/Entropia do grafo
        # Energia: média dos pesos (estável); Entropia: H de distribuição de pesos normalizada
        weights: List[float] = []
        for _, _, d in G.edges(data=True):
            w = float(d.get("w", d.get("w_q", 0) * d.get("w_scale", 0.0)))
            if w > 0:
                weights.append(w)
        if weights:
            W_arr = np.asarray(weights, dtype=float)
            E_norm = float(np.mean(W_arr))  # energia média por aresta
            p_w = W_arr / float(np.sum(W_arr))
            H_graph = -float(np.sum(p_w * np.log(p_w + 1e-12)))
            H_norm = float(H_graph / max(1e-9, np.log(len(W_arr))))  # [0,1]
            # fator: maior quando H_norm alto e E_norm baixo → mais exploração; caso contrário reduz T
            scale = float(np.clip(H_norm / max(E_norm, 1e-6), 0.0, 5.0))
            # mistura com PID/sigmóide
            mix = float(np.clip(self.en_ent_blend, 0.0, 1.0))
            temp = float(np.clip((1.0 - mix) * temp + mix * (temp * (0.5 + 0.5 * np.tanh(scale))), self.T_min, self.T_max))

        # Entropia dos estados de plasticidade → top_p proporcional à incerteza global
        counts = {"normal": 0, "potentiated": 0, "depressed": 0}
        total_frag = 0
        for f in self.cx.fragments.values():
            state = str(f.get("plasticity_state", "normal")).lower()
            if state not in counts:
                state = "normal"
            counts[state] += 1
            total_frag += 1
        if total_frag <= 0:
            H = 0.0
        else:
            probs = []
            for k in ["normal", "potentiated", "depressed"]:
                p = counts[k] / total_frag
                if p > 0:
                    probs.append(p)
            H = -float(np.sum([p * np.log(p) for p in probs])) if probs else 0.0
        H_max = float(np.log(3.0))
        top_p = self.top_p_min + (self.top_p_max - self.top_p_min) * (H / max(H_max, 1e-9))
        top_p = float(np.clip(top_p, self.top_p_min, self.top_p_max))

        return DecodeHints(
            temperature=temp,
            top_p=top_p,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )

    def _blend_with_profile(self, hints: DecodeHints, base_profile: Dict[str, float] | None) -> DecodeHints:
        if not isinstance(base_profile, dict):
            return hints
        def _blend(a: float | None, b: float | None, wa=0.6, wb=0.4, lo=0.0, hi=1.0) -> float | None:
            if a is None and b is None:
                return None
            if a is None:
                val = b
            elif b is None:
                val = a
            else:
                val = wa * float(a) + wb * float(b)
            return float(min(max(val, lo), hi))
        t_prof = float(base_profile.get("temperature")) if "temperature" in base_profile else None
        p_prof = float(base_profile.get("top_p")) if "top_p" in base_profile else None
        temperature = _blend(hints.temperature, t_prof, wa=0.6, wb=0.4, lo=0.3, hi=0.95)
        top_p = _blend(hints.top_p, p_prof, wa=0.5, wb=0.5, lo=0.85, hi=0.97)
        return DecodeHints(
            temperature=temperature if temperature is not None else hints.temperature,
            top_p=top_p if top_p is not None else hints.top_p,
            presence_penalty=hints.presence_penalty,
            frequency_penalty=hints.frequency_penalty,
        )

    def to_prompt_state(self, user_query: str, base_profile: Dict[str, float] | None = None) -> Dict[str, Any]:
        fids, edges = self._pick_hot_subgraph()
        frag_tokens = self._summarize_fragments(fids)
        lines: List[str] = []
        lines.append("<<CORTEX-STATE v1>>")
        if frag_tokens:
            lines.append("FRAGMENTS: " + ", ".join(frag_tokens))
        if edges:
            edge_strs = [f"({u}->{v}|w≈{w:.2f})" for (u, v, w) in edges]
            lines.append("TRAJECTORIES: " + ", ".join(edge_strs))
        # Seleciona regra de plasticidade via Thompson Sampling (loja de equações)
        rule = self._thompson_pick_rule(["hebb", "stdp", "bcm"])
        lines.append("PRINCIPLES: hebb, ltd, metaplasticity, stdp, bcm, fast-paths, homeostasis")
        lines.append(f"PLASTICITY-RULE: {rule}")
        lines.append("GUIDE: use os padrões acima como viés de raciocínio, favorecendo trajetórias fortes mas explorando conexões recentes e plausíveis.")
        lines.append(f"QUERY: {user_query}")
        prefix = "\n".join(lines)
        hints = self._homeostasis_to_hints(selected_fids=fids)
        hints = self._blend_with_profile(hints, base_profile)
        # Atualiza posterior da regra com sinal leve baseado em incerteza atual (H_norm e voláteis)
        try:
            G = self.cx.fragment_graph
            weights = []
            total_edges = 0
            volatile = 0
            for _, _, d in G.edges(data=True):
                total_edges += 1
                w = float(d.get('w', 0.0))
                if w > 0:
                    weights.append(w)
                if d.get('volatile', False):
                    volatile += 1
            H_norm = 0.0
            if weights:
                W = np.asarray(weights, dtype=float)
                p = W / float(np.sum(W))
                H = -float(np.sum(p * np.log(p + 1e-12)))
                H_norm = float(H / max(1e-9, np.log(len(W))))
            u_ratio = float(volatile / max(1, total_edges)) if total_edges > 0 else 0.0
            # Recompensa proxy: maior sob incerteza (estimula regras que exploram melhor)
            r_proxy = float(np.clip(0.5 * H_norm + 0.5 * u_ratio, 0.0, 1.0))
            self._update_rule_posterior(rule, r_proxy)
        except Exception:
            pass
        return {"prefix": prefix, "hints": hints}


