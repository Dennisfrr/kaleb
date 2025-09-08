"""Cliente de LLM desacoplado para integração com o sistema cortical.

Atualmente suporta Gemini via biblioteca google-generativeai.
"""

from __future__ import annotations

import os
from typing import Optional


class LLMClientError(Exception):
    pass


class LLMClient:
    """
    Cliente simples para modelos de linguagem.
    Implementação atual: Gemini (google-generativeai).
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        self._provider = "gemini"
        self._model_name = model_name
        self._available = False
        self._last_error: Optional[str] = None

        try:
            import google.generativeai as genai  # type: ignore
        except Exception as e:  # ImportError ou outros
            self._last_error = (
                "Biblioteca 'google-generativeai' não encontrada. "
                "Instale com: pip install google-generativeai"
            )
            self._genai = None
            self._model = None
            return

        self._genai = genai

        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            self._last_error = (
                "Chave GEMINI_API_KEY não encontrada no ambiente. "
                "Defina GEMINI_API_KEY para habilitar o LLM."
            )
            self._model = None
            return

        try:
            self._genai.configure(api_key=key)
            self._model = self._genai.GenerativeModel(self._model_name)
            self._available = True
        except Exception as e:
            self._last_error = f"Falha ao inicializar o modelo Gemini: {e}"
            self._model = None

    @property
    def available(self) -> bool:
        return bool(self._available and self._model is not None)

    @property
    def model_name(self) -> str:
        return self._model_name

    def health_check(self) -> bool:
        return self.available

    def last_error(self) -> Optional[str]:
        return self._last_error

    def generate_text(
        self,
        prompt: str,
        temperature: float | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        extra_config: Optional[dict] = None,
    ) -> str:
        if not self.available:
            raise LLMClientError(self._last_error or "LLM não está disponível.")
        try:
            gen_cfg: dict = {}
            if temperature is not None:
                gen_cfg["temperature"] = float(temperature)
            if top_p is not None:
                gen_cfg["top_p"] = float(top_p)
            # A API do Gemini não suporta diretamente presence/frequency penalties como no OpenAI,
            # mas mantemos para compatibilidade futura e provedores alternativos.
            if extra_config:
                gen_cfg.update(dict(extra_config))

            if gen_cfg:
                result = self._model.generate_content(prompt, generation_config=gen_cfg)
            else:
                result = self._model.generate_content(prompt)
            return getattr(result, "text", "") or ""
        except Exception as e:
            raise LLMClientError(f"Erro ao gerar texto no LLM: {e}")


