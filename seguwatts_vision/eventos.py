"""Saída dos eventos de desperdício.

Todo evento vai para um CSV local (registro auditável que o pipeline de dados pode
ingerir) e, se houver SEGUWATTS_WEBHOOK_URL, para o webhook do n8n, que decide com IA
se vale mandar e-mail ao morador.

Se a internet cair, o evento continua no CSV com enviado=false: o detector nunca para
por falha de rede.
"""

from __future__ import annotations

import csv
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

from seguwatts_vision.regra import Evento

log = logging.getLogger("seguwatts_vision")

COLUNAS = [
    "id", "tipo", "comodo", "aparelhos", "vazio_desde", "aberto_em", "encerrado_em",
    "duracao_s", "energia_estimada_kwh", "custo_estimado_rs", "enviado",
]


class SaidaEventos:
    def __init__(self, arquivo_csv: str, tarifa_rs_kwh: float, webhook_url: str | None = None, timeout_s: float = 5.0):
        self.arquivo = Path(arquivo_csv)
        self.tarifa = tarifa_rs_kwh
        self.webhook_url = webhook_url
        self.timeout_s = timeout_s
        self.arquivo.parent.mkdir(parents=True, exist_ok=True)
        if not self.arquivo.exists():
            with self.arquivo.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(COLUNAS)

    def publicar(self, evento: Evento) -> dict:
        dados = evento.para_dict(self.tarifa)
        dados["enviado"] = self._enviar(dados)
        linha = {**dados, "aparelhos": ";".join(dados["aparelhos"])}
        with self.arquivo.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUNAS).writerow(linha)
        log.info("evento %s %s: %s", dados["tipo"], dados["id"], json.dumps(dados, ensure_ascii=False))
        return dados

    def _enviar(self, dados: dict) -> bool:
        if not self.webhook_url:
            return False
        if not self.webhook_url.startswith("https://") and not self.webhook_url.startswith("http://localhost"):
            log.warning("webhook ignorado: use https (ou http://localhost para testes)")
            return False
        corpo = json.dumps(dados, ensure_ascii=False).encode("utf-8")
        pedido = urllib.request.Request(
            self.webhook_url, data=corpo, method="POST",
            headers={"Content-Type": "application/json", "User-Agent": "seguwatts-vision/1.0"},
        )
        try:
            with urllib.request.urlopen(pedido, timeout=self.timeout_s) as resposta:  # nosec B310: esquema validado acima
                return 200 <= resposta.status < 300
        except (urllib.error.URLError, TimeoutError, OSError) as erro:
            log.warning("webhook indisponível (%s); evento mantido no CSV", erro)
            return False
