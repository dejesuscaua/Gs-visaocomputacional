"""Configuração do SeguWatts Vision.

Tudo que muda de uma casa para outra (cômodo, tempo-limite, tarifa, webhook) vem de
variáveis de ambiente ou da linha de comando. Nada sensível fica no código.
"""

import os
from dataclasses import dataclass, field

# Classes treinadas no dataset da equipe (ver dataset/README.md).
CLASSES_LIGADO = {"lampada_acesa", "tela_ligada"}
CLASSES_DESLIGADO = {"lampada_apagada", "tela_desligada"}
# "person" permite testar a detecção de presença com os pesos COCO pré-treinados.
CLASSES_PRESENCA = {"pessoa", "person"}

# Potência típica usada para estimar o desperdício em kWh e R$.
# Valores de referência do Procel/Inmetro para aparelhos comuns; ajuste para a casa real.
POTENCIA_W = {
    "lampada_acesa": 12.0,  # lâmpada LED de sala (equivalente a 75 W incandescente)
    "tela_ligada": 90.0,  # TV LED de 43 a 50 polegadas
}


def _float(nome: str, padrao: float) -> float:
    try:
        return float(os.getenv(nome, padrao))
    except ValueError:
        return padrao


@dataclass
class Config:
    comodo: str = field(default_factory=lambda: os.getenv("COMODO", "sala"))
    limite_s: float = field(default_factory=lambda: _float("LIMITE_SEGUNDOS", 120.0))
    janela_s: float = field(default_factory=lambda: _float("JANELA_SEGUNDOS", 2.0))
    confianca_min: float = field(default_factory=lambda: _float("CONFIANCA_MIN", 0.5))
    # Tarifa média residencial em SP com impostos (R$/kWh). Ajuste pela conta do morador.
    tarifa_rs_kwh: float = field(default_factory=lambda: _float("TARIFA_RS_KWH", 0.85))
    webhook_url: str | None = field(default_factory=lambda: os.getenv("SEGUWATTS_WEBHOOK_URL") or None)
    arquivo_eventos: str = field(default_factory=lambda: os.getenv("ARQUIVO_EVENTOS", "resultados/eventos.csv"))
