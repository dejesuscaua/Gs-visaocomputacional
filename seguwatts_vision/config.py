"""Configuração do SeguWatts Vision.

Tudo que muda de uma casa para outra (cômodo, tempo-limite, tarifa, webhook) vem de
variáveis de ambiente ou da linha de comando. Nada sensível fica no código.
"""

import os
from dataclasses import dataclass, field

# Classes do SeguWatts. Qualquer nome vindo de um dataset (próprio ou público) é
# traduzido para estes nomes por normalizar_classe().
CLASSES_LIGADO = {"lampada_acesa", "tela_ligada"}
CLASSES_DESLIGADO = {"lampada_apagada", "tela_desligada"}
CLASSES_PRESENCA = {"pessoa"}
CLASSES_SEGUWATTS = CLASSES_LIGADO | CLASSES_DESLIGADO | CLASSES_PRESENCA

_LUZ = {"lamp", "lamps", "bulb", "bulbs", "light", "lights", "led", "luz", "lampada", "lampadas", "luminaria"}
_TELA = {"tv", "television", "screen", "monitor", "tela"}
_LIGADO = {"on", "acesa", "aceso", "ligada", "ligado", "lit"}
_DESLIGADO = {"off", "apagada", "apagado", "desligada", "desligado", "unlit"}


def normalizar_classe(nome: str) -> str | None:
    """Traduz o nome de classe de um dataset para o vocabulário do SeguWatts.

    Exemplos: "Lamp On" e "Light Bulb On" → lampada_acesa; "light_off" → lampada_apagada;
    "person" (COCO) → pessoa; "tv" do COCO, sem estado → tela_ligada só no modo de teste.
    Devolve None para classes que não interessam ao SeguWatts (cadeira, garrafa...).
    """
    n = nome.strip().lower()
    if n in CLASSES_SEGUWATTS:
        return n
    palavras = set(n.replace("-", " ").replace("_", " ").replace("/", " ").split())
    if palavras & {"person", "pessoa", "people", "human"}:
        return "pessoa"
    if palavras & _LUZ:
        if palavras & _DESLIGADO:
            return "lampada_apagada"
        if palavras & _LIGADO:
            return "lampada_acesa"
        return None  # luz sem estado não ajuda a regra
    if palavras & _TELA:
        if palavras & _DESLIGADO:
            return "tela_desligada"
        return "tela_ligada"
    return None

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
