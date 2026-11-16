import pytest

from seguwatts_vision.config import normalizar_classe


@pytest.mark.parametrize("nome, esperado", [
    # Datasets públicos de lâmpadas (Roboflow Universe)
    ("Lamp On", "lampada_acesa"),
    ("Lamp Off", "lampada_apagada"),
    ("Light Bulb On", "lampada_acesa"),
    ("Light Bulb Off", "lampada_apagada"),
    ("light_on", "lampada_acesa"),
    ("light_off", "lampada_apagada"),
    ("LED_On", "lampada_acesa"),
    ("ligth-off", None),  # erro de digitação de um dataset público: não adivinhamos
    # COCO
    ("person", "pessoa"),
    ("tv", "tela_ligada"),
    ("chair", None),
    ("bottle", None),
    # Nomes do próprio SeguWatts passam direto
    ("lampada_acesa", "lampada_acesa"),
    ("tela_desligada", "tela_desligada"),
    ("pessoa", "pessoa"),
    # Luz sem estado não serve para a regra
    ("lamp", None),
])
def test_normalizar_classe(nome, esperado):
    assert normalizar_classe(nome) == esperado


def test_on_nao_casa_dentro_de_outra_palavra():
    """'on' em 'detection' ou 'monitor' não pode virar 'ligado'."""
    assert normalizar_classe("lamp detection") is None
    assert normalizar_classe("monitor") == "tela_ligada"
