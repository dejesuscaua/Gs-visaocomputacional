import pytest

from seguwatts_vision.config import estado_tela, nome_tem_estado, normalizar_classe


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


@pytest.mark.parametrize("brilho, desvio, quadro, esperado", [
    (125, 76, 33, "tela_ligada"),     # medido numa foto real: sala escura, TV com imagem
    (157, 81, 56, "tela_ligada"),     # idem, outro ângulo
    (150, 30, 120, "tela_ligada"),    # sala clara, tela clara e uniforme
    (25, 8, 33, "tela_desligada"),    # tela preta em sala escura
    (55, 15, 120, "tela_desligada"),  # tela preta refletindo uma sala clara
    (80, 10, 75, "tela_desligada"),   # tão clara quanto a parede: não emite luz
])
def test_estado_tela_pelo_brilho(brilho, desvio, quadro, esperado):
    assert estado_tela(brilho, desvio, quadro) == esperado


def test_nome_tem_estado():
    assert not nome_tem_estado("tv")
    assert nome_tem_estado("TV On") and nome_tem_estado("tv_off")
