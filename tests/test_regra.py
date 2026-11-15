import pytest

from seguwatts_vision.regra import Deteccao, Estado, MonitorDesperdicio

LAMPADA = Deteccao("lampada_acesa", 0.9)
LAMPADA_FRACA = Deteccao("lampada_acesa", 0.3)
APAGADA = Deteccao("lampada_apagada", 0.9)
TV = Deteccao("tela_ligada", 0.9)
PESSOA = Deteccao("pessoa", 0.9)


def rodar(monitor, quadros, inicio=0.0, fps=10):
    """Alimenta o monitor com uma lista de detecções por quadro, a `fps` quadros por segundo."""
    resultados = []
    for i, deteccoes in enumerate(quadros):
        resultados.append(monitor.atualizar(deteccoes, inicio + i / fps))
    return resultados


def test_luz_acesa_com_pessoa_nao_gera_evento():
    m = MonitorDesperdicio("sala", limite_s=5)
    rs = rodar(m, [[LAMPADA, PESSOA]] * 100)
    assert all(r.estado is Estado.OCUPADO for r in rs)
    assert m.evento_atual is None


def test_nada_ligado_nao_gera_evento():
    m = MonitorDesperdicio("sala", limite_s=5)
    rs = rodar(m, [[APAGADA]] * 100)
    assert all(r.estado is Estado.OCUPADO for r in rs)


def test_abre_evento_apos_limite_e_so_uma_vez():
    m = MonitorDesperdicio("sala", limite_s=5)
    rs = rodar(m, [[LAMPADA]] * 80)  # 8 s a 10 FPS
    abertos = [r.evento_aberto for r in rs if r.evento_aberto]
    assert len(abertos) == 1
    assert abertos[0].aparelhos == ["lampada_acesa"]
    primeiro = next(i for i, r in enumerate(rs) if r.evento_aberto)
    assert rs[primeiro - 1].estado is Estado.VAZIO_LIGADO
    assert rs[-1].estado is Estado.DESPERDICIO


def test_pessoa_voltando_encerra_evento_com_duracao_e_custo():
    m = MonitorDesperdicio("sala", limite_s=5)
    rs = rodar(m, [[LAMPADA, TV]] * 100 + [[LAMPADA, TV, PESSOA]] * 30)
    encerrados = [r.evento_encerrado for r in rs if r.evento_encerrado]
    assert len(encerrados) == 1
    ev = encerrados[0]
    assert ev.aparelhos == ["lampada_acesa", "tela_ligada"]
    # 10 s reais, menos ~1 s de aquecimento no início, mais ~0,6 s de latência da
    # suavização no fim (a pessoa precisa aparecer em 30% da janela de 2 s)
    assert 9.0 <= ev.duracao_s <= 11.0
    dados = ev.para_dict(tarifa_rs_kwh=1.0)
    assert dados["tipo"] == "encerrado"
    assert dados["energia_estimada_kwh"] > 0
    assert dados["custo_estimado_rs"] == pytest.approx(dados["energia_estimada_kwh"])


def test_apagar_a_luz_encerra_evento():
    m = MonitorDesperdicio("sala", limite_s=5)
    rs = rodar(m, [[LAMPADA]] * 70 + [[APAGADA]] * 30)
    assert sum(1 for r in rs if r.evento_encerrado) == 1
    assert rs[-1].estado is Estado.OCUPADO


def test_quadro_isolado_sem_pessoa_nao_zera_presenca():
    """Pessoa sai do enquadramento por poucos quadros: continua ocupado (suavização)."""
    m = MonitorDesperdicio("sala", limite_s=5)
    quadros = []
    for i in range(200):
        quadros.append([LAMPADA] if i % 4 == 0 else [LAMPADA, PESSOA])
    rs = rodar(m, quadros)
    assert all(r.estado is Estado.OCUPADO for r in rs)


def test_piscada_da_luz_nao_reinicia_contagem():
    """Um quadro sem a lâmpada detectada não zera o contador."""
    m = MonitorDesperdicio("sala", limite_s=5)
    quadros = [[] if i % 10 == 5 else [LAMPADA] for i in range(80)]
    rs = rodar(m, quadros)
    assert sum(1 for r in rs if r.evento_aberto) == 1


def test_deteccao_de_baixa_confianca_e_ignorada():
    m = MonitorDesperdicio("sala", limite_s=5, confianca_min=0.5)
    rs = rodar(m, [[LAMPADA_FRACA]] * 100)
    assert all(r.estado is Estado.OCUPADO for r in rs)


def test_janela_em_segundos_independe_do_fps():
    """O evento abre no mesmo tempo a 5 FPS e a 30 FPS."""
    tempos = []
    for fps in (5, 30):
        m = MonitorDesperdicio("sala", limite_s=5)
        rs = rodar(m, [[LAMPADA]] * (fps * 8), fps=fps)
        i = next(i for i, r in enumerate(rs) if r.evento_aberto)
        tempos.append(i / fps)
    assert abs(tempos[0] - tempos[1]) < 0.5


def test_finalizar_fecha_evento_aberto():
    m = MonitorDesperdicio("sala", limite_s=5)
    rodar(m, [[LAMPADA]] * 70)
    ev = m.finalizar(7.0)
    assert ev is not None and ev.encerrado_em == 7.0
    assert m.finalizar(8.0) is None


def test_pessoa_coco_conta_como_presenca():
    m = MonitorDesperdicio("sala", limite_s=5)
    rs = rodar(m, [[LAMPADA, Deteccao("person", 0.9)]] * 80)
    assert all(r.estado is Estado.OCUPADO for r in rs)


def test_aquecimento_nao_conta_vazio_no_primeiro_quadro():
    m = MonitorDesperdicio("sala", limite_s=5, janela_s=2)
    rs = rodar(m, [[LAMPADA]] * 5 + [[LAMPADA, PESSOA]] * 50)
    assert all(r.estado is Estado.OCUPADO for r in rs)


def test_parametros_invalidos():
    with pytest.raises(ValueError):
        MonitorDesperdicio("sala", limite_s=0)
