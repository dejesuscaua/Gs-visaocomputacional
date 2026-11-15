"""Regra de desperdício: transforma detecções quadro a quadro em eventos.

Este módulo não depende de câmera nem de modelo. Recebe a lista de detecções de cada
quadro e o instante em que ele foi capturado, e decide o estado do cômodo:

    OCUPADO        tem gente no cômodo (ou nada ligado): nada a fazer
    VAZIO_LIGADO   aparelho ligado sem ninguém, contando o tempo
    DESPERDICIO    passou do tempo-limite: um evento foi aberto

Suavização temporal
-------------------
Um único quadro ruim (pessoa de costas, reflexo na TV) não pode abrir nem fechar evento.
Por isso o estado é decidido sobre uma janela dos últimos `janela_s` segundos:

    ligado   = aparelho ligado em pelo menos 70% dos quadros da janela
    presenca = pessoa em pelo menos 30% dos quadros da janela

Os limiares são assimétricos de propósito. Errar "há gente" quando não há só atrasa o
alerta; errar "não há gente" quando há incomoda o morador com um alerta falso. Então
basta a pessoa aparecer em parte dos quadros para contar como presença.

A janela é medida em segundos, não em quadros, porque o FPS varia de um computador
para outro.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from seguwatts_vision.config import CLASSES_LIGADO, CLASSES_PRESENCA, POTENCIA_W


class Estado(str, Enum):
    OCUPADO = "ocupado"
    VAZIO_LIGADO = "vazio_ligado"
    DESPERDICIO = "desperdicio"


@dataclass(frozen=True)
class Deteccao:
    classe: str
    confianca: float
    caixa: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass
class Evento:
    """Possível desperdício. Contém só metadados: nenhuma imagem sai do dispositivo."""

    comodo: str
    aparelhos: list[str]
    inicio: float  # instante em que o cômodo ficou vazio com aparelho ligado
    aberto_em: float  # instante em que passou do tempo-limite
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    encerrado_em: float | None = None

    @property
    def duracao_s(self) -> float:
        fim = self.encerrado_em if self.encerrado_em is not None else self.aberto_em
        return max(0.0, fim - self.inicio)

    def energia_kwh(self) -> float:
        potencia = sum(POTENCIA_W.get(a, 0.0) for a in self.aparelhos)
        return potencia * self.duracao_s / 3600 / 1000

    def para_dict(self, tarifa_rs_kwh: float) -> dict:
        def iso(t: float | None) -> str | None:
            return None if t is None else datetime.fromtimestamp(t, tz=timezone.utc).isoformat(timespec="seconds")

        kwh = self.energia_kwh()
        return {
            "id": self.id,
            "tipo": "encerrado" if self.encerrado_em is not None else "aberto",
            "comodo": self.comodo,
            "aparelhos": self.aparelhos,
            "vazio_desde": iso(self.inicio),
            "aberto_em": iso(self.aberto_em),
            "encerrado_em": iso(self.encerrado_em),
            "duracao_s": round(self.duracao_s, 1),
            "energia_estimada_kwh": round(kwh, 6),
            "custo_estimado_rs": round(kwh * tarifa_rs_kwh, 6),
        }


@dataclass
class Resultado:
    estado: Estado
    aparelhos_ligados: list[str]
    presenca: bool
    segundos_vazio: float
    evento_aberto: Evento | None = None  # preenchido só no quadro em que o evento abre
    evento_encerrado: Evento | None = None  # preenchido só no quadro em que ele fecha


class MonitorDesperdicio:
    def __init__(
        self,
        comodo: str,
        limite_s: float = 120.0,
        janela_s: float = 2.0,
        confianca_min: float = 0.5,
        fracao_ligado: float = 0.7,
        fracao_presenca: float = 0.3,
    ) -> None:
        if limite_s <= 0 or janela_s <= 0:
            raise ValueError("limite_s e janela_s precisam ser positivos")
        self.comodo = comodo
        self.limite_s = limite_s
        self.janela_s = janela_s
        self.confianca_min = confianca_min
        self.fracao_ligado = fracao_ligado
        self.fracao_presenca = fracao_presenca
        self._janela: deque[tuple[float, frozenset[str], bool]] = deque()
        self._vazio_desde: float | None = None
        self._evento: Evento | None = None

    @property
    def evento_atual(self) -> Evento | None:
        return self._evento

    def atualizar(self, deteccoes: list[Deteccao], agora: float) -> Resultado:
        validas = [d for d in deteccoes if d.confianca >= self.confianca_min]
        ligados = frozenset(d.classe for d in validas if d.classe in CLASSES_LIGADO)
        pessoa = any(d.classe in CLASSES_PRESENCA for d in validas)

        self._janela.append((agora, ligados, pessoa))
        while self._janela and agora - self._janela[0][0] > self.janela_s:
            self._janela.popleft()

        total = len(self._janela)
        quadros_ligados = sum(1 for _, lig, _ in self._janela if lig)
        quadros_pessoa = sum(1 for _, _, p in self._janela if p)
        ligado = quadros_ligados >= self.fracao_ligado * total
        presenca = quadros_pessoa >= self.fracao_presenca * total

        # Aparelhos vistos ligados em qualquer quadro da janela (para o relatório do evento).
        aparelhos = sorted(set().union(*(lig for _, lig, _ in self._janela)))

        # Aquecimento: sem meia janela de histórico ainda não dá para afirmar que o cômodo
        # está vazio (evita contar a partir do primeiro quadro ao ligar a câmera).
        aquecendo = agora - self._janela[0][0] < self.janela_s / 2 and self._vazio_desde is None

        if not ligado or presenca or aquecendo:
            encerrado = self._encerrar(agora)
            self._vazio_desde = None
            return Resultado(Estado.OCUPADO, aparelhos if ligado else [], presenca, 0.0, evento_encerrado=encerrado)

        if self._vazio_desde is None:
            self._vazio_desde = agora
        vazio = agora - self._vazio_desde

        if self._evento is None and vazio >= self.limite_s:
            self._evento = Evento(self.comodo, aparelhos, inicio=self._vazio_desde, aberto_em=agora)
            return Resultado(Estado.DESPERDICIO, aparelhos, False, vazio, evento_aberto=self._evento)

        if self._evento is not None:
            # Um aparelho pode ser ligado no meio do evento: acumula na lista.
            self._evento.aparelhos = sorted(set(self._evento.aparelhos) | set(aparelhos))
            return Resultado(Estado.DESPERDICIO, aparelhos, False, vazio)

        return Resultado(Estado.VAZIO_LIGADO, aparelhos, False, vazio)

    def _encerrar(self, agora: float) -> Evento | None:
        if self._evento is None:
            return None
        evento, self._evento = self._evento, None
        evento.encerrado_em = agora
        return evento

    def finalizar(self, agora: float) -> Evento | None:
        """Fecha o evento em aberto quando o programa termina."""
        self._vazio_desde = None
        return self._encerrar(agora)
