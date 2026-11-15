# SeguWatts Vision: detecção de aparelhos ligados em cômodos vazios com YOLO

**Global Solution 2026.2 · Applied Computer Vision · FIAP, Engenharia de Software, 4º ano**

> Modelo do documento da disciplina. Os trechos `[preencher]` dependem do treino. Todo o resto já está escrito e pode ser ajustado. Exporte para PDF ao final.

## Integrantes

| Nome | RM |
|---|---|
| Cauã Gonçalves de Jesus | RM97648 |
| Davi Passanha de Sousa Guerra | RM551605 |
| Luan Silveira Macea | RM98290 |
| Luigi Ferrara Sinno | RM98047 |
| Rui Amorim Siqueira | RM98436 |

## Resumo

*(até 250 palavras; esta versão tem cerca de 220 sem os números)*

O SeguWatts é um assistente de uso consciente de energia para famílias de renda baixa e média no estado de São Paulo. A conta de luz mostra só o total do mês e não revela desperdícios do dia a dia, como uma luz ou uma TV que ficam ligadas em um cômodo vazio. O sensor de luminosidade do protótipo físico do projeto (EcoBot) percebe que há luz, mas não distingue luz natural de lâmpada nem sabe se há alguém no cômodo. Este trabalho resolve essa limitação com visão computacional. Treinamos uma rede YOLO11 nano, por transferência de aprendizado a partir dos pesos COCO, com cerca de [preencher] fotografias de cômodos reais das casas da equipe, anotadas em cinco classes: lâmpada acesa, lâmpada apagada, tela ligada, tela desligada e pessoa. Sobre as detecções de cada quadro, uma regra com suavização temporal identifica quando há aparelho ligado sem ninguém por mais de dois minutos e gera um evento de possível desperdício, com estimativa de energia e custo. O evento, sem nenhuma imagem, é enviado ao fluxo de alertas do SeguWatts no n8n. No conjunto de teste, o modelo atingiu mAP@50 de [preencher] e mAP@50-95 de [preencher], e o detector rodou a [preencher] quadros por segundo em CPU comum. O resultado mostra que é possível, com hardware acessível, transformar uma câmera em um sensor de desperdício que respeita a privacidade do morador.

## Introdução

Para famílias de renda baixa e média, a conta de luz pesa no orçamento, e o desperdício é invisível: a fatura informa o total do mês, mas não diz em que horário, em que cômodo ou em qual aparelho a energia foi usada. No SeguWatts, os sensores de tomada medem o consumo por aparelho, e o pipeline de dados detecta aparelhos que passam a gastar acima do normal. Na simulação com 100 residências, 30 casas tinham aparelhos fora do padrão, com economia potencial média de R$ 20,73 por mês por casa.

Há, porém, um desperdício que nenhum medidor de consumo enxerga: o aparelho que funciona normalmente, mas está ligado **sem ninguém usando**. O EcoBot, componente físico do projeto, tentou detectar isso com um sensor de luz (LDR). O próprio relatório do EcoBot registra o limite: o LDR não diferencia sol de lâmpada e não sabe se o cômodo está ocupado. A visão computacional responde às duas perguntas ao mesmo tempo, e por isso é o componente que fecha a lacuna.

## Objetivo

*(50 palavras)*

Treinar um modelo YOLO capaz de identificar, em tempo real, lâmpadas e telas ligadas ou desligadas e a presença de pessoas, transformando essas detecções em eventos de possível desperdício de energia, integrados ao fluxo de alertas do SeguWatts, sem armazenar nem transmitir imagens do morador.

## Explicação técnica das ferramentas

| Ferramenta | Uso no projeto |
|---|---|
| **YOLO11 (Ultralytics)** | Detector de objetos de estágio único: em uma passada pela rede, prevê caixas e classes para a imagem inteira, o que permite tempo real em CPU. Usamos a variante *nano* (2,6 milhões de parâmetros), que privilegia velocidade. |
| **Transferência de aprendizado** | O treino parte dos pesos aprendidos no COCO (80 classes, 118 mil imagens). As camadas iniciais já sabem extrair bordas, texturas e formas, então poucas centenas de imagens bastam para ensinar as classes novas. O COCO tem "tv", mas não sabe se ela está ligada: essa distinção é o que nosso treino adiciona. |
| **Roboflow** | Anotação das caixas, divisão treino/validação/teste e aumento de dados (flip, brilho, rotação, blur). |
| **Google Colab (GPU T4)** | Ambiente gratuito com GPU para o treino. |
| **OpenCV** | Captura da webcam ou do vídeo, desenho das caixas e do painel, gravação do vídeo anotado. |
| **Regra temporal (Python)** | Converte detecções quadro a quadro em estados do cômodo (ocupado, vazio com aparelho ligado, desperdício), com janela deslizante de 2 s. |
| **n8n** | Recebe o evento por webhook e reaproveita o fluxo de alertas do SeguWatts (componente de automação), que decide com IA se envia e-mail ao morador. |

### Métricas

- **Precisão:** das caixas que o modelo previu, quantas estavam certas.
- **Recall:** dos objetos que existiam, quantos o modelo encontrou.
- **mAP@50:** média, entre as classes, da área sob a curva precisão × recall, contando como acerto a caixa que sobrepõe a verdadeira em pelo menos 50% (IoU ≥ 0,5).
- **mAP@50-95:** a mesma média, exigindo sobreposições de 50% a 95%. Mede também a justeza da caixa.

## Metodologia

1. **Coleta.** Cada integrante fotografou a própria casa seguindo o protocolo de `dataset/README.md`: 2 cômodos ou mais, dia e noite, 3 ângulos e todas as combinações de aparelho ligado ou desligado, com e sem pessoa. Total: [preencher] imagens.
2. **Anotação.** Caixas desenhadas no Roboflow nas 5 classes, com revisão cruzada entre integrantes.
3. **Divisão e aumento.** 70% treino, 20% validação e 10% teste. Aumento só no treino (3×): flip horizontal, brilho ±25%, exposição ±15%, rotação ±10° e blur leve. Aumentos de cor foram evitados porque alteram a aparência de "aceso" e "apagado".
4. **Treino.** YOLO11n a partir dos pesos COCO, 80 épocas com parada antecipada (paciência de 20), entrada de 640 px, lote de 16, semente 42 (`notebooks/treino_colab.ipynb`).
5. **Avaliação.** Métricas no conjunto de **teste**, nunca visto no treino (`scripts/avaliar.py`), e FPS medido em CPU (`scripts/medir_fps.py`).
6. **Regra de desperdício.** A cada quadro, o cômodo é considerado com aparelho ligado se lâmpada acesa ou tela ligada aparece em 70% dos quadros dos últimos 2 s, e ocupado se uma pessoa aparece em 30% deles. Os limiares são assimétricos de propósito: um falso "vazio" geraria alerta incômodo ao morador, então basta a pessoa aparecer em parte dos quadros. Depois de 120 s de cômodo vazio com aparelho ligado, um evento é aberto. Ele fecha quando alguém volta ou o aparelho é desligado. A regra tem 13 testes automatizados (`tests/test_regra.py`).
7. **Integração.** O evento (cômodo, aparelhos, horários, duração, kWh e R$ estimados) vai para um CSV local e para o webhook do n8n. Nenhum quadro é salvo ou transmitido.

## Exemplos de resultados

### Métricas no conjunto de teste

| Classe | Precisão | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| lampada_acesa | [ ] | [ ] | [ ] | [ ] |
| lampada_apagada | [ ] | [ ] | [ ] | [ ] |
| tela_ligada | [ ] | [ ] | [ ] | [ ] |
| tela_desligada | [ ] | [ ] | [ ] | [ ] |
| pessoa | [ ] | [ ] | [ ] | [ ] |
| **Geral** | [ ] | [ ] | [ ] | [ ] |

*(copie de `resultados/metricas_teste.json`)*

### Figuras

- Curvas de treino: `resultados/curvas_treino.png`
- Matriz de confusão normalizada: `resultados/matriz_confusao.png`
- Curva precisão × recall: `resultados/curva_pr.png`
- Detecções em imagens de teste: `resultados/exemplo_1.jpg` a `exemplo_6.jpg`
- Painel do detector nos três estados (ocupado, contando, desperdício): prints do vídeo

### Desempenho em tempo real (CPU)

| Entrada | FPS médio | Latência p50 | Latência p95 |
|---|---|---|---|
| 640 px | [ ] | [ ] ms | [ ] ms |
| 480 px | [ ] | [ ] ms | [ ] ms |
| 320 px | [ ] | [ ] ms | [ ] ms |

*(copie de `resultados/fps.json`; informe o processador usado)*

### Erros analisados

Descreva 3 erros reais encontrados no teste ou no vídeo, com a imagem e a explicação. Os mais prováveis:

1. **TV desligada em cômodo escuro** confundida com fundo (recall baixo de `tela_desligada`).
2. **Reflexo da lâmpada** em vidro ou espelho detectado como `lampada_acesa`.
3. **Pessoa parcialmente fora do quadro**: a suavização evita o alerta falso, mas atrasa a detecção de presença em até ~0,6 s.

## Conclusão

[Ajuste com os números finais.] O modelo treinado reconhece o estado dos aparelhos e a presença de pessoas com mAP@50 de [preencher], rodando a [preencher] FPS em CPU comum, o que permite usar um notebook antigo ou um mini PC como sensor. A regra temporal transforma detecções instáveis quadro a quadro em eventos confiáveis, e a integração com o n8n reaproveita o fluxo de alertas do SeguWatts sem que nenhuma imagem saia da casa.

**Limitações:** o modelo foi treinado com casas de cinco famílias e pode errar em ambientes muito diferentes; ele não mede consumo, então a estimativa em R$ usa potências típicas; e uma câmera dentro de casa exige consentimento explícito de todos os moradores.

**Próximos passos:** ampliar o dataset com casas de outras regiões, adicionar classes (ventilador, ar-condicionado), rodar o modelo em uma ESP32-S3 com câmera, integrando-o ao EcoBot, e cruzar os eventos com o consumo medido pelos sensores de tomada para confirmar o desperdício em kWh reais.

## Links

- Repositório: https://github.com/dejesuscaua/Gs-visaocomputacional
- Vídeo de demonstração: [preencher]
- Vitrine do SeguWatts: https://github.com/dejesuscaua/Global-Solution
