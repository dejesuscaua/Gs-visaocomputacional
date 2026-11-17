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

O SeguWatts é um assistente de uso consciente de energia para famílias de renda baixa e média no estado de São Paulo. A conta de luz mostra só o total do mês e não revela desperdícios do dia a dia, como uma luz ou uma TV que ficam ligadas em um cômodo vazio. O sensor de luminosidade do protótipo físico do projeto (EcoBot) percebe que há luz, mas não distingue luz natural de lâmpada nem sabe se há alguém no cômodo. Este trabalho resolve essa limitação com visão computacional. Treinamos uma rede YOLO11 nano, por transferência de aprendizado a partir dos pesos COCO, com um dataset público de 695 imagens anotadas de lâmpadas acesas e apagadas, e combinamos esse modelo com a YOLO pré-treinada no COCO para detectar pessoas no mesmo quadro. Sobre as detecções de cada quadro, uma regra com suavização temporal identifica quando há aparelho ligado sem ninguém por mais de dois minutos e gera um evento de possível desperdício de iluminação, com estimativa de energia e custo. O evento, sem nenhuma imagem, é enviado ao fluxo de alertas do SeguWatts no n8n. No conjunto de teste, o modelo atingiu mAP@50 de 0,783 e mAP@50-95 de 0,529 (0,896 e 0,924 de mAP@50 para luminária acesa e apagada), e o detector rodou a [preencher] quadros por segundo em CPU comum. O resultado mostra que é possível, com hardware acessível, transformar uma câmera em um sensor de desperdício que respeita a privacidade do morador.

## Introdução

Para famílias de renda baixa e média, a conta de luz pesa no orçamento, e o desperdício é invisível: a fatura informa o total do mês, mas não diz em que horário, em que cômodo ou em qual aparelho a energia foi usada. No SeguWatts, os sensores de tomada medem o consumo por aparelho, e o pipeline de dados detecta aparelhos que passam a gastar acima do normal. Na simulação com 100 residências, 30 casas tinham aparelhos fora do padrão, com economia potencial média de R$ 20,73 por mês por casa.

Há, porém, um desperdício que nenhum medidor de consumo enxerga: o aparelho que funciona normalmente, mas está ligado **sem ninguém usando**. O EcoBot, componente físico do projeto, tentou detectar isso com um sensor de luz (LDR). O próprio relatório do EcoBot registra o limite: o LDR não diferencia sol de lâmpada e não sabe se o cômodo está ocupado. A visão computacional responde às duas perguntas ao mesmo tempo, e por isso é o componente que fecha a lacuna.

## Objetivo

*(50 palavras)*

Treinar um modelo YOLO que reconheça, em tempo real, lâmpadas acesas e apagadas e, junto à detecção de pessoas, transforme o cenário de luz acesa em cômodo vazio em eventos de possível desperdício, integrados ao fluxo de alertas do SeguWatts, sem armazenar nem transmitir imagens.

## Explicação técnica das ferramentas

| Ferramenta | Uso no projeto |
|---|---|
| **YOLO11 (Ultralytics)** | Detector de objetos de estágio único: em uma passada pela rede, prevê caixas e classes para a imagem inteira, o que permite tempo real em CPU. Usamos a variante *nano* (2,6 milhões de parâmetros), que privilegia velocidade. |
| **Transferência de aprendizado** | O treino parte dos pesos aprendidos no COCO (80 classes, 118 mil imagens). As camadas iniciais já sabem extrair bordas, texturas e formas, então poucas centenas de imagens bastam para ensinar classes novas. O COCO não tem a classe "lâmpada" nem o estado aceso/apagado: essa distinção é o que nosso treino adiciona. |
| **Dois modelos no mesmo quadro** | O modelo treinado detecta o estado da iluminação; a YOLO pré-treinada no COCO, restrita à classe *person*, detecta pessoas. Os nomes de classe de qualquer dataset são traduzidos para o vocabulário do SeguWatts por uma função testada (`normalizar_classe`), o que permite trocar de dataset sem mudar a regra. |
| **Roboflow Universe** | Fonte do dataset público já anotado; download direto no Colab pela API. |
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

1. **Dataset.** Usamos o dataset público *Light Detection* do Roboflow Universe ([link](https://universe.roboflow.com/infocomm-thingy/light-detection-gecjq), versão 3, 695 imagens, licença CC BY 4.0), com as classes `Lamp On`, `Lamp Off`, `Light Bulb On` e `Light Bulb Off`, traduzidas para `lampada_acesa` e `lampada_apagada`.
2. **Divisão.** Mantivemos as divisões de treino e validação do dataset. Quando ele não traz conjunto de teste, separamos 20% da validação como teste, com semente fixa, para avaliar em imagens nunca vistas no treino. Treino: 603 imagens; validação: 58; teste: 34. Caixas no treino: Lamp Off 186, Lamp On 240, Light Bulb Off 132, Light Bulb On 141 (`resultados/contagem_classes.csv`).
3. **Detecção de pessoas.** Em vez de anotar pessoas, usamos a YOLO11n pré-treinada no COCO, restrita à classe *person*, que já tem alto desempenho nessa tarefa.
4. **Treino.** YOLO11n a partir dos pesos COCO, só com as classes de iluminação, até 80 épocas com parada antecipada (paciência de 20): o treino parou na época 69, com o melhor modelo na época 49, em cerca de 10 minutos numa GPU T4, entrada de 640 px, lote de 16, semente 42 (`notebooks/treino_colab.ipynb`).
5. **Avaliação.** Métricas no conjunto de **teste**, nunca visto no treino (`scripts/avaliar.py`), e FPS medido em CPU (`scripts/medir_fps.py`).
6. **Regra de desperdício.** A cada quadro, o cômodo é considerado com aparelho ligado se lâmpada acesa ou tela ligada aparece em 70% dos quadros dos últimos 2 s, e ocupado se uma pessoa aparece em 30% deles. Os limiares são assimétricos de propósito: um falso "vazio" geraria alerta incômodo ao morador, então basta a pessoa aparecer em parte dos quadros. Depois de 120 s de cômodo vazio com a luz acesa, um evento é aberto. Ele fecha quando alguém volta ou o aparelho é desligado. A regra tem 13 testes automatizados (`tests/test_regra.py`) e a tradução de classes tem 18 (`tests/test_config.py`).
7. **Demonstração.** Um vídeo gravado no celular, no cômodo de um integrante (pessoa presente → sai → sala vazia → volta), foi processado pelo detector no Colab (célula 10 do notebook), gerando o vídeo anotado e o CSV de eventos.
8. **Integração.** O evento (cômodo, aparelhos, horários, duração, kWh e R$ estimados) vai para um CSV local e para o webhook do n8n. Nenhum quadro é salvo ou transmitido.

## Exemplos de resultados

### Métricas no conjunto de teste

| Classe (dataset) | Vira | Precisão | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| Lamp On | lampada_acesa | 0,628 | 0,846 | 0,896 | 0,587 |
| Lamp Off | lampada_apagada | 0,878 | 0,889 | 0,924 | 0,631 |
| Light Bulb On | lampada_acesa | 0,750 | 0,727 | 0,787 | 0,580 |
| Light Bulb Off | lampada_apagada | 0,567 | 0,375 | 0,524 | 0,319 |
| **Geral** | | **0,706** | **0,709** | **0,783** | **0,529** |

Na validação (58 imagens), o melhor modelo teve mAP@50 de 0,744 e mAP@50-95 de 0,510. As luminárias (acesa e apagada), que são o caso típico de um cômodo, ficaram acima de 0,89 de mAP@50. O ponto fraco é a lâmpada solta apagada (*Light Bulb Off*: recall de 0,375), objeto pequeno e sem brilho, com apenas 8 instâncias no teste; por isso o resultado dessa classe tem alta variância.

A detecção de pessoas usa a YOLO11n do COCO sem retreino; o desempenho de referência publicado pela Ultralytics para esse modelo é mAP@50-95 de 39,5 no COCO val2017 (todas as classes).

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

Erros observados no teste e no vídeo (inclua a imagem de cada um a partir de `resultados/` e do vídeo anotado):

1. **Lâmpada solta apagada confundida com acesa** (`resultados/matriz_confusao.png`): 37% das lâmpadas *Light Bulb Off* foram classificadas como *Light Bulb On*, e outros 25% não foram detectadas. É o erro mais relevante para o SeguWatts, porque uma lâmpada apagada lida como acesa pode gerar um **alerta falso**. Duas defesas já existentes reduzem o efeito: a regra exige luz acesa em 70% dos quadros de uma janela de 2 s e confiança mínima de 0,5. A correção definitiva é acrescentar ao treino fotos de lâmpadas apagadas em ambientes reais.
2. **Falsos positivos no fundo:** a coluna *background* da matriz mostra como se distribuem as caixas previstas onde não havia lâmpada: 33% viraram *Light Bulb Off*, 30% *Lamp On*, 22% *Lamp Off* e 15% *Light Bulb On*. São tipicamente reflexos e pontos claros. Pelo mesmo motivo, a suavização temporal evita que um reflexo isolado abra um evento.
3. **Luminárias estão firmes:** *Lamp Off* e *Lamp On* acertam 89% e 90% na diagonal da matriz, com confusão de cerca de 10% entre si. É o caso mais comum em uma casa (luminária de teto ou abajur).
4. **Domínio diferente:** fotos do dataset público versus a iluminação e o ângulo da casa do vídeo (compare a confiança das detecções no vídeo com a do teste).
5. **Reflexo da lâmpada** em vidro ou espelho detectado como `lampada_acesa`.
6. **Pessoa parcialmente fora do quadro**: a suavização evita o alerta falso, mas atrasa a detecção de presença em até ~0,6 s.

## Conclusão

O modelo treinado reconhece o estado da iluminação com mAP@50 de 0,783 no conjunto de teste (0,896 e 0,924 para luminária acesa e apagada) e, combinado à detecção de pessoas do COCO, rodando a [preencher] FPS em CPU comum, o que permite usar um notebook antigo ou um mini PC como sensor. A regra temporal transforma detecções instáveis quadro a quadro em eventos confiáveis, e a integração com o n8n reaproveita o fluxo de alertas do SeguWatts sem que nenhuma imagem saia da casa.

**Limitações:** o modelo foi treinado com um dataset público e pode errar em casas com luminárias muito diferentes; a TV ficou fora desta versão por falta de dataset público com o estado ligada/desligada; ele não mede consumo, então a estimativa em R$ usa potências típicas; e uma câmera dentro de casa exige consentimento explícito de todos os moradores.

**Próximos passos:** acrescentar fotos das casas da equipe ao dataset (protocolo em `dataset/README.md`), incluir TV ligada/desligada e ventilador, rodar o modelo em uma ESP32-S3 com câmera, integrando-o ao EcoBot, e cruzar os eventos com o consumo medido pelos sensores de tomada para confirmar o desperdício em kWh reais.

## Links

- Repositório: https://github.com/dejesuscaua/Gs-visaocomputacional
- Vídeo de demonstração: [preencher]
- Vitrine do SeguWatts: https://github.com/dejesuscaua/Global-Solution
