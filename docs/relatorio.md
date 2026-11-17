# SeguWatts Vision: detecção de aparelhos ligados em cômodos vazios com YOLO

**Global Solution 2026.2 · Applied Computer Vision · FIAP, Engenharia de Software, 4º ano**

## Integrantes

| Nome | RM |
|---|---|
| Cauã Gonçalves de Jesus | RM97648 |
| Davi Passanha de Sousa Guerra | RM551605 |
| Luan Silveira Macea | RM98290 |
| Luigi Ferrara Sinno | RM98047 |
| Rui Amorim Siqueira | RM98436 |

## Resumo

O SeguWatts é um assistente de uso consciente de energia para famílias de renda baixa e média no estado de São Paulo. A conta de luz mostra só o total do mês e não revela desperdícios do dia a dia, como uma luz ou uma TV que ficam ligadas em um cômodo vazio. O sensor de luminosidade do protótipo físico do projeto (EcoBot) percebe que há luz, mas não distingue luz natural de lâmpada nem sabe se há alguém no cômodo. Treinamos uma rede YOLO11 nano, por transferência de aprendizado a partir dos pesos COCO, com um dataset público de 695 imagens anotadas de lâmpadas acesas e apagadas, e combinamos esse modelo com a YOLO pré-treinada no COCO, que detecta pessoas e TVs no mesmo quadro; o estado da TV é decidido pelo brilho da tela. Sobre as detecções de cada quadro, uma regra com suavização temporal identifica quando há aparelho ligado sem ninguém por mais de dois minutos e gera um evento de possível desperdício, com estimativa de energia e custo. O evento, sem nenhuma imagem, é enviado ao fluxo de alertas do SeguWatts no n8n. No conjunto de teste, o modelo atingiu mAP@50 de 0,783 e mAP@50-95 de 0,529 (0,896 e 0,924 de mAP@50 para luminária acesa e apagada), e rodou a 9,9 quadros por segundo em CPU, sem GPU. Com hardware acessível, uma câmera vira um sensor de desperdício que respeita a privacidade do morador.

## Introdução

Para famílias de renda baixa e média, a conta de luz pesa no orçamento, e o desperdício é invisível: a fatura informa o total do mês, mas não diz em que horário, em que cômodo ou em qual aparelho a energia foi usada. No SeguWatts, os sensores de tomada medem o consumo por aparelho, e o pipeline de dados detecta aparelhos que passam a gastar acima do normal. Na simulação com 100 residências, 30 casas tinham aparelhos fora do padrão, com economia potencial média de R$ 20,73 por mês por casa.

Há, porém, um desperdício que nenhum medidor de consumo enxerga: o aparelho que funciona normalmente, mas está ligado **sem ninguém usando**. O EcoBot, componente físico do projeto, tentou detectar isso com um sensor de luz (LDR). O próprio relatório do EcoBot registra o limite: o LDR não diferencia sol de lâmpada e não sabe se o cômodo está ocupado. A visão computacional responde às duas perguntas ao mesmo tempo, e por isso é o componente que fecha a lacuna.

## Objetivo

Treinar um modelo YOLO que reconheça, em tempo real, lâmpadas acesas e apagadas e, junto à detecção de pessoas e TVs, transforme luz ou TV ligada em cômodo vazio em eventos de possível desperdício, integrados aos alertas do SeguWatts, sem armazenar nem transmitir imagens.

## Explicação técnica das ferramentas

| Ferramenta | Uso no projeto |
|---|---|
| **YOLO11 (Ultralytics)** | Detector de objetos de estágio único: em uma passada pela rede, prevê caixas e classes para a imagem inteira, o que permite tempo real em CPU. Usamos a variante *nano* (2,6 milhões de parâmetros), que privilegia velocidade. |
| **Transferência de aprendizado** | O treino parte dos pesos aprendidos no COCO (80 classes, 118 mil imagens). As camadas iniciais já sabem extrair bordas, texturas e formas, então poucas centenas de imagens bastam para ensinar classes novas. O COCO não tem a classe "lâmpada" nem o estado aceso/apagado: essa distinção é o que nosso treino adiciona. |
| **Dois modelos no mesmo quadro** | O modelo treinado detecta o estado da iluminação; a YOLO pré-treinada no COCO, restrita às classes *person* e *tv*, detecta pessoas e TVs. Os nomes de classe de qualquer dataset são traduzidos para o vocabulário do SeguWatts por uma função testada (`normalizar_classe`), o que permite trocar de dataset sem mudar a regra. |
| **Roboflow Universe** | Fonte do dataset público já anotado; download direto no Colab pela API. |
| **Google Colab (GPU T4)** | Ambiente gratuito com GPU para o treino. |
| **Estado da TV pelo brilho** | O COCO acha a TV, mas não diz se ela está ligada. Medimos o brilho dentro da caixa (sem a moldura): tela bem mais clara que o resto do quadro, ou com imagem (alta variação de brilho), conta como ligada; retângulo escuro e liso, como desligada (`estado_tela`, em `config.py`). |
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
3. **Detecção de pessoas e TVs.** Em vez de anotar pessoas e TVs, usamos a YOLO11n pré-treinada no COCO, restrita às classes *person* e *tv*, que já tem alto desempenho nessas tarefas. Para a TV, o estado ligada/desligada vem do brilho da tela. Calibramos os limiares com 7 fotos da sala de um integrante: a tela ligada tinha brilho médio de 124 a 157 (escala 0–255) contra 26 a 56 do quadro inteiro, e as 7 foram classificadas como ligadas.
4. **Treino.** YOLO11n a partir dos pesos COCO, só com as classes de iluminação, até 80 épocas com parada antecipada (paciência de 20): o treino parou na época 69, com o melhor modelo na época 49, em cerca de 10 minutos numa GPU T4, entrada de 640 px, lote de 16, semente 42 (`notebooks/treino_colab.ipynb`).
5. **Avaliação.** Métricas no conjunto de **teste**, nunca visto no treino (`scripts/avaliar.py`), e FPS medido em CPU (`scripts/medir_fps.py`).
6. **Regra de desperdício.** A cada quadro, o cômodo é considerado com aparelho ligado se lâmpada acesa ou tela ligada aparece em 70% dos quadros dos últimos 2 s, e ocupado se uma pessoa aparece em 30% deles. Os limiares são assimétricos de propósito: um falso "vazio" geraria alerta incômodo ao morador, então basta a pessoa aparecer em parte dos quadros. Depois de 120 s de cômodo vazio com a luz acesa, um evento é aberto. Ele fecha quando alguém volta ou o aparelho é desligado. O projeto tem 41 testes automatizados, rodados no GitHub Actions: 13 da regra (`tests/test_regra.py`), 24 da tradução de classes e do estado da TV (`tests/test_config.py`) e 4 da saída de eventos (`tests/test_eventos.py`).
7. **Demonstração ao vivo.** No Colab, a webcam do navegador envia os quadros para o detector (célula 11 do notebook, `seguwatts_vision/ao_vivo.py`), que mostra as caixas e o painel em tempo real. Gravamos a sala de um integrante com a TV ligada: pessoa presente → sai → sala vazia → volta. A célula 10 faz o mesmo com um vídeo gravado no celular.
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

### Figuras

- Curvas de treino: `resultados/curvas_treino.png`
- Matriz de confusão normalizada: `resultados/matriz_confusao.png`
- Curva precisão × recall: `resultados/curva_pr.png`
- Detecções em imagens de teste: `resultados/exemplo_1.jpg` a `exemplo_6.jpg`
- Painel do detector ao vivo nos três estados (ocupado, contando, desperdício): [vídeo de demonstração](https://youtu.be/TOoxmunTYGY)

### Desempenho em tempo real (CPU)

| Entrada | FPS médio | Latência média por quadro |
|---|---|---|
| 640 px | 9,9 | ~101 ms |
| 480 px | 12,2 | ~82 ms |

Medido na CPU do Google Colab (sem GPU), com `device="cpu"`, sobre as imagens de teste (célula 8 do notebook). Na GPU T4, a inferência levou 8 ms por imagem. Na demonstração, o detector roda dois modelos por quadro (iluminação e pessoas), então o FPS efetivo em CPU fica abaixo desses valores. Isso não afeta a regra, que decide sobre janelas de 2 s e não depende de uma taxa fixa de quadros.

### Erros analisados

Erros observados no conjunto de teste e nos testes na sala:

1. **Lâmpada solta apagada confundida com acesa** (`resultados/matriz_confusao.png`): 37% das lâmpadas *Light Bulb Off* foram classificadas como *Light Bulb On*, e outros 25% não foram detectadas. É o erro mais relevante para o SeguWatts, porque uma lâmpada apagada lida como acesa pode gerar um **alerta falso**. Duas defesas já existentes reduzem o efeito: a regra exige luz acesa em 70% dos quadros de uma janela de 2 s e confiança mínima de 0,5. A correção definitiva é acrescentar ao treino fotos de lâmpadas apagadas em ambientes reais.
2. **Falsos positivos no fundo:** a coluna *background* da matriz mostra como se distribuem as caixas previstas onde não havia lâmpada: 33% viraram *Light Bulb Off*, 30% *Lamp On*, 22% *Lamp Off* e 15% *Light Bulb On*. São tipicamente reflexos e pontos claros. Pelo mesmo motivo, a suavização temporal evita que um reflexo isolado abra um evento.
3. **Luminárias estão firmes:** *Lamp Off* e *Lamp On* acertam 89% e 90% na diagonal da matriz, com confusão de cerca de 10% entre si. É o caso mais comum em uma casa (luminária de teto ou abajur).
4. **Domínio diferente:** as fotos do dataset público têm a lâmpada em destaque; numa sala real, vista do canto, a lâmpada aparece pequena e a confiança cai. Por isso, na demonstração, o aparelho em destaque foi a TV, que o COCO detectou com confiança de 0,64 a 0,91 nas fotos da sala.
5. **Reflexo da lâmpada** em vidro ou espelho pode ser detectado como `lampada_acesa`; a suavização temporal reduz, mas não elimina, esse risco.
6. **Pessoa parcialmente fora do quadro**: a suavização evita o alerta falso, mas atrasa a detecção de presença em até ~0,6 s.

## Conclusão

O modelo treinado reconhece o estado da iluminação com mAP@50 de 0,783 no conjunto de teste (0,896 e 0,924 para luminária acesa e apagada) e, combinado à detecção de pessoas e TVs do COCO, roda a cerca de 10 FPS em CPU, sem GPU, o que permite usar um notebook antigo ou um mini PC como sensor. A regra temporal transforma detecções instáveis quadro a quadro em eventos confiáveis, e a integração com o n8n reaproveita o fluxo de alertas do SeguWatts sem que nenhuma imagem saia da casa.

**Limitações:** o modelo foi treinado com um dataset público e pode errar em casas com luminárias muito diferentes; o estado da TV vem de uma regra de brilho, e não de um modelo treinado, e pode errar com uma tela preta refletindo uma janela muito clara; o sistema não mede consumo, então a estimativa em R$ usa potências típicas; e uma câmera dentro de casa exige consentimento explícito de todos os moradores.

**Próximos passos:** acrescentar fotos das casas da equipe ao dataset (protocolo em `dataset/README.md`), treinar o estado da TV com fotos próprias no lugar da regra de brilho, incluir ventilador, rodar o modelo em uma ESP32-S3 com câmera, integrando-o ao EcoBot, e cruzar os eventos com o consumo medido pelos sensores de tomada para confirmar o desperdício em kWh reais.

## Links

- Repositório: https://github.com/dejesuscaua/Gs-visaocomputacional
- Vídeo de demonstração: https://youtu.be/TOoxmunTYGY
- Vitrine do SeguWatts: https://github.com/dejesuscaua/Global-Solution
