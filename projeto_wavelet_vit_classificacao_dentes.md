# Projeto de Pesquisa

## Wavelet-Enhanced Vision Transformer for Tooth Classification in Intraoral Images

### 1. Visão geral

Este projeto propõe avaliar se representações multiescala baseadas em transformadas wavelet podem melhorar o desempenho de Vision Transformers (ViTs) na classificação automática de dentes individuais extraídos de fotografias intraorais.

O conjunto de dados disponível contém imagens intraorais acompanhadas de anotações no formato YOLO, nas quais cada objeto é representado por uma classe e uma bounding box normalizada:

\[
[\text{class}, x_c, y_c, w, h].
\]

As classes observadas são codificadas numericamente, de 0 a 8. O significado clínico/anatômico de cada código deverá ser recuperado do arquivo de configuração do conjunto de dados (`data.yaml`, `dataset.yaml` ou equivalente) antes da redação final do artigo.

A proposta evita transformar o estudo em um problema completo de detecção. As bounding boxes já existentes serão usadas apenas para recortar os dentes individualmente. Cada recorte será então tratado como uma amostra de classificação.

O foco científico será comparar uma representação convencional em RGB com representações baseadas em wavelets antes da entrada no Vision Transformer.

---

# 2. Título provisório

**Wavelet-Enhanced Vision Transformer for Tooth Classification in Intraoral Images**

Alternativas:

1. **Frequency-Aware Tooth Classification in Intraoral Photographs Using Wavelet-Enhanced Vision Transformers**
2. **Multiscale Wavelet Representations for Vision Transformer-Based Tooth Classification**
3. **Do Wavelets Improve Vision Transformers for Tooth Classification in Intraoral Images?**

O terceiro título é especialmente adequado caso o artigo seja centrado em comparação experimental e não em proposição de uma arquitetura nova.

---

# 3. Problema de pesquisa

A classificação automática de dentes em fotografias intraorais depende da identificação de características morfológicas relativamente sutis, tais como:

- forma global;
- largura e altura;
- contorno;
- orientação;
- bordas incisais;
- cúspides;
- textura superficial;
- padrões locais de transição;
- relação espacial entre partes da coroa dental.

Vision Transformers são capazes de modelar relações espaciais globais por meio do mecanismo de atenção, mas a transformação direta da imagem em patches pode não explicitar adequadamente componentes de alta e baixa frequência.

A transformada wavelet permite decompor uma imagem em componentes multiescala, representando separadamente estruturas de baixa frequência e detalhes direcionais de alta frequência.

Dessa forma, a questão central deste estudo é:

> **A incorporação de representações wavelet melhora o desempenho de Vision Transformers na classificação de dentes em fotografias intraorais?**

---

# 4. Questões de pesquisa

## RQ1

**Does wavelet-based image representation improve Vision Transformer performance for tooth classification in intraoral photographs?**

A RQ1 será a principal questão do trabalho.

---

## RQ2

**Which wavelet family provides the most informative representation for tooth classification?**

Serão comparadas inicialmente três wavelets:

\[
\text{Haar},\quad db2,\quad db4.
\]

---

## RQ3

**Which wavelet frequency components contribute most to tooth classification performance?**

Será realizada uma análise de ablação considerando:

\[
LL,
\]

\[
LH+HL+HH,
\]

e

\[
LL+LH+HL+HH.
\]

---

# 5. Hipóteses

## H1 — ganho pela representação wavelet

\[
H_1:
F1_{\text{Wavelet-ViT}}
>
F1_{\text{ViT}}.
\]

Espera-se que a decomposição multiescala forneça ao modelo informações complementares à imagem RGB convencional.

---

## H2 — importância da informação de alta frequência

\[
H_2:
F1_{LL+LH+HL+HH}
>
F1_{LL}.
\]

A hipótese pressupõe que contornos e detalhes presentes nas bandas de alta frequência contribuam para distinguir classes dentárias com morfologias semelhantes.

---

## H3 — dependência da wavelet

\[
H_3:
F1_{\psi_i}
\neq
F1_{\psi_j}
\]

para pelo menos duas wavelets distintas \(\psi_i\) e \(\psi_j\).

Essa hipótese avalia se a escolha da função wavelet influencia significativamente o desempenho final.

---

# 6. Objetivo geral

Avaliar o efeito de representações multiescala baseadas em transformadas wavelet sobre o desempenho de Vision Transformers na classificação automática de dentes em fotografias intraorais.

---

# 7. Objetivos específicos

1. Organizar as imagens e anotações YOLO do conjunto de dados.

2. Extrair automaticamente os dentes individuais utilizando as bounding boxes existentes.

3. Construir um conjunto de dados de classificação contendo os recortes dentários e suas respectivas classes.

4. Implementar um Vision Transformer como modelo de referência.

5. Aplicar diferentes transformadas wavelet às imagens dentárias.

6. Desenvolver uma estratégia de entrada wavelet para o Vision Transformer.

7. Comparar o desempenho do ViT convencional com versões baseadas em Haar, db2 e db4.

8. Avaliar separadamente a contribuição das bandas de baixa e alta frequência.

9. Analisar os erros por meio de matrizes de confusão e métricas por classe.

10. Determinar se as representações wavelet oferecem ganhos consistentes e estatisticamente relevantes.

---

# 8. Delimitação do estudo

Este artigo não pretende:

- desenvolver um sistema completo de diagnóstico odontológico;
- substituir avaliação profissional;
- realizar detecção de dentes a partir de imagens não anotadas;
- propor uma arquitetura Transformer de grande escala;
- realizar segmentação semântica ou de instâncias;
- classificar doenças bucais;
- estimar condição periodontal;
- identificar cáries;
- realizar análise ortodôntica completa.

O problema será restrito à:

\[
\boxed{\text{classificação de dentes previamente localizados}}
\]

usando as bounding boxes existentes.

Essa delimitação reduz significativamente a complexidade experimental e permite concentrar o estudo na contribuição das representações wavelet.

---

# 9. Conjunto de dados

## 9.1 Estrutura das anotações

Cada arquivo de anotação segue o formato YOLO:

\[
c,\;x_c,\;y_c,\;w,\;h,
\]

em que:

- \(c\) representa a classe;
- \(x_c\) representa a coordenada horizontal normalizada do centro;
- \(y_c\) representa a coordenada vertical normalizada do centro;
- \(w\) representa a largura normalizada;
- \(h\) representa a altura normalizada.

As coordenadas deverão ser convertidas para pixels para extração de cada dente.

Para uma imagem com largura \(W\) e altura \(H\):

\[
x = x_c W,
\]

\[
y = y_c H,
\]

\[
w_p = wW,
\]

\[
h_p = hH.
\]

Os limites da bounding box serão:

\[
x_{\min}=x-\frac{w_p}{2},
\]

\[
x_{\max}=x+\frac{w_p}{2},
\]

\[
y_{\min}=y-\frac{h_p}{2},
\]

\[
y_{\max}=y+\frac{h_p}{2}.
\]

---

## 9.2 Classes

O conjunto contém classes numéricas observadas no intervalo:

\[
c\in\{0,1,\ldots,8\}.
\]

Antes do início do treinamento será construída uma tabela contendo:

| ID | Nome da classe | Número de amostras |
|---:|---|---:|
| 0 | a definir | |
| 1 | a definir | |
| 2 | a definir | |
| 3 | a definir | |
| 4 | a definir | |
| 5 | a definir | |
| 6 | a definir | |
| 7 | a definir | |
| 8 | a definir | |

O significado de cada classe será obtido do arquivo de configuração original do dataset.

---

# 10. Preparação dos dados

## 10.1 Extração dos dentes

Para cada imagem:

1. carregar o arquivo correspondente de labels;
2. converter as coordenadas normalizadas para pixels;
3. recortar cada bounding box;
4. associar o recorte ao respectivo rótulo;
5. salvar o resultado em diretórios organizados por classe.

Estrutura sugerida:

```text
dataset_crops/
├── train/
│   ├── class_0/
│   ├── class_1/
│   ├── ...
│   └── class_8/
├── val/
└── test/
```

---

## 10.2 Margem ao redor do dente

Pode ser utilizada pequena margem em torno da bounding box original:

\[
m \in [5\%,10\%].
\]

Isso permite preservar parte do contexto local sem incluir uma região excessivamente grande da imagem.

O valor deverá permanecer fixo em todos os experimentos.

---

## 10.3 Redimensionamento

Os recortes serão redimensionados para:

\[
224\times224
\]

pixels, tamanho compatível com configurações convencionais de Vision Transformers pré-treinados.

---

## 10.4 Normalização

Para o modelo RGB:

\[
I'=\frac{I-\mu}{\sigma}.
\]

Caso seja utilizado um modelo pré-treinado, serão utilizados os parâmetros de normalização correspondentes ao treinamento original.

---

# 11. Divisão treino, validação e teste

A separação dos dados deverá ocorrer **antes da extração dos dentes**.

Isso é essencial para impedir que dentes oriundos da mesma fotografia apareçam simultaneamente nos conjuntos de treino e teste.

O protocolo preferencial será:

\[
70\% \text{ treinamento},
\]

\[
15\% \text{ validação},
\]

\[
15\% \text{ teste}.
\]

Se houver identificação individual dos pacientes, o particionamento deverá ser realizado por paciente.

Caso essa informação não esteja disponível, a separação deverá ser realizada por imagem.

Nunca deverá ser utilizado split aleatório diretamente sobre os recortes.

---

# 12. Data augmentation

Para reduzir overfitting, poderão ser utilizadas transformações leves:

- pequenas rotações;
- pequenas translações;
- variação moderada de brilho;
- variação moderada de contraste;
- pequenas alterações de escala.

Deve-se evitar transformações que alterem significativamente a anatomia dentária.

Flip horizontal deverá ser analisado com cautela, pois pode modificar a interpretação de lateralidade caso essa informação esteja codificada nas classes.

---

# 13. Transformada Wavelet Discreta

## 13.1 Definição

A DWT bidimensional decompõe a imagem em quatro componentes:

\[
I
\xrightarrow{DWT}
\{LL,LH,HL,HH\}.
\]

As sub-bandas representam diferentes conteúdos espaciais.

### LL

\[
LL
\]

representa a aproximação de baixa frequência e concentra grande parte da estrutura global da imagem.

### LH

\[
LH
\]

representa detalhes associados predominantemente a uma direção espacial.

### HL

\[
HL
\]

representa detalhes na direção complementar.

### HH

\[
HH
\]

representa detalhes de alta frequência e estruturas diagonais.

---

## 13.2 Nível de decomposição

O estudo principal utilizará:

\[
J=1.
\]

Níveis adicionais serão considerados apenas se os resultados iniciais justificarem experimentação complementar.

Essa decisão mantém o estudo simples e reduz a expansão do espaço experimental.

---

## 13.3 Wavelets avaliadas

Serão utilizadas:

\[
\text{Haar},
\]

\[
db2,
\]

\[
db4.
\]

A Haar fornece uma representação simples e localizada.

As Daubechies permitem avaliar se funções de suporte e suavidade diferentes alteram o desempenho do modelo.

---

# 14. Construção das representações wavelet

Serão consideradas três formas de entrada.

## 14.1 Representação completa

\[
X_W =
[LL,LH,HL,HH].
\]

As quatro bandas serão normalizadas e combinadas em um tensor multicanal.

---

## 14.2 Apenas baixa frequência

\[
X_{LL}=LL.
\]

Essa configuração avalia quanto da classificação pode ser explicada pela estrutura global da imagem.

---

## 14.3 Apenas alta frequência

\[
X_{HF}=[LH,HL,HH].
\]

Essa configuração avalia a contribuição de bordas, contornos e detalhes finos.

---

# 15. Modelos

## 15.1 Baseline principal — Vision Transformer

O baseline será um Vision Transformer convencional.

Fluxo:

\[
I
\rightarrow
\text{Patch Embedding}
\rightarrow
\text{Transformer Encoder}
\rightarrow
\text{Classification Head}
\rightarrow
\hat{y}.
\]

A imagem é dividida em patches:

\[
I\rightarrow\{p_1,p_2,\ldots,p_N\}.
\]

Cada patch é convertido em um vetor:

\[
z_i = E(p_i)+e_i,
\]

em que \(E\) representa o embedding e \(e_i\) representa a codificação posicional.

O Transformer utiliza atenção:

\[
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V.
\]

---

## 15.2 Wavelet-ViT

No modelo baseado em wavelet:

\[
I
\xrightarrow{DWT}
X_W
\rightarrow
\text{Wavelet Representation}
\rightarrow
\text{Patch Embedding}
\rightarrow
\text{Transformer Encoder}
\rightarrow
\hat{y}.
\]

A principal diferença será a representação de entrada.

O objetivo do artigo não será afirmar que uma nova arquitetura completa de Transformer foi criada, mas verificar se a representação multiescala melhora a classificação.

---

# 16. Configuração experimental

Serão avaliados inicialmente:

| ID | Modelo | Entrada |
|---|---|---|
| E1 | ViT | RGB |
| E2 | Wavelet-ViT | Haar |
| E3 | Wavelet-ViT | db2 |
| E4 | Wavelet-ViT | db4 |

Os hiperparâmetros deverão ser mantidos tão equivalentes quanto possível.

Parâmetros iniciais sugeridos:

- image size: 224 × 224;
- optimizer: AdamW;
- learning rate inicial: \(10^{-4}\);
- batch size: 16 ou 32;
- epochs: 30–50;
- early stopping: 7–10 épocas;
- loss: cross-entropy;
- weight decay: \(10^{-4}\).

Os valores finais deverão ser registrados para garantir reprodutibilidade.

---

# 17. Transfer learning

Se o dataset for pequeno, será preferível utilizar pesos pré-treinados.

O backbone ViT poderá ser inicializado a partir de treinamento em ImageNet.

Duas estratégias poderão ser avaliadas:

### Estratégia A

Congelar inicialmente o backbone e treinar apenas a cabeça de classificação.

### Estratégia B

Realizar fine-tuning completo com learning rate reduzido.

A estratégia escolhida deverá ser utilizada igualmente nos modelos comparados.

---

# 18. Função de perda

A função principal será a cross-entropy:

\[
\mathcal{L}
=
-\sum_{k=1}^{K}
y_k \log(\hat{p}_k),
\]

em que:

- \(K\) é o número de classes;
- \(y_k\) representa o rótulo verdadeiro;
- \(\hat{p}_k\) representa a probabilidade prevista.

Caso exista forte desbalanceamento, poderão ser utilizados pesos:

\[
\mathcal{L}_{weighted}
=
-\sum_{k=1}^{K}
w_k y_k\log(\hat{p}_k).
\]

Uma escolha possível é:

\[
w_k=
\frac{N}{K N_k},
\]

em que \(N_k\) representa a quantidade de exemplos da classe \(k\).

---

# 19. Métricas

## Accuracy

\[
Accuracy
=
\frac{\text{predições corretas}}{\text{total de exemplos}}.
\]

---

## Precision

Para a classe \(k\):

\[
Precision_k
=
\frac{TP_k}{TP_k+FP_k}.
\]

---

## Recall

\[
Recall_k
=
\frac{TP_k}{TP_k+FN_k}.
\]

---

## F1-score

\[
F1_k
=
2
\frac{Precision_k Recall_k}
{Precision_k+Recall_k}.
\]

---

## Macro-F1

A principal métrica do trabalho será:

\[
MacroF1
=
\frac{1}{K}
\sum_{k=1}^{K}F1_k.
\]

Ela atribui o mesmo peso a todas as classes e é adequada quando há diferença no número de amostras entre categorias.

---

# 20. Análise de ablação

Após a comparação principal, o melhor modelo wavelet será utilizado na análise de ablação.

| ID | Representação |
|---|---|
| A1 | LL |
| A2 | LH + HL + HH |
| A3 | LL + LH + HL + HH |

A análise responderá:

\[
\text{estrutura global}
\quad vs.\quad
\text{detalhes}
\quad vs.\quad
\text{representação completa}.
\]

---

# 21. Matriz de confusão

Para o melhor modelo será construída uma matriz:

\[
C_{ij},
\]

em que:

\[
C_{ij}
=
\text{número de objetos da classe }i
\text{ classificados como }j.
\]

Essa análise permitirá identificar quais classes dentárias são mais frequentemente confundidas.

---

# 22. Análise estatística

Idealmente, cada experimento será repetido com pelo menos três seeds:

\[
s\in\{1,2,3\}.
\]

Serão reportados:

\[
\mu \pm \sigma.
\]

Para comparação entre ViT e o melhor Wavelet-ViT, poderá ser aplicado teste estatístico sobre resultados obtidos em múltiplas execuções.

Dependendo da distribuição dos resultados, poderá ser utilizado:

- teste t pareado; ou
- teste de Wilcoxon.

O nível de significância será:

\[
\alpha=0.05.
\]

A análise estatística será considerada complementar, especialmente se o número de repetições for suficiente.

---

# 23. Reprodutibilidade

O estudo deverá documentar:

- versão do Python;
- versão do PyTorch;
- versão da biblioteca de wavelets;
- GPU ou CPU utilizada;
- seeds;
- configuração de treino;
- organização dos conjuntos;
- número de imagens;
- número de recortes;
- distribuição das classes;
- hiperparâmetros;
- código utilizado para extração;
- código utilizado para treino;
- código de avaliação.

Sugere-se utilizar:

```text
Python
PyTorch
torchvision
PyWavelets
scikit-learn
pandas
numpy
matplotlib
```

---

# 24. Resultados esperados

O objetivo não é pressupor que a representação wavelet será superior.

Existem três resultados cientificamente válidos.

## Cenário 1 — Wavelet melhora

\[
F1_{Wavelet}
>
F1_{RGB}.
\]

Isso indicaria que a representação multiescala fornece informação adicional útil ao Transformer.

---

## Cenário 2 — desempenho semelhante

\[
F1_{Wavelet}
\approx
F1_{RGB}.
\]

Nesse caso, seria possível concluir que a transformação wavelet não oferece benefício suficiente para justificar maior complexidade.

---

## Cenário 3 — Wavelet piora

\[
F1_{Wavelet}
<
F1_{RGB}.
\]

Esse resultado também é relevante, pois demonstraria que o ViT convencional já extrai adequadamente as informações necessárias ou que a decomposição remove informação discriminativa.

---

# 25. Figuras planejadas

## Figura 1 — Pipeline geral

```text
Intraoral image
      |
      v
YOLO annotation
      |
      v
Tooth crop
      |
      +------------------+
      |                  |
      v                  v
     RGB                DWT
      |           LL LH HL HH
      |                  |
      v                  v
     ViT          Wavelet-ViT
      |                  |
      +--------+---------+
               |
               v
        Tooth class
```

---

## Figura 2 — Exemplos do dataset

Apresentar:

- imagem intraoral;
- bounding boxes;
- recortes de diferentes classes.

---

## Figura 3 — Decomposição wavelet

Mostrar para um mesmo dente:

- RGB;
- LL;
- LH;
- HL;
- HH.

---

## Figura 4 — Matriz de confusão

Apresentar a matriz de confusão do melhor modelo.

---

## Figura 5 — Comparação de desempenho

Gráfico com:

\[
MacroF1
\]

para:

- ViT;
- Haar-ViT;
- db2-ViT;
- db4-ViT.

---

# 26. Tabelas planejadas

## Tabela 1 — Distribuição do dataset

| Classe | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| 0 | | | | |
| 1 | | | | |
| ... | | | | |
| 8 | | | | |

---

## Tabela 2 — Configuração dos modelos

| Modelo | Entrada | Wavelet | Nível |
|---|---|---|---:|
| ViT | RGB | — | — |
| Haar-ViT | wavelet | Haar | 1 |
| db2-ViT | wavelet | db2 | 1 |
| db4-ViT | wavelet | db4 | 1 |

---

## Tabela 3 — Resultados principais

| Modelo | Accuracy | Precision | Recall | Macro-F1 |
|---|---:|---:|---:|---:|
| ViT | | | | |
| Haar-ViT | | | | |
| db2-ViT | | | | |
| db4-ViT | | | | |

---

## Tabela 4 — Ablation study

| Entrada | Accuracy | Macro-F1 |
|---|---:|---:|
| LL | | |
| LH+HL+HH | | |
| LL+LH+HL+HH | | |

---

# 27. Estrutura proposta do artigo

O limite máximo será de aproximadamente **6.000 palavras**.

Sugestão de distribuição:

| Seção | Palavras |
|---|---:|
| Abstract | 200–250 |
| Introduction | 800–1.000 |
| Materials and Methods | 1.700–2.000 |
| Results | 900–1.100 |
| Discussion | 900–1.100 |
| Conclusions | 250–350 |
| **Total aproximado** | **4.750–5.800** |

Referências, tabelas e legendas normalmente não devem ser consideradas nessa estimativa até que as normas específicas da revista sejam confirmadas.

---

# 28. Abstract estruturado

O resumo deverá seguir aproximadamente:

### Background

Apresentar o problema da classificação automática de dentes e a necessidade de representação adequada das características morfológicas.

### Methods

Descrever:

- dataset de imagens intraorais;
- bounding boxes YOLO;
- extração dos dentes;
- ViT;
- DWT;
- wavelets Haar, db2 e db4;
- métricas.

### Results

Inserir posteriormente os principais resultados quantitativos.

### Conclusions

Indicar se a representação wavelet melhorou ou não o desempenho do ViT e quais componentes apresentaram maior contribuição.

---

# 29. Introdução — estrutura narrativa

A introdução deverá conter aproximadamente quatro blocos.

## Parágrafo 1 — contexto

Apresentar a importância crescente da análise automatizada de imagens odontológicas.

## Parágrafo 2 — classificação baseada em deep learning

Apresentar CNNs e Vision Transformers aplicados a imagens.

## Parágrafo 3 — limitação e oportunidade

Explicar que a morfologia dentária contém informação em diferentes escalas espaciais e que representações convencionais podem não separar explicitamente essas frequências.

## Parágrafo 4 — wavelets

Apresentar DWT como ferramenta de análise multiescala.

## Parágrafo 5 — lacuna

Estabelecer que ainda é relevante investigar experimentalmente se a introdução explícita de componentes wavelet melhora a classificação de dentes usando Vision Transformers.

## Parágrafo final — objetivo

> This study evaluates whether wavelet-based multiscale representations improve Vision Transformer performance for tooth classification in intraoral photographs. We compare conventional RGB input with Haar, Daubechies-2, and Daubechies-4 wavelet representations and perform an ablation analysis to quantify the contribution of low- and high-frequency components.

---

# 30. Materials and Methods — subseções

## 30.1 Dataset

Descrever:

- origem;
- número total de imagens;
- resolução;
- número de pacientes, se disponível;
- protocolo de captura, se disponível;
- classes;
- número de objetos;
- licença e aspectos éticos.

---

## 30.2 Annotation format

Explicar o formato YOLO.

---

## 30.3 Tooth extraction

Descrever conversão das bounding boxes e geração dos recortes.

---

## 30.4 Data partition

Explicar o split por paciente ou imagem.

---

## 30.5 Image preprocessing

Descrever resize, normalização e augmentation.

---

## 30.6 Discrete Wavelet Transform

Apresentar a fundamentação matemática necessária.

---

## 30.7 Vision Transformer

Descrever resumidamente o modelo.

---

## 30.8 Wavelet-ViT representation

Descrever como as bandas são utilizadas na entrada do Transformer.

---

## 30.9 Training procedure

Incluir:

- optimizer;
- learning rate;
- epochs;
- batch size;
- early stopping;
- hardware.

---

## 30.10 Evaluation metrics

Apresentar Accuracy, Precision, Recall e Macro-F1.

---

# 31. Results — organização

## 31.1 Dataset distribution

Apresentar quantidade de objetos por classe.

## 31.2 Main comparison

Comparar:

\[
ViT
\]

com:

\[
Haar-ViT,
\]

\[
db2-ViT,
\]

\[
db4-ViT.
\]

## 31.3 Per-class performance

Apresentar F1 por classe.

## 31.4 Confusion analysis

Identificar classes mais confundidas.

## 31.5 Wavelet ablation

Comparar LL, HF e representação completa.

---

# 32. Discussion — questões orientadoras

A discussão deverá responder diretamente:

1. Wavelets melhoraram a classificação?

2. Qual wavelet apresentou melhor desempenho?

3. A diferença foi consistente?

4. As bandas de alta frequência foram importantes?

5. Quais classes apresentaram maior dificuldade?

6. Os erros parecem estar associados à similaridade morfológica?

7. O ganho observado justifica a complexidade adicional?

8. Como os resultados se relacionam com trabalhos anteriores de classificação dentária e visão multiescala?

---

# 33. Limitações

O artigo deverá reconhecer:

- possível tamanho reduzido do dataset;
- origem única das imagens, caso aplicável;
- ausência de validação externa;
- dependência das bounding boxes previamente anotadas;
- ausência de avaliação clínica completa;
- possível desbalanceamento entre classes;
- possível presença de variações na aquisição das imagens;
- utilização de apenas três famílias wavelet;
- avaliação restrita à classificação e não à detecção.

---

# 34. Conclusão esperada

A conclusão deverá ser baseada estritamente nos resultados.

Estrutura:

1. retomar o objetivo;
2. informar se wavelets melhoraram ou não o ViT;
3. indicar a wavelet de melhor desempenho;
4. indicar a importância relativa das bandas;
5. destacar a viabilidade da abordagem;
6. sugerir como trabalho futuro a extensão para detecção automática.

---

# 35. Trabalhos futuros

Possíveis extensões:

- detecção e classificação simultâneas;
- integração com YOLO;
- DETR com representação wavelet;
- segmentação de dentes;
- Transformers hierárquicos;
- Wave-ViT completo;
- análise de robustez;
- imagens de diferentes centros;
- validação externa;
- classificação anatômica completa;
- interpretabilidade;
- aprendizado auto-supervisionado.

Esses itens não devem ser adicionados ao estudo inicial.

---

# 36. Critério para decidir se o paper está pronto

O artigo poderá ser considerado experimentalmente suficiente quando houver:

- dataset organizado;
- mapeamento correto das classes;
- split sem leakage;
- baseline ViT;
- três versões wavelet;
- métricas completas;
- resultados por classe;
- matriz de confusão;
- ablation study;
- repetição com seeds;
- discussão dos ganhos e limitações;
- código reprodutível.

---

# 37. Etapas práticas do projeto

## Etapa 1

Identificar o significado das classes no arquivo `data.yaml`.

## Etapa 2

Contar:

\[
N_{\text{images}}
\]

e:

\[
N_k
\]

para cada classe.

## Etapa 3

Verificar se existem IDs de pacientes.

## Etapa 4

Criar o split.

## Etapa 5

Gerar os crops.

## Etapa 6

Treinar ViT baseline.

## Etapa 7

Implementar Haar-ViT.

## Etapa 8

Adicionar db2 e db4.

## Etapa 9

Executar análise de ablação.

## Etapa 10

Produzir tabelas e figuras.

## Etapa 11

Realizar análise estatística.

## Etapa 12

Redigir o artigo.

---

# 38. Estrutura mínima de diretórios

```text
wavelet_tooth/
├── data/
│   ├── images/
│   ├── labels/
│   ├── splits/
│   └── crops/
├── src/
│   ├── prepare_dataset.py
│   ├── wavelet.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── configs/
│   ├── vit.yaml
│   ├── haar.yaml
│   ├── db2.yaml
│   └── db4.yaml
├── experiments/
├── results/
│   ├── tables/
│   ├── figures/
│   └── checkpoints/
├── requirements.txt
└── README.md
```

---

# 39. Contribuição principal do artigo

A contribuição deve ser apresentada de forma moderada.

Não:

> We propose a revolutionary new Wave-ViT architecture.

Preferir:

> We provide an experimental evaluation of wavelet-based multiscale representations for Vision Transformer-based tooth classification in intraoral photographs.

As contribuições podem ser apresentadas como:

1. avaliação sistemática de ViT convencional e representações wavelet;
2. comparação entre Haar, db2 e db4;
3. análise de baixa e alta frequência;
4. avaliação por classe em imagens intraorais.

---

# 40. Formulação final do projeto

## Problema

Classificação automática de dentes individuais em fotografias intraorais.

## Dados

Imagens intraorais anotadas com bounding boxes YOLO e classes 0–8.

## Baseline

Vision Transformer com entrada RGB.

## Método avaliado

Vision Transformer com representação baseada em DWT.

## Wavelets

Haar, db2 e db4.

## Principal métrica

Macro-F1.

## Principal análise complementar

Ablation study:

\[
LL
\quad vs.\quad
LH+HL+HH
\quad vs.\quad
LL+LH+HL+HH.
\]

## Pergunta central

\[
\boxed{
\text{Do wavelet representations improve Vision Transformer-based tooth classification?}
}
\]

Esse desenho mantém o estudo suficientemente simples para execução por estudantes, mas preserva uma pergunta científica objetiva, um protocolo reprodutível e uma contribuição experimental compatível com um Research Article de até aproximadamente 6.000 palavras.
