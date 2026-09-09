# Análise Exploratória de Dados (EDA) — Etapas 1 e 2

**Projeto:** *Wavelet-Enhanced Vision Transformer for Tooth Classification in Intraoral Images*  
**Data:** 09/09/2026  
**Status das Etapas:** Concluídas (Etapa 1: Identificação de Classes e Etapa 2: Contagem e Distribuição)

---

## 1. Visão Geral e Estatísticas Globais do Dataset

A base de dados é composta por fotografias intraorais acompanhadas de anotações no formato YOLO ($[\text{class}, x_c, y_c, w, h]$).

### Tabela 1: Resumo Global do Dataset de Imagens Intraorais

| Métrica | Valor | Descrição |
|---|---:|---|
| **Total de Arquivos de Imagem** | 174 | Fotografias intraorais rotuladas no conjunto |
| **Total de Arquivos de Rótulo (.txt)** | 174 | Arquivos YOLO correspondentes |
| **Imagens-Base Únicas (Stems)** | 58–59 | Conjuntos de imagens originais (3 versões aumentadas por foto via Roboflow) |
| **Total de Bounding Boxes (Dentes)** | 3.672 | Amostras individuais de dentes anotadas para recorte |
| **Resolução Padrão das Imagens** | 640 × 640 px | Dimensões nativas das imagens no conjunto de dados |
| **Média de Dentes por Imagem** | 21,10 ± 4,25 | Dentes visíveis e anotados por fotografia intraoral |
| **Intervalo de Dentes por Imagem** | 0 a 26 | Variação de arcada visível (mediana: 22 dentes) |

---

## 2. Etapa 1 & 2: Distribuição de Frequência e Morfologia das Classes

O conjunto contém 9 classes numéricas observadas ( \in \{0, 1, 2, 3, 4, 5, 6, 7, 8\}$).

### Tabela 2: Distribuição Quantitativa e Propriedades Morfológicas das Classes Dentárias

| Class ID | Amostras ($) | Proporção (%) | Largura Média ($) | Altura Média ($) | Aspect Ratio (/h$) | Centroide Médio (, y_c$) | Posição Anatômica Inferida |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **0** | 333 | 9,07% | 0,0947 ± 0,0355 | 0,2909 ± 0,0682 | 0,3618 ± 0,2782 | (0,5037, 0,6246) | Mandíbula (Arcada Inferior) |
| **1** | 330 | 8,99% | 0,0964 ± 0,0347 | 0,3293 ± 0,0638 | 0,3527 ± 0,4651 | (0,5063, 0,3481) | Maxila (Arcada Superior) |
| **2** | 324 | 8,82% | 0,0932 ± 0,0276 | 0,2303 ± 0,0680 | 0,4443 ± 0,2428 | (0,5102, 0,6655) | Mandíbula (Incisivos Centrais/Laterais) |
| **3** | 342 | 9,31% | 0,1540 ± 0,0324 | 0,3857 ± 0,0559 | 0,4123 ± 0,1572 | (0,5050, 0,3859) | Maxila (Incisivos Centrais - Maior área) |
| **4** | 315 | 8,58% | 0,0929 ± 0,0330 | 0,2570 ± 0,0694 | 0,3994 ± 0,2613 | (0,5097, 0,6424) | Mandíbula (Pré-molares/Caninos inferiores) |
| **5** | 336 | 9,15% | 0,0972 ± 0,0318 | 0,3031 ± 0,0640 | 0,3465 ± 0,2395 | (0,5059, 0,3731) | Maxila (Incisivos Laterais / Caninos superiores) |
| **6** | 333 | 9,07% | 0,0812 ± 0,0250 | 0,2211 ± 0,0531 | 0,4227 ± 0,4131 | (0,5072, 0,5857) | Mandíbula (Região de transição inferior) |
| **7** | 330 | 8,99% | 0,0637 ± 0,0264 | 0,2587 ± 0,0549 | 0,3613 ± 0,9665 | (0,5087, 0,3515) | Maxila (Região lateral superior / Pré-molares) |
| **8** | 1.029 | 28,02% | 0,0626 ± 0,0178 | 0,1938 ± 0,0447 | 0,3509 ± 0,2459 | (0,4873, 0,4117) | Molares / Região Posterior Bilateral |
| **Total** | **3.672** | **100,00%** | **0,0832 ± 0,0347** | **0,2477 ± 0,0763** | **0,3717 ± 0,3891** | **(0,5012, 0,4820)** | — |

---

## 3. Principais Descobertas e Insights Anatômicos/Experimentais

1. **Balanceamento entre Classes 0 a 7:**
   - As classes de 0 a 7 apresentam volume equilibrado, variando entre 315 e 342 dentes anotados cada (aproximadamente \%$ por classe).
   - A classe 8 é a mais prevalente ( = 1.029$, ,02\%$), pois agrupa os múltiplos dentes posteriores (molares superiores e inferiores dos quatro quadrantes).

2. **Diferenciação Espacial Vertical (Maxila vs. Mandíbula):**
   - As classes 1, 3, 5 e 7 concentram-se na porção superior da imagem ( \in [0,34, 0,39]$), correspondendo à **arcada dentária superior (maxilar)**.
   - As classes 0, 2, 4 e 6 concentram-se na porção inferior da imagem ( \in [0,58, 0,67]$), correspondendo à **arcada dentária inferior (mandibular)**.

3. **Morfologia e Dimensões dos Dentes:**
   - A **Classe 3** possui a maior largura ( = 0,1540$) e altura ( = 0,3857$), compatível anatomicamente com os **incisivos centrais superiores**, que apresentam as maiores coroas clínicas vestibulares anteriores.
   - A **Classe 8** apresenta menores dimensões normalizadas ( = 0,0626, h = 0,1938$) devido ao encurtamento em perspectiva dos dentes posteriores nas fotos frontais intraorais.

4. **Implicações para o Treinamento do ViT:**
   - O uso de **Macro-F1** como métrica principal é essencial para evitar que a Classe 8 domine a avaliação.
   - A ponderação de classes na Cross-Entropy ($\mathcal{L}_{weighted}$) conforme proposto no plano do projeto ( = \frac{N}{K N_k}$) garantirá convergência equilibrada.
   - A divisão do dataset em treino/validação/teste deve ser feita estritamente com base no ase_stem (imagem-base original) para garantir que versões aumentadas não vazem entre partições.

---

## 4. Figuras Geradas para o Artigo (300 DPI, Padrão Internacional)

Todas as figuras foram salvas no diretório 
esults/figures/ com resolução de 300 DPI, prontas para inclusão no manuscrito.

- **Figura 1: Distribuição das Classes Dentárias**
  - Arquivo: 
esults/figures/fig1_class_distribution.png (300 DPI)
  - *Gráfico de barras detalhando o número absoluto de recortes e o percentual de cada classe dentária anotada no dataset.*
- **Figura 2: Distribuição Espacial Normalizada (, y_c$)**
  - Arquivo: 
esults/figures/fig2_spatial_distribution.png (300 DPI)
  - *Mapa de dispersão das posições centrais normalizadas de cada classe no plano da imagem intraoral, evidenciando a separação anatômica entre maxila e mandíbula.*
- **Figura 3: Características Morfológicas das Bounding Boxes**
  - Arquivo: 
esults/figures/fig3_bbox_dimensions.png (300 DPI)
  - *Boxplots comparativos de largura normalizada, altura normalizada e razão de aspecto (/h$) entre as 9 classes dentárias.*
- **Figura 4: Frequência de Dentes Anotados por Fotografia Intraoral**
  - Arquivo: 
esults/figures/fig4_teeth_per_image.png (300 DPI)
  - *Histograma com curva de densidade estimada (KDE) da quantidade de dentes anotados por imagem intraoral.*

---

## 5. Texto Sugerido para a Seção de Materiais e Métodos do Artigo

*(O trecho abaixo pode ser diretamente aproveitado na redação do artigo científico)*

> ### Dataset and Exploratory Analysis
> The intraoral dataset consists of 174 photographic images ( \times 640$ pixels) derived from 59 unique intraoral capture instances. A total of 3,672 individual tooth instances were annotated with bounding boxes following the YOLO format $[c, x_c, y_c, w, h]$, encompassing 9 distinct dental classes ( \in \{0, \dots, 8\}$). The average number of annotated teeth per intraoral photograph was .10 \pm 4.25$ (range: 0–26 teeth per frame).
>
> Classes 0 through 7 exhibit balanced representation, each comprising between 315 and 342 annotations ($\approx 9\%$ per class), corresponding to anterior and premolar maxillary and mandibular teeth. Class 8 accounts for 1,029 annotations (.02\%$), representing posterior molars across quadrants. Spatial centroid analysis demonstrates distinct anatomical clustering: maxillary classes (IDs 1, 3, 5, 7) are localized within  \in [0.34, 0.39]$, while mandibular classes (IDs 0, 2, 4, 6) reside within  \in [0.58, 0.67]$.
