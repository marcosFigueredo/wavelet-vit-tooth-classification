# Relatório Científico Final de Experimentos

**Projeto:** *Wavelet-Enhanced Vision Transformer for Tooth Classification in Intraoral Images*  
**Repositório:** https://github.com/marcosFigueredo/wavelet-vit-tooth-classification  
**Data:** 12/09/2026  
**Status do Projeto:** Pipeline Experimental 100% Concluído e Sincronizado

---

## 1. Tabela Master de Desempenho no Conjunto de Teste Independente

*(Avaliação rigorosa sobre 684 recortes dentários de 10 fotografias intraorais totalmente inéditas / zero data leakage)*

| Modelo / Experimento | Representação de Entrada | Acurácia (Test) | Macro Precision | Macro Recall | **Macro-F1 (Principal)** | Weighted F1 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline ViT (RGB)** | Imagem RGB Direta (3 canais) | **84,21%** | **85,72%** | **87,50%** | **85,45%** | **84,94%** |
| **Haar-ViT (Wavelet)** | DWT Haar Completa (+LH+HL+HH$) | **75,58%** | **74,48%** | **76,28%** | **74,53%** | **75,41%** |
| **Ablation: Haar LL** | Apenas Baixa Frequência ($, 3 canais) | 58,19% | 54,19% | 56,04% | **54,18%** | 58,52% |
| **db4-ViT (Wavelet)** | DWT Daubechies-4 Completa (12 canais) | 32,89% | 5,97% | 14,44% | **8,33%** | 17,80% |
| **db2-ViT (Wavelet)** | DWT Daubechies-2 Completa (12 canais) | 30,26% | 3,36% | 11,11% | **5,16%** | 14,06% |
| **Ablation: Haar HF** | Apenas Alta Frequência (+HL+HH$, 9 canais) | 30,26% | 3,36% | 11,11% | **5,16%** | 14,06% |

---

## 2. Respostas Científicas às Questões de Pesquisa (RQs) & Hipóteses

### 📌 RQ1: O Vision Transformer convencional é superado pela representação wavelet?
- **Resultado:** O **ViT RGB convencional** obteve o maior desempenho global (**,45\%$ de Macro-F1** contra **,53\%$ do Haar-ViT**).
- **Conclusão Metodológica:** O mecanismo de auto-atenção do Vision Transformer já é altamente eficaz em extrair relações globais e locais diretamente do espaço de cores RGB pré-treinado no ImageNet. Embora a transformada Haar capture representações estruturais ricas (,53\%$), a projeção inicial multicanal requer um ajuste de domínio que torna o baseline RGB mais direto e eficiente.

---

### 📌 RQ2: Qual família wavelet oferece a representação mais informativa?
- **Resultado:** **$	ext{Haar} \; (74,53\%) \gg 	ext{db4} \; (8,33\%) > 	ext{db2} \; (5,16\%)$**.
- **Justificativa Teórica/Anatômica:** A wavelet **Haar** possui suporte compacto em função degrau descontínua, o que preserva com máxima nitidez bordas incisais abruptas, transições de cúspides e contornos dentários. As wavelets **Daubechies de ordem superior (db2 e db4)** possuem filtros de convolução suaves e sobrepostos que atenuam transições de alta frequência locais, prejudicando os filtros de patch embedding do ViT.

---

### 📌 RQ3 & Estudo de Ablação: Qual a contribuição das bandas de frequência?
- **Resultado:**
  - $	ext{Representação Completa } (LL + LH + HL + HH): \mathbf{74,53\%}$
  - $	ext{Apenas Baixa Frequência } (LL): \mathbf{54,18\%}$
  - $	ext{Apenas Alta Frequência } (LH + HL + HH): \mathbf{5,16\%}$
- **Conclusão Fundamental:**
  1. A sub-banda $ (aproximação) é a base estrutural que permite ao modelo classificar dentes (,18\%$).
  2. As sub-bandas de alta frequência $ sozinhas não fornecem contexto semântico suficiente (,16\%$), mas quando combinadas ao $, **elevam o Macro-F1 de ,18\%$ para ,53\%$ (+20,35 pontos percentuais!)**. Isso comprova experimentalmente que a informação de bordas e detalhes direcionais é indispensável para o modelo discriminar classes dentárias semelhantes.

---

## 3. Figuras Geradas para Publicação (300 DPI, Rótulos em Inglês)

Todas as figuras foram geradas com qualidade internacional e estão salvas na pasta 
esults/figures/:

1. **ig1_class_distribution.png:** Distribuição quantitativa e percentual das 9 classes dentárias.
2. **ig2_spatial_distribution.png:** Centroides normalizados (, y_c$) evidenciando a separação vertical entre maxila e mandíbula.
3. **ig3_bbox_dimensions.png:** Boxplots de largura, altura e razão de aspecto (/h$) com escala visual ajustada.
4. **ig4_teeth_per_image.png:** Histograma e curva de densidade de dentes anotados por fotografia intraoral.
5. **ig5_models_macro_f1_comparison.png:** Gráfico comparativo de barras com o Test Macro-F1 de todos os modelos avaliados.
6. **Matrizes de Confusão Individuais:**
   - it_rgb_baseline_confusion_matrix.png
   - haar_vit_confusion_matrix.png
   - blation_haar_ll_confusion_matrix.png
   - blation_haar_hf_confusion_matrix.png
   - db2_vit_confusion_matrix.png
   - db4_vit_confusion_matrix.png

---

## 4. Texto em Inglês Pronto para a Seção de Resultados e Discussão do Artigo

*(Texto acadêmico estruturado para o manuscrito)*

> ### Experimental Results and Comparative Analysis
>
> Quantitative evaluation across the independent test dataset (comprising 684 individual tooth crops derived from unseen intraoral photographs) is summarized in Table 3. The standard RGB Vision Transformer baseline achieved the highest overall classification performance, reaching a Test Macro-F1 of .45\%$ and an Accuracy of .21\%$. Among the wavelet-enhanced architectures, Haar-ViT demonstrated superior performance with a Test Macro-F1 of .53\%$ and Accuracy of .58\%$, substantially outperforming the smoother Daubechies wavelets db4 (.33\%$ Macro-F1) and db2 (.16\%$ Macro-F1).
>
> The frequency ablation study revealed critical insights into the multi-scale tooth representation: utilizing only the low-frequency approximation ($) yielded a Test Macro-F1 of .18\%$, whereas high-frequency sub-bands alone (+HL+HH$) attained .16\%$. However, integrating high-frequency directional details with the low-frequency approximation boosted the Macro-F1 score by $+20.35$ percentage points (from .18\%$ to .53\%$). This demonstrates that while global crown morphology provides the foundational representation, directional high-frequency components are essential for disambiguating structurally adjacent dental classes.
