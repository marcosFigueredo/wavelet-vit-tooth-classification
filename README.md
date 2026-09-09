# Wavelet-Enhanced Vision Transformer for Tooth Classification in Intraoral Images

Este repositório contém a implementação completa, reprodutível e modular do projeto de pesquisa que avalia a contribuição de representações multiescala baseadas em Transformadas Wavelet Discretas (DWT) em Vision Transformers (ViTs) para a classificação de dentes em imagens intraorais.

---

## 📂 Estrutura do Repositório

```text
├── configs/                   # Arquivos de configuração dos experimentos (.yaml)
│   ├── vit_rgb.yaml           # Baseline: Vision Transformer (RGB)
│   ├── haar_vit.yaml          # Wavelet-ViT com wavelet Haar
│   ├── db2_vit.yaml           # Wavelet-ViT com wavelet Daubechies-2
│   ├── db4_vit.yaml           # Wavelet-ViT com wavelet Daubechies-4
│   ├── ablation_haar_ll.yaml  # Ablação: Apenas baixa frequência (LL)
│   └── ablation_haar_hf.yaml  # Ablação: Apenas alta frequência (LH + HL + HH)
├── data/
│   ├── splits/                # Metadados e manifests de divisão (treino/val/teste)
│   └── crops/                 # Dentes individuais extraídos (224x224 px)
├── src/                       # Código-fonte modular
│   ├── __init__.py
│   ├── utils.py               # Métricas, seeds, geração de gráficos 300 DPI
│   ├── wavelet.py             # Módulo 2D DWT (Haar, db2, db4, modos LL/HF/All)
│   ├── dataset.py             # PyTorch Dataset e Data Augmentation
│   ├── models.py              # Vision Transformer e Wavelet-ViT com adaptação de canais
│   ├── prepare_dataset.py     # Pipeline de split sem vazamento e extração de crops
│   ├── train.py               # Treinamento com pesos de classe e early stopping
│   └── evaluate.py            # Avaliação no teste e matriz de confusão
├── results/
│   ├── figures/               # Gráficos em alta resolução (300 DPI em inglês)
│   ├── tables/                # Tabelas de métricas e relatórios estatísticos
│   └── checkpoints/           # Pesos e histórico dos modelos treinados
├── eda_etapas_1_e_2.md        # Relatório de Análise Exploratória (Etapas 1 e 2)
├── requirements.txt           # Dependências do projeto
├── .gitignore                 # Arquivos ignorados pelo controle de versão
└── README.md                  # Documentação do projeto
```

---

## ⚙️ Instalação e Requisitos

1. Clone o repositório e acerte o ambiente:
```bash
git clone <URL_DO_REPOSITORIO>
cd teethDetection
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

---

## 🚀 Como Executar

### 1. Preparar o Dataset e Extrair os Recortes (Etapas 3, 4 e 5)
Executa a divisão agrupada por imagem-base original (70% treino, 15% validação, 15% teste) e extrai os dentes individuais com margem de 8% redimensionados para $224 	imes 224$:
```bash
python -m src.prepare_dataset
```

### 2. Treinar o Baseline ViT (RGB)
```bash
python -m src.train --config configs/vit_rgb.yaml
```

### 3. Treinar os Modelos Wavelet-ViT
```bash
# Haar-ViT
python -m src.train --config configs/haar_vit.yaml

# db2-ViT
python -m src.train --config configs/db2_vit.yaml

# db4-ViT
python -m src.train --config configs/db4_vit.yaml
```

### 4. Executar Estudos de Ablação (Frequência)
```bash
# Apenas Baixa Frequência (LL)
python -m src.train --config configs/ablation_haar_ll.yaml

# Apenas Alta Frequência (LH + HL + HH)
python -m src.train --config configs/ablation_haar_hf.yaml
```

### 5. Avaliação no Conjunto de Teste
```bash
python -m src.evaluate --checkpoint results/checkpoints/haar_vit/best_model.pth --split test
```

---

## 📊 Métricas e Resultados

Os resultados quantitativos (Macro-F1, Acurácia, Precisão, Recall e Matriz de Confusão em 300 DPI) são automaticamente salvos em `results/tables/` e `results/figures/`.
