# Telco Customer Churn — Previsão de cancelamento de clientes

Projeto de Machine Learning sobre a base **[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)** do Kaggle (dados de exemplo da IBM).
O objetivo é prever quais clientes de uma operadora de telecomunicações vão cancelar o serviço (*churn*) e entender **quais fatores estão associados ao cancelamento**, para apoiar ações de retenção.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)

---

## Sobre a base

- **7.043 clientes** e **21 colunas**: dados demográficos, serviços contratados (telefone, internet, segurança online, suporte técnico, streaming…), informações da conta (tempo de casa, contrato, forma de pagamento, valores cobrados) e o alvo `Churn`.
- **Classes desbalanceadas:** 26,5% dos clientes cancelaram. Um modelo que sempre prevê "não cancela" acerta 73%, por isso a acurácia não é usada como métrica principal — o foco está na classe churn (**Average Precision**, ROC-AUC, precisão, sensibilidade e F1).

## Etapas do projeto

| # | Notebook | Conteúdo | Status |
|---|---|---|---|
| 01 | [`01_preparacao_dados`](notebooks/01_preparacao_dados.ipynb) | Tipos, valores ausentes, duplicados, inconsistências, codificação e divisão treino/teste | ✅ |
| 02 | [`02_analise_exploratoria`](notebooks/02_analise_exploratoria.ipynb) | Taxa de churn por grupo com intervalos de confiança | ✅ |
| 03 | [`03_selecao_transformacao`](notebooks/03_selecao_transformacao.ipynb) | Codificação, padronização, redundâncias (VIF), WoE/IV e engenharia de variáveis | ✅ |
| 04 | Construção dos modelos | Regressão Logística, KNN, Árvore de Decisão, Random Forest e Gradient Boosting com busca de hiperparâmetros | 🚧 |
| 05 | Comparação de métodos | Ponderação de classes × distribuição original | 🚧 |
| 06 | Avaliação | Métricas no teste, curvas ROC/PR, escolha do limiar e captura no top 10% | 🚧 |
| 07 | Balanceamento | Subamostragem, sobreamostragem e SMOTE-NC dentro do pipeline | 🚧 |
| 08 | Interpretação | Coeficientes, importância por permutação, dependência parcial | 🚧 |

## Principais resultados até aqui

### Preparação dos dados
- `TotalCharges` vem como texto; os 11 valores em branco são todos de clientes com `tenure = 0` (ainda não cobrados) e foram preenchidos com 0.
- As categorias "No internet service" / "No phone service" são totalmente determinadas por `InternetService` / `PhoneService` e foram unificadas com "No".
- Divisão **estratificada 80/20** (semente 42), feita uma única vez; todas as decisões seguintes usam apenas o treino.

### Análise exploratória
Perfis com maior taxa de cancelamento (taxa geral = 26,5%):

| Característica | Taxa de churn | n |
|---|---|---|
| Tempo de casa de 0–3 meses | 56,2% | 1.062 |
| Pagamento por *electronic check* | 45,3% | 2.365 |
| Contrato mensal | 42,7% | 3.875 |
| Internet por fibra óptica | 41,9% | 3.096 |
| Sem OnlineSecurity / TechSupport | ~42% | ~3.500 |
| Idosos (SeniorCitizen) | 41,7% | 1.142 |

O perfil de maior risco é o cliente **novo, com contrato mensal e fibra óptica, sem serviços de segurança e suporte**.

### Seleção e transformação das variáveis
- `MonthlyCharges` é praticamente a soma dos preços dos serviços contratados (R² = 0,999, VIF ≈ 860), e `TotalCharges` é redundante com `tenure × MonthlyCharges`.
- Foram criadas 5 variáveis (`log_tenure`, `n_servicos_adicionais`, `servicos_suporte`, `variacao_mensalidade`, `mensal_x_fibra`). Elas elevam a AP da Regressão Logística de **0,661 para 0,671** (melhor em 4 de 5 folds); o Random Forest praticamente não muda (os conjuntos foram comparados com Regressão Logística e Random Forest, um modelo de cada família).
- A remoção de variáveis com baixo *Information Value* **piorou** os resultados: pouca informação individual não significa pouca informação em conjunto.
- Por isso, dois conjuntos seguem para a construção dos modelos: o **base** e o **base + variáveis criadas**, para medir o ganho real das novas variáveis com hiperparâmetros ajustados (e regularização, quando for o caso).

## Metodologia

- Todo pré-processamento (one-hot, padronização, reamostragem) fica **dentro de pipelines** do scikit-learn, ajustado apenas nos folds de treino — sem vazamento de dados.
- Todos os modelos são comparados nos **mesmos 5 folds** de validação cruzada estratificada, o que permite comparações pareadas fold a fold.
- O conjunto de teste é usado apenas para relatar o desempenho final, nunca para tomar decisões.

## Estrutura

```
├── data/
│   ├── raw/telco_churn_data.csv     # base original (Kaggle)
│   └── processed/                   # arquivos gerados pelos notebooks
├── notebooks/                       # um notebook por etapa
└── requirements.txt
```

## Como executar

```bash
git clone https://github.com/imarcososhiro/telco-churn-ml.git
cd telco-churn-ml
pip install -r requirements.txt
jupyter lab
```

Execute os notebooks **em ordem** (01 → 03): cada um salva em `data/processed/` os arquivos que o próximo lê.

## Tecnologias

Python · pandas · NumPy · SciPy · scikit-learn · imbalanced-learn · Matplotlib · Seaborn · Jupyter

## Autor

**Marcos Oshiro** — [GitHub](https://github.com/imarcososhiro)
