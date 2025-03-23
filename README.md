# Time Series Analysis Project

This time series analysis project focuses on studying global migration trends between 1990 and 2020. It uses various advanced time series analysis methods to understand patterns and dynamics of global emigration.

## Project Description

The project analyzes a recent dataset (2024) on global migration trends using several advanced statistical approaches:
- ARIMA and GARCH models for forecasting
- Copulas for analyzing dependencies between regions
- Rare event analysis
- Hawkes processes for modeling self-excitation dynamics

## Dataset

The dataset is sourced from Kaggle: [Global Emigration Trends 1990-2020](https://www.kaggle.com/datasets/shreyasur965/global-emigration-trends-1990-2020)

It contains information about:
- Total number of emigrants by region
- Annual migration trends
- Regional variations in emigration

## Project Structure

```
timeseries/
├── data/
│   └── total-number-of-emigrants.csv    # Main dataset
├── scripts/
│   ├── arima_optimization.py           # ARIMA analysis with optimal parameter search
│   ├── copula_analysis.py              # Copula analysis between regions
│   ├── hawkes_analysis.py              # Hawkes process modeling
│   └── garch_analysis.py               # GARCH volatility analysis
├── .venv/                              # Virtual environment
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/Trick5t3r/timeseries.git
cd timeseries
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
- On Windows:
```bash
.venv\Scripts\activate
```
- On Unix or MacOS:
```bash
source .venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Each script can be run independently to perform a specific analysis:

```bash
# ARIMA Analysis
python scripts/arima_optimization.py

# Copula Analysis
python scripts/copula_analysis.py

# Hawkes Analysis
python scripts/hawkes_analysis.py

# GARCH Analysis
python scripts/garch_analysis.py
```

## Contributors

- Théo LE PENDEVEN
- Eliott PRADELEIX
- Léandre SIMON

## Main Dependencies

- pandas
- numpy
- matplotlib
- scipy
- statsmodels
- seaborn
- scikit-learn

## Notes

This project is under development and has no specific license. 