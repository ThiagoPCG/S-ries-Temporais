import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Simulação de dados históricos de acidentes (2021-2023) com sazonalidade e ruídos
np.random.seed(42)
dates = pd.date_range(start='2021-01-01', end='2023-12-01', freq='MS')
acidentes = [
    200 + 10 * i + 30 * np.sin(2 * np.pi * i / 12) + np.random.normal(0, 20) 
    for i in range(len(dates))
]
df = pd.DataFrame({'Data': dates, 'Acidentes': acidentes})
df.set_index('Data', inplace=True)

# Aplicação do modelo SARIMA (p=1, d=1, q=1) com sazonalidade (P=1, D=1, Q=1, S=12)
model = SARIMAX(
    df['Acidentes'], 
    order=(1, 1, 1), 
    seasonal_order=(1, 1, 1, 12),  # Sazonalidade anual (12 meses)
    enforce_stationarity=False, 
    enforce_invertibility=False
)
model_fit = model.fit(disp=False)

# Fazendo previsões para os próximos 2 anos (24 meses)
future_dates = pd.date_range(start='2024-01-01', periods=24, freq='MS')
forecast = model_fit.get_forecast(steps=24)
forecast_values = forecast.predicted_mean

# Criando uma série para as previsões com índice futuro
forecast_series = pd.Series(forecast_values.values, index=future_dates)

# Combinação contínua de dados históricos e previsão para manter a conexão
combined_series = pd.concat([df['Acidentes'], forecast_series])

# Plotando dados históricos e previsão conectados, mas com cores diferentes
plt.figure(figsize=(14, 7))
plt.plot(combined_series[:'2023-12-01'], marker='o', color='orange', label='Dados Históricos')  # Histórico em laranja
plt.plot(combined_series['2023-12-01':], marker='o', color='blue', label='Previsão')  # Previsão em azul
plt.axvline(x=pd.Timestamp('2023-12-01'), color='gray', linestyle='--', label='Início da Previsão')  # Linha divisória
plt.title('Previsão de Acidentes em Recife com Sazonalidade (2021-2026)')
plt.xlabel('Data')
plt.ylabel('Número de Acidentes')
plt.legend()
plt.grid()
plt.show()
