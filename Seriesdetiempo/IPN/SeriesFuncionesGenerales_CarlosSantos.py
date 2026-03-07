# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:12:20 2026

@author: LENOVO
"""

# ============================================
# LIBRERÍAS BÁSICAS
# ============================================

import numpy as np              # Operaciones numéricas y manejo de arreglos
import pandas as pd             # Manipulación y análisis de datos en DataFrames

import os

# ============================================
# VISUALIZACIÓN
# ============================================

import matplotlib.pyplot as plt # Creación de gráficos y visualizaciones



# ============================================
# MÉTRICAS DE EVALUACIÓN
# ============================================

from sklearn.metrics import (
    mean_squared_error,  # Error cuadrático medio (MSE)
    mean_absolute_error, # Error absoluto medio (MAE)
    r2_score             # Coeficiente de determinación R²
)

# ============================================
# DIAGNÓSTICOS Y ANÁLISIS ESTADÍSTICO
# ============================================

from statsmodels.tsa.stattools import acf                # Cálculo de autocorrelación
from statsmodels.graphics.tsaplots import plot_acf       # Gráfico de autocorrelación
from statsmodels.graphics.gofplots import qqplot         # Gráfico QQ para normalidad
import statsmodels.api as sm                             # Funciones estadísticas generales
import scipy.stats as stats                              # Pruebas estadísticas

def fft_spectrum(x: np.ndarray, dt: float = 1.0):
    """
    Calcula el espectro de frecuencias de una señal usando FFT.

    Retorna:
    - freqs: vector de frecuencias positivas presentes en la señal
    - mag: magnitud normalizada del espectro

    Parámetro:
    dt: intervalo de muestreo entre observaciones.
        Si los datos están espaciados uniformemente (1 muestra por unidad de tiempo),
        entonces dt = 1.0
    """
    print(f"La serie tiene {x.ndim} dimension(es) (Tener cuidado con esto)\n\n ")
    # Convierte la entrada en un arreglo de numpy de tipo float
    x = np.asarray(x, dtype=float) # Esto asegura que las operaciones matemáticas funcionen correctamente
    x = x[~np.isnan(x)]   # elimina NaN por si acaso
    x = x.ravel()          # asegura que sea 1D, esto es muy importante
    # Centra la señal restando su media
    x = x - np.mean(x) # Esto elimina el componente DC (promedio de la señal)  y evita que domine el espectro de frecuencias

    # Número total de observaciones en la señal
    n = len(x)

    # Calcula la Transformada Rápida de Fourier para señales reales
    # rfft solo devuelve las frecuencias positivas (>= 0)  lo cual es suficiente cuando la señal es real
    X = np.fft.rfft(x)

    # Genera el vector de frecuencias asociado a cada componente de la FFT
    # n -> número de observaciones
    # d=dt -> intervalo entre muestras
    freqs = np.fft.rfftfreq(n, d=dt)

    # Calcula la magnitud del espectro
    # np.abs(X) obtiene la amplitud de cada componente de frecuencia
    # Se divide entre n para normalizar la magnitud
    mag = np.abs(X) / n
     # --- Resumen de la FFT ---
    #print(f"Tamaño de la serie original: {n}")
    #print(f"Tamaño de la FFT (frecuencias positivas): {len(X)}")
    #print(f"Primeros 5 valores de la FFT: {X[:5]}")
    #print(f"Tamaño del vector de frecuencias: {len(freqs)}")
    #print(f"Tamaño de la magnitud del espectro: {len(mag)}")
    # Retorna las frecuencias y su magnitud correspondiente
    return freqs, mag


def calcular_metricas(y_real, y_pred, n_params=0, dataset_name=""): # PARA SARIMA
    """
    Calcula métricas de evaluación para modelos de series temporales.
    """

    # Convertir a arrays
    y_real = np.array(y_real).flatten()
    y_pred = np.array(y_pred).flatten()

    # Igualar longitud
    min_len = min(len(y_real), len(y_pred))
    y_real = y_real[:min_len]
    y_pred = y_pred[:min_len]

    n = len(y_real)

    # ============================================================
    # MSE — Mean Squared Error
    # ============================================================
    #
    # MSE = (1/n) Σ (y_i − ŷ_i)^2
    #
    mse = mean_squared_error(y_real, y_pred)

    # ============================================================
    # RMSE — Root Mean Squared Error
    # ============================================================
    #
    # RMSE = √MSE
    #
    rmse = np.sqrt(mse)

    # ============================================================
    # MAE — Mean Absolute Error
    # ============================================================
    #
    # MAE = (1/n) Σ |y_i − ŷ_i|
    #
    mae = mean_absolute_error(y_real, y_pred)

    # ============================================================
    # SMAPE — Symmetric Mean Absolute Percentage Error
    # ============================================================
    #
    # SMAPE = 100/n Σ (2|ŷ − y| / (|y| + |ŷ|))
    #
    smape = 100 * np.mean(
        2 * np.abs(y_pred - y_real) /
        (np.abs(y_real) + np.abs(y_pred) + 1e-10)
    )

    # ============================================================
    # R² — Coeficiente de determinación
    # ============================================================
    #
    # R² = 1 − (SS_res / SS_tot)
    #
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)

    r2 = 1 - (ss_res / (ss_tot + 1e-10))

    # ============================================================
    # AIC y BIC
    # ============================================================
    #
    # AIC = n log(MSE) + 2k
    #
    # BIC = n log(MSE) + k log(n)
    #
    aic = bic = np.nan

    if n_params > 0 and n > 0:
        aic = n * np.log(mse) + 2 * n_params
        bic = n * np.log(mse) + n_params * np.log(n)

    # ============================================================
    # Error relativo respecto al rango de la serie
    # ============================================================

    data_range = np.max(y_real) - np.min(y_real)

    mse_percent = 100 * mse / (data_range ** 2 + 1e-10) if data_range > 0 else 0
    rmse_percent = 100 * rmse / data_range if data_range > 0 else 0

    return {
        f'MSE_{dataset_name}': mse,
        f'MSE_{dataset_name}_%': mse_percent,
        f'RMSE_{dataset_name}': rmse,
        f'RMSE_{dataset_name}_%': rmse_percent,
        f'MAE_{dataset_name}': mae,
        f'SMAPE_{dataset_name}_%': smape,
        f'R2_{dataset_name}': r2,
        f'AIC_{dataset_name}': aic,
        f'BIC_{dataset_name}': bic,
        f'n_{dataset_name}': n
    }

def forecast_sarima_conindice(modelo, test_df, steps):
    """
    Genera un pronóstico SARIMA para un conjunto de prueba y devuelve
    la media pronosticada junto con los intervalos de confianza,
    usando el índice del DataFrame de prueba.
    
    Parámetros:
    - modelo: modelo SARIMA ajustado (statsmodels SARIMAXResults)
    - test_df: DataFrame de prueba (para tomar el índice)
    - steps: número de pasos a predecir (generalmente len(test_df))
    
    Retorna:
    - mean: Serie con la media pronosticada, indexada igual que test_df
    - low: Serie con el límite inferior del intervalo de confianza
    - up: Serie con el límite superior del intervalo de confianza
    """

    # --- Generar predicción para los pasos futuros ---
    fc = modelo.get_forecast(steps=len(test_df))

    # --- Media pronosticada convertida a Serie con índice del DataFrame de prueba ---
    mean = pd.Series(fc.predicted_mean, index=test_df.index)

    # --- Intervalos de confianza al 95% ---
    ci = fc.conf_int(alpha=0.05)  # devuelve DataFrame con dos columnas: [lower, upper]

    # --- Extraer límites inferior y superior como Series con mismo índice ---
    low = pd.Series(ci[:, 0], index=test_df.index)  # límite inferior
    up  = pd.Series(ci[:, 1], index=test_df.index)  # límite superior

    # --- Retornar media y límites ---
    return mean, low, up

def forecast_sarima(modelo, y_train, steps): #Forecast para SARIMA
    """
    Genera pronóstico usando un modelo SARIMA.
    """

    try:
        fc = modelo.get_forecast(steps=steps) # Predicción futura
        mean = fc.predicted_mean # Media pronosticada
        ci = fc.conf_int(alpha=0.05) # Intervalos de confianza
        #print(ci)
        low = ci.iloc[:, 0]
        up = ci.iloc[:, 1]

        # Índice futuro
        idx_future = pd.RangeIndex(
            len(y_train) + 1,
            len(y_train) + 1 + steps
        )

        mean.index = idx_future
        low.index = idx_future
        up.index = idx_future

        return mean, low, up

    except Exception as e:

        #print(f"Error en forecast: {e}")

        idx_future = pd.RangeIndex(
            len(y_train) + 1,
            len(y_train) + 1 + steps
        )

        mean = pd.Series([np.nan] * steps, index=idx_future)
        low = pd.Series([np.nan] * steps, index=idx_future)
        up = pd.Series([np.nan] * steps, index=idx_future)

        return mean, low, up



# Función para aplicar diferenciación a una serie temporal
def apply_differencing(series: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Aplica diferenciación de orden 'order'.
    La diferenciación se usa en series de tiempo para eliminar tendencia
    y volver la serie más estacionaria.
    """
    diff_series = series.copy() # Crear una copia de la serie original para no modificar el arreglo original
    # Aplicar la diferenciación tantas veces como indique 'order' Cada iteración calcula la diferencia entre valores consecutivos
    for _ in range(order):
        diff_series = np.diff(diff_series, axis=0)
    return diff_series # Regresar la serie diferenciada


# Función para reconstruir la serie original a partir de sus diferencias
def reconstruct_series_from_diffs(diffs: np.ndarray, first_value: float) -> np.ndarray:
    """
    Reconstruye la serie original usando:
    - las diferencias consecutivas
    - el primer valor de la serie original (y0)
    """
    diffs = diffs.flatten()  # Asegurar que el arreglo de diferencias sea un vector unidimensional

    # Crear un arreglo vacío para almacenar la serie reconstruida 
    # Se suma 1 porque las diferencias siempre tienen un elemento menos que la serie original
    reconstructed = np.zeros(len(diffs) + 1, dtype=float)

    # Colocar el primer valor conocido de la serie original
    reconstructed[0] = float(first_value)

    # Reconstruir cada valor acumulando las diferencias
    # y_t = y_(t-1) + diff_(t-1)
    for i in range(1, len(reconstructed)):
        reconstructed[i] = reconstructed[i - 1] + diffs[i - 1]
    return reconstructed # Devolver la serie reconstruida



# Función para crear un conjunto de datos autorregresivo a partir de una serie temporal
def create_autoregressive_data(data: np.ndarray, delays: list[int], target: np.ndarray | None = None):
    """
    Crea un dataset con retrasos temporales (lags).

    X[t] = [x(t-d1), x(t-d2), ...]
    y[t] = target[t]

    Donde:
    - data: serie temporal original
    - delays: lista de retrasos (lags) que se utilizarán como variables explicativas
    - target: variable objetivo (si es distinta de la serie original)
    """

    # Obtener el retraso máximo para saber desde qué índice comenzar. Esto evita acceder a índices negativos en la serie
    max_delay = max(delays)
    X_delayed = [] # Lista donde se almacenarán las observaciones con retrasos
    for i in range(max_delay, len(data)): # Lista donde se almacenarán las observaciones con retrasos
        delayed_values = [] # Lista temporal para guardar los valores retrasados de una observación
        for d in sorted(delays): # Ordenar los retrasos para mantener consistencia en el orden de las variables
            delayed_values.extend(data[i - d])# Agregar el valor de la serie en el tiempo (t - d). Esto representa un lag de la serie
        X_delayed.append(delayed_values) # Agregar el valor de la serie en el tiempo (t - d). Esto representa un lag de la serie

    X_delayed = np.array(X_delayed, dtype=float) # Convertir la lista a un arreglo de numpy

    # Si se proporciona una variable objetivo
    if target is not None:

        y = target[max_delay:] # Ajustar la variable objetivo eliminando los primeros valores que no tienen suficientes retrasos
        return X_delayed, y # Regresar variables predictoras y objetivo

    # Si no hay variable objetivo, solo se regresan las variables con retrasos
    return X_delayed



# ===========RED NEURONAL - MÉTRICAS
def calculate_metrics(y_true, y_pred, n_params, dataset_name=""):
    """
    Calcula métricas de desempeño de predicción entre valores reales y predichos.
    
    Parámetros:
    - y_true: valores reales
    - y_pred: valores predichos
    - n_params: número de parámetros del modelo (para AIC/BIC)
    - dataset_name: nombre opcional del dataset (p. ej. 'train' o 'test')
    
    Retorna:
    Diccionario con varias métricas: MSE, RMSE, SMAPE, R2, AIC, BIC, tanto en valor absoluto como relativo.
    """

    # Convertir entradas a arrays planos de NumPy
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Ajustar longitud en caso de que y_true y y_pred tengan tamaños diferentes
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    n = len(y_true)  # Número de muestras

    # ==========================
    # Métricas de error absoluto
    # ==========================
    mse = np.mean((y_true - y_pred) ** 2)      # Error cuadrático medio
    rmse = np.sqrt(mse)                         # Raíz del MSE

    # Symmetric Mean Absolute Percentage Error (SMAPE)
    # Permite comparar errores en % considerando valores positivos y negativos
    smape = 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)
    )

    # ==========================
    # Correlación y R^2
    # ==========================
    corr = np.corrcoef(y_true, y_pred)          # Matriz de correlación de Pearson
    r2 = float(corr[0, 1] ** 2) if np.isfinite(corr[0, 1]) else np.nan  # R^2 como cuadrado de Pearson

    # ==========================
    # Criterios de información
    # ==========================
    # AIC (Akaike Information Criterion)
    aic = n * np.log(mse + 1e-12) + 2 * n_params
    # BIC (Bayesian Information Criterion)
    bic = n * np.log(mse + 1e-12) + n_params * np.log(max(n, 2))

    # ==========================
    # Métricas relativas (%)
    # ==========================
    data_range = np.max(y_true) - np.min(y_true)  # Rango de los datos reales
    mse_percent = 100 * mse / (data_range ** 2 + 1e-10)   # MSE relativo al rango
    rmse_percent = 100 * rmse / (data_range + 1e-10)     # RMSE relativo al rango

    # ==========================
    # Devolver resultados como diccionario
    # ==========================
    return {
        f"MSE_{dataset_name}": mse,
        f"MSE_{dataset_name}_%": mse_percent,
        f"RMSE_{dataset_name}": rmse,
        f"RMSE_{dataset_name}_%": rmse_percent,
        f"SMAPE_{dataset_name}_%": smape,
        f"R2_{dataset_name}_Pearson2": r2,
        f"AIC_{dataset_name}": aic,
        f"BIC_{dataset_name}": bic,
    }

def plot_forecast_indice(real, pred, lower=None, upper=None, 
                  title='Pronóstico con Intervalos de Confianza',
                  color='green'):
    """
    Grafica valores reales y predichos con el mismo color y opcional intervalo de confianza.

    Parámetros:
    -----------
    real : pd.Series o np.array
        Valores reales.
    pred : pd.Series o np.array
        Valores predichos.
    lower : pd.Series o np.array, opcional
        Límite inferior del intervalo de confianza.
    upper : pd.Series o np.array, opcional
        Límite superior del intervalo de confianza.
    title : str, opcional
        Título de la gráfica.
    color : str, opcional
        Color de la serie real y predicha.
    """
    # Índice
    idx = real.index if hasattr(real, 'index') else range(len(real))

    fig, ax = plt.subplots(figsize=(10,5))

    # Intervalo de confianza
    if lower is not None and upper is not None:
        ax.fill_between(
            idx,
            lower.values if hasattr(lower, 'values') else lower,
            upper.values if hasattr(upper, 'values') else upper,
            alpha=0.3,
            color='lightgray',
            label='IC 95%'
        )

    # Serie real
    ax.plot(idx, real.values if hasattr(real, 'values') else real, color=color, linestyle='-', alpha=0.7, label='y_test')

    # Serie predicha
    ax.plot(idx, pred.values if hasattr(pred, 'values') else pred, color=color, linestyle='--', alpha=0.7, label='y_test_pred')

    # Título y etiquetas
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Índice')
    ax.set_ylabel('Valor')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Rotar etiquetas del eje x
    plt.setp(ax.get_xticklabels(), rotation=90)

    plt.tight_layout()
    plt.show()

    return fig

def plot_series(serie, titulo, ylabel="Valor"):
    """
    Grafica una serie de tiempo de forma general.

    Parámetros
    ----------
    serie : pandas.Series
        Serie que se desea graficar.
    titulo : str
        Título de la gráfica.
    ylabel : str
        Nombre del eje Y.
    """

    # Crear la figura con tamaño estándar
    plt.figure(figsize=(12, 5))

    # Graficar la serie
    plt.plot(serie, linewidth=1)

    # Título de la gráfica
    plt.title(titulo)

    # Etiqueta del eje X dependiendo del tipo de índice
    if isinstance(serie.index, pd.DatetimeIndex):
        plt.xlabel("Fecha")
    else:
        plt.xlabel("Tiempo")

    # Etiqueta del eje Y definida por el usuario
    plt.ylabel(ylabel)

    # Activar cuadrícula
    plt.grid(True)

    # Ajustar márgenes
    plt.tight_layout()

    # Mostrar gráfica
    plt.show()


def graficar_diagnostico_modelo_SARIMA(
    y_train_actual,
    y_train_pred_vals,
    y_test_series,
    y_test_pred,
    y_test_low,
    y_test_up,
    metrics_in,
    metrics_out,
    modelo_final,
    ORDER,
    SEASONAL_ORDER,
    TREND,
    y_train_series
):
    """
    Genera un panel completo de diagnóstico para modelos de series de tiempo SARIMA.
    """

    # =========================================================
    # CREAR FIGURA PRINCIPAL
    # =========================================================
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # =========================================================
    # 1. SERIE TEMPORAL COMPLETA
    # =========================================================
    ax1 = axes[0, 0]

    ax1.plot(
        np.arange(len(y_train_actual)),
        y_train_actual,
        'b-',
        alpha=0.7,
        label='Entrenamiento (Real)'
    )

    ax1.plot(
        np.arange(len(y_train_actual)),
        y_train_pred_vals,
        'b--',
        alpha=0.7,
        label='Entrenamiento (Predicho)'
    )

    ax1.plot(
        np.arange(len(y_train_actual),
        len(y_train_actual) + len(y_test_series)),
        y_test_series.values,
        'g-',
        alpha=0.7,
        label='Prueba (Real)'
    )

    ax1.plot(
        np.arange(len(y_train_actual),
        len(y_train_actual) + len(y_test_series)),
        y_test_pred.values,
        'g--',
        alpha=0.7,
        label='Prueba (Predicho)'
    )

    ax1.axvline(
        x=len(y_train_actual),
        color='r',
        linestyle='--',
        alpha=0.5
    )

    ax1.set_title('Serie Temporal Completa', fontweight='bold')
    ax1.set_xlabel('Índice')
    ax1.set_ylabel('Valor')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # =========================================================
    # 2. INTERVALOS DE CONFIANZA
    # =========================================================
    ax2 = axes[0, 1]

    test_idx = np.arange(
        len(y_train_actual),
        len(y_train_actual) + len(y_test_series)
    )

    ax2.fill_between(
        test_idx,
        y_test_low.values,
        y_test_up.values,
        alpha=0.3,
        color='gray',
        label='IC 95%'
    )

    ax2.plot(test_idx, y_test_series.values, 'g-', alpha=0.7, label='Real')
    ax2.plot(test_idx, y_test_pred.values, 'g--', alpha=0.7, label='Predicho')

    ax2.set_title(
        'Pronóstico con Intervalos de Confianza',
        fontweight='bold'
    )

    ax2.set_xlabel('Índice')
    ax2.set_ylabel('Valor')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # =========================================================
    # 3. ANÁLISIS DE RESIDUALES
    # =========================================================
    ax3 = axes[0, 2]

    residuales_train = y_train_actual - y_train_pred_vals
    residuales_test = y_test_series.values - y_test_pred.values

    ax3.scatter(
        y_train_pred_vals,
        residuales_train,
        alpha=0.5,
        s=10,
        label='Entrenamiento'
    )

    ax3.scatter(
        y_test_pred.values,
        residuales_test,
        alpha=0.5,
        s=10,
        color='orange',
        label='Prueba'
    )

    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    ax3.set_title('Análisis de Residuales', fontweight='bold')
    ax3.set_xlabel('Valores Predichos')
    ax3.set_ylabel('Residuales')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # =========================================================
    # 4. DISTRIBUCIÓN DE RESIDUALES
    # =========================================================
    ax4 = axes[1, 0]

    ax4.hist(
        residuales_train,
        bins=30,
        alpha=0.7,
        density=True,
        label='Entrenamiento'
    )

    ax4.hist(
        residuales_test,
        bins=30,
        alpha=0.7,
        density=True,
        color='orange',
        label='Prueba'
    )

    ax4.set_title('Distribución de Residuales', fontweight='bold')
    ax4.set_xlabel('Residuales')
    ax4.set_ylabel('Densidad')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # =========================================================
    # 5. QQ PLOT
    # =========================================================
    ax5 = axes[1, 1]

    stats.probplot(residuales_train, dist="norm", plot=ax5)

    ax5.set_title('QQ-Plot (Entrenamiento)', fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # =========================================================
    # 6. AUTOCORRELACIÓN
    # =========================================================
    ax6 = axes[1, 2]

    plot_acf(residuales_train, lags=50, ax=ax6, alpha=0.05)

    ax6.set_title('Autocorrelación de Residuales', fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # =========================================================
    # 7. MÉTRICAS
    # =========================================================
    ax7 = axes[2, 0]

    metricas = ['MSE%', 'RMSE%', 'SMAPE%', 'R²']

    valores_in = [
        metrics_in['MSE_in_%'],
        metrics_in['RMSE_in_%'],
        metrics_in['SMAPE_in_%'],
        metrics_in['R2_in']
    ]

    valores_out = [
        metrics_out['MSE_out_%'],
        metrics_out['RMSE_out_%'],
        metrics_out['SMAPE_out_%'],
        metrics_out['R2_out']
    ]

    x = np.arange(len(metricas))
    ancho = 0.35

    ax7.bar(x - ancho/2, valores_in, ancho, label='In-sample', alpha=0.7)
    ax7.bar(x + ancho/2, valores_out, ancho, label='Out-of-sample', alpha=0.7)

    ax7.set_title('Comparación de Métricas', fontweight='bold')
    ax7.set_xlabel('Métrica')
    ax7.set_ylabel('Valor')
    ax7.set_xticks(x)
    ax7.set_xticklabels(metricas)

    ax7.legend(fontsize=8)
    ax7.grid(True, axis='y', alpha=0.3)

    # =========================================================
    # 8. INFORMACIÓN DEL MODELO
    # =========================================================
    ax8 = axes[2, 1]
    ax8.axis("off")

    info_text = f"""
MODELO SARIMA

order = {ORDER}
seasonal_order = {SEASONAL_ORDER}
trend = {TREND}

RMSE  : {metrics_out['RMSE_out']:.4f}
SMAPE : {metrics_out['SMAPE_out_%']:.2f} %
R²    : {metrics_out['R2_out']:.4f}

AIC : {modelo_final.aic:.2f}
BIC : {modelo_final.bic:.2f}
"""

    ax8.text(
        0.5, 0.5,
        info_text,
        ha="center",
        va="center",
        fontsize=10,
        family="monospace",
        transform=ax8.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    # =========================================================
    # 9. PRONÓSTICO A LARGO PLAZO
    # =========================================================
    ax9 = axes[2, 2]

    h_largo = 200

    y_largo_pred, y_largo_low, y_largo_up = forecast_sarima(
        modelo_final,
        y_train_series,
        h_largo
    )

    ultimos = 100
    inicio_grafico = max(len(y_train_series) - ultimos, 0)

    idx_train_plot = np.arange(inicio_grafico, len(y_train_series))
    idx_largo_plot = np.arange(len(y_train_series), len(y_train_series) + h_largo)

    ax9.plot(
        idx_train_plot,
        y_train_series.values[inicio_grafico:],
        'b-',
        alpha=0.7,
        label='Últimos datos'
    )

    ax9.plot(
        idx_largo_plot,
        y_largo_pred.values,
        'g-',
        alpha=0.7,
        label='Pronóstico'
    )

    ax9.fill_between(
        idx_largo_plot,
        y_largo_low.values,
        y_largo_up.values,
        alpha=0.3,
        color='gray',
        label='IC 95%'
    )

    ax9.set_title(
        f'Pronóstico a Largo Plazo ({h_largo} pasos)',
        fontweight='bold'
    )

    ax9.set_xlabel('Índice')
    ax9.set_ylabel('Valor')
    ax9.legend(loc='best', fontsize=8)
    ax9.grid(True, alpha=0.3)

    # =========================================================
    # TÍTULO GENERAL
    # =========================================================
    fig.suptitle(
        "ANÁLISIS SARIMA - NASDAQ Dataset\n"
        f"Modelo: SARIMA{ORDER}{SEASONAL_ORDER[:-1]}[{SEASONAL_ORDER[-1]}]",
        fontsize=14,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout()

    # Guardar figura
    plt.savefig(
        'sarima_NASDAQ_analysis_fixed.png',
        dpi=150,
        bbox_inches='tight'
    )

    plt.show()

    return fig






def graficar_diagnostico_modelo_SARIMA_conindice(
    y_train_actual,
    y_train_pred_vals,
    y_test_series,
    y_test_pred,
    y_test_low,
    y_test_up,
    metrics_in,
    metrics_out,
    modelo_final,
    ORDER,
    SEASONAL_ORDER,
    TREND,
    y_train_series,
    train_idx,
    test_idx
):
    """
    Genera un panel completo de diagnóstico para modelos SARIMA usando índices explícitos.
    
    Parámetros adicionales:
    - train_idx : índice (pandas Index o array) para el conjunto de entrenamiento
    - test_idx  : índice para el conjunto de prueba
    - full_idx  : índice de toda la serie
    """

    # =========================================================
    # CREAR FIGURA PRINCIPAL
    # =========================================================
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # =========================================================
    # 1. SERIE TEMPORAL COMPLETA
    # =========================================================
    ax1 = axes[0, 0]
    # Entrenamiento real y predicho
    ax1.plot(train_idx, y_train_actual, 'b-', alpha=0.7, label='Entrenamiento (Real)')
    ax1.plot(train_idx, y_train_pred_vals, 'b--', alpha=0.7, label='Entrenamiento (Predicho)')
    # Prueba real y predicha
    ax1.plot(test_idx, y_test_series.values, 'g-', alpha=0.7, label='Prueba (Real)')
    ax1.plot(test_idx, y_test_pred.values, 'g--', alpha=0.7, label='Prueba (Predicho)')
    # Línea de separación
    ax1.axvline(x=train_idx[-1], color='r', linestyle='--', alpha=0.5)
    ax1.set_title('Serie Temporal Completa', fontweight='bold')
    ax1.set_xlabel('Índice')
    ax1.set_ylabel('Valor')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # =========================================================
    # 2. INTERVALOS DE CONFIANZA
    # =========================================================
    ax2 = axes[0, 1]
    ax2.fill_between(test_idx, y_test_low.values, y_test_up.values,
                     alpha=0.3, color='gray', label='IC 95%')
    ax2.plot(test_idx, y_test_series.values, 'g-', alpha=0.7, label='Real')
    ax2.plot(test_idx, y_test_pred.values, 'g--', alpha=0.7, label='Predicho')
    ax2.set_title('Pronóstico con Intervalos de Confianza', fontweight='bold')
    ax2.set_xlabel('Índice')
    ax2.set_ylabel('Valor')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    # Hacer etiquetas del eje x en vertical
    plt.setp(ax2.get_xticklabels(), rotation=90)

    # =========================================================
    # 3. ANÁLISIS DE RESIDUALES
    # =========================================================
    ax3 = axes[0, 2]
    residuales_train = y_train_actual - y_train_pred_vals
    residuales_test = y_test_series.values - y_test_pred.values
    ax3.scatter(y_train_pred_vals, residuales_train, alpha=0.5, s=10, label='Entrenamiento')
    ax3.scatter(y_test_pred.values, residuales_test, alpha=0.5, s=10, color='orange', label='Prueba')
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax3.set_title('Análisis de Residuales', fontweight='bold')
    ax3.set_xlabel('Valores Predichos')
    ax3.set_ylabel('Residuales')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # =========================================================
    # 4. DISTRIBUCIÓN DE RESIDUALES
    # =========================================================
    ax4 = axes[1, 0]
    ax4.hist(residuales_train, bins=30, alpha=0.7, density=True, label='Entrenamiento')
    ax4.hist(residuales_test, bins=30, alpha=0.7, density=True, color='orange', label='Prueba')
    ax4.set_title('Distribución de Residuales', fontweight='bold')
    ax4.set_xlabel('Residuales')
    ax4.set_ylabel('Densidad')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # =========================================================
    # 5. QQ PLOT
    # =========================================================
    ax5 = axes[1, 1]
    stats.probplot(residuales_train, dist="norm", plot=ax5)
    ax5.set_title('QQ-Plot (Entrenamiento)', fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # =========================================================
    # 6. AUTOCORRELACIÓN
    # =========================================================
    ax6 = axes[1, 2]
    plot_acf(residuales_train, lags=50, ax=ax6, alpha=0.05)
    ax6.set_title('Autocorrelación de Residuales', fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # =========================================================
    # 7. MÉTRICAS
    # =========================================================
    ax7 = axes[2, 0]
    metricas = ['MSE%', 'RMSE%', 'SMAPE%', 'R²']
    valores_in = [metrics_in['MSE_in_%'], metrics_in['RMSE_in_%'], metrics_in['SMAPE_in_%'], metrics_in['R2_in']]
    valores_out = [metrics_out['MSE_out_%'], metrics_out['RMSE_out_%'], metrics_out['SMAPE_out_%'], metrics_out['R2_out']]
    x = np.arange(len(metricas))
    ancho = 0.35
    ax7.bar(x - ancho/2, valores_in, ancho, label='In-sample', alpha=0.7)
    ax7.bar(x + ancho/2, valores_out, ancho, label='Out-of-sample', alpha=0.7)
    ax7.set_title('Comparación de Métricas', fontweight='bold')
    ax7.set_xlabel('Métrica')
    ax7.set_ylabel('Valor')
    ax7.set_xticks(x)
    ax7.set_xticklabels(metricas)
    ax7.legend(fontsize=8)
    ax7.grid(True, axis='y', alpha=0.3)

    # =========================================================
    # 8. INFORMACIÓN DEL MODELO
    # =========================================================
    ax8 = axes[2, 1]
    ax8.axis("off")
    info_text = f"""
MODELO SARIMA

order = {ORDER}
seasonal_order = {SEASONAL_ORDER}
trend = {TREND}

RMSE  : {metrics_out['RMSE_out']:.4f}
SMAPE : {metrics_out['SMAPE_out_%']:.2f} %
R²    : {metrics_out['R2_out']:.4f}

AIC : {modelo_final.aic:.2f}
BIC : {modelo_final.bic:.2f}
"""
    ax8.text(0.5, 0.5, info_text, ha="center", va="center", fontsize=10,
             family="monospace", transform=ax8.transAxes,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    ax9 = axes[2, 2]
    ax9.cla()  # Limpia el gráfico dentro del subplot
    # =========================================================
    # TÍTULO GENERAL
    # =========================================================
    fig.suptitle(
        "ANÁLISIS SARIMA - NASDAQ Dataset\n"
        f"Modelo: SARIMA{ORDER}{SEASONAL_ORDER[:-1]}[{SEASONAL_ORDER[-1]}]",
        fontsize=14, fontweight='bold', y=0.98
    )

    plt.tight_layout()
   # plt.savefig('sarima_NASDAQ_analysis_fixed.png', dpi=150, bbox_inches='tight')
    plt.show()

    return fig



def graficar_dashboard_modelo_ARnet(
    train_series_true,
    train_series_pred,
    test_series_true,
    test_series_pred,
    idx_tr,
    idx_te,
    idx_full,
    series_true_full,
    series_pred_full,
    metrics_out,
    n_tr,
    n_params_real,
    config_text="",
    titulo="ANÁLISIS DEL MODELO"
):
    """
    Genera un dashboard de diagnóstico para modelos de series de tiempo
    y retorna la figura generada.
    """

    # ----------------------------
    # Residuales en escala original
    # ----------------------------
    res_train_orig = train_series_true - train_series_pred
    res_test_orig  = test_series_true  - test_series_pred

    # Métricas
    rmse_test  = metrics_out["RMSE_out"]
    smape_test = metrics_out["SMAPE_out_%"]
    r2_test    = metrics_out["R2_out_Pearson2"]

    # ----------------------------
    # Figura tipo dashboard
    # ----------------------------
    fig = plt.figure(figsize=(20, 14))
    plt.suptitle(titulo, fontsize=18, fontweight="bold")

    # (1) Serie completa
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(idx_full, series_true_full, label="Real", linewidth=1)
    ax1.plot(idx_full, series_pred_full, label="Predicho", linestyle="--", linewidth=1)
    ax1.axvline(n_tr - 1, linestyle="--", color="red", linewidth=1)
    ax1.set_title("Serie Temporal Completa")
    ax1.set_xlabel("Índice")
    ax1.set_ylabel("Valor")
    ax1.legend()
    ax1.grid(True)

    # (2) Entrenamiento
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(idx_tr, train_series_true, label="Real")
    ax2.plot(idx_tr, train_series_pred, linestyle="--", label="Predicho")
    ax2.set_title("Entrenamiento")
    ax2.legend()
    ax2.grid(True)

    # (3) Prueba
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(idx_te, test_series_true, label="Real")
    ax3.plot(idx_te, test_series_pred, linestyle="--", label="Predicho")
    ax3.set_title("Prueba")
    ax3.legend()
    ax3.grid(True)

    # (4) Residuales vs predicción
    ax4 = plt.subplot(3, 3, 4)
    ax4.scatter(train_series_pred, res_train_orig, alpha=0.5, label="Entrenamiento")
    ax4.scatter(test_series_pred,  res_test_orig,  alpha=0.5, label="Prueba")
    ax4.axhline(0, linestyle="--", color="red")
    ax4.set_title("Residuales vs Predicción")
    ax4.legend()
    ax4.grid(True)

    # (5) Histograma residuales
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(res_train_orig, bins=30, alpha=0.6, label="Train", density=True)
    ax5.hist(res_test_orig,  bins=30, alpha=0.6, label="Test", density=True)
    ax5.set_title("Distribución de Residuales")
    ax5.legend()
    ax5.grid(True)

    # (6) QQ-Plot
    ax6 = plt.subplot(3, 3, 6)
    sm.qqplot(res_train_orig, line="45", ax=ax6)
    ax6.set_title("QQ-Plot")

    # (7) ACF
    ax7 = plt.subplot(3, 3, 7)
    plot_acf(res_train_orig, ax=ax7, lags=40)
    ax7.set_title("ACF Residuales")

    # (8) Métricas
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis("off")

    text_metrics = f"""
    CONFIGURACIÓN
    {config_text}

    n_params: {n_params_real}

                
    MÉTRICAS PRUEBA
    RMSE: {rmse_test:.4f}
    SMAPE: {smape_test:.2f}%
    R²: {r2_test:.4f}
    """
    ax8.text(0.02, 0.98, text_metrics, fontsize=11, va="top")

    plt.tight_layout()
    
    # Mostrar gráfica
    plt.show()

    return fig


def asegurar_carpeta(ruta_carpeta):
    """
    Verifica si una carpeta existe y la crea si no existe.
    
    Parámetros:
    - ruta_carpeta: str, ruta de la carpeta a asegurar
    
    Retorna:
    - True si la carpeta ahora existe (ya existía o se creó)
    """
    if not os.path.exists(ruta_carpeta):
        os.makedirs(ruta_carpeta)
        print(f"La carpeta '{ruta_carpeta}' no existía y se ha creado.")
    else:
        print(f"La carpeta '{ruta_carpeta}' ya existe.")
    return True