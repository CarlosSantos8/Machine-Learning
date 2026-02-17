# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 19:55:54 2026

@author: LENOVO
"""
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns


def resumen_faltantes(dataframe):
    """
    Muestra el resumen de los valores faltantes en el DataFrame, incluyendo la cantidad de faltantes y el porcentaje por columna.
    
    Args:
    dataframe (pd.DataFrame): El dataframe para el que se calcularán los faltantes.
    
    Returns:
    pd.DataFrame: Un DataFrame con el resumen de faltantes (cantidad y porcentaje).
    """
    # Calcular los faltantes por columna
    faltantes = dataframe.isna().sum().sort_values(ascending=False)
    porcentaje = (dataframe.isna().mean() * 100).round(3).sort_values(ascending=False)
    
    # Crear un resumen en un nuevo DataFrame
    resumen_faltantes = pd.DataFrame({"faltantes": faltantes, "porcentaje_%": porcentaje})
    
    # Mostrar el resumen de faltantes
    print("\nResumen de faltantes por columna:\n")
    
    return resumen_faltantes  # Retorna el DataFrame con el resumen de faltantes


def plot_missing_values(dataframe, top_n=20):
    """
    Crea un gráfico de barras mostrando los primeros 'top_n' valores faltantes en el DataFrame.
    
    Args:
    dataframe (pd.DataFrame): El dataframe para el cual se graficarán los valores faltantes.
    top_n (int): El número de columnas a mostrar en el gráfico (por defecto, 20).
    """
    # --- Crear la figura ---
    plt.figure(figsize=(8, 6))  # Tamaño de la figura

    # Calcular el total de valores faltantes por columna
    total = dataframe.isnull().sum().sort_values(ascending=False)
    
    # Seleccionar los primeros 'top_n' valores faltantes
    total_select = total.head(top_n)
    
    # Crear el gráfico de barras
    total_select.plot(kind="bar", fontsize=10, color="skyblue")

    # Etiquetas y título
    plt.xlabel("Columns", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title(f"Total Missing Values (Top {top_n})", fontsize=16)

    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()


# Función para crear el boxplot
def crear_boxplot(data, columna):
    """
    Esta función crea un boxplot personalizado para una columna de un DataFrame.
    
    Parámetros:
    - data: DataFrame que contiene la columna.
    - columna: Nombre de la columna a graficar.
    """
    # Configuración de estilo de seaborn
    sns.set(style="whitegrid")  # Fondo blanco con una cuadrícula

    # Crear la figura con un tamaño adecuado
    plt.figure(figsize=(8, 6))

    # Crear el boxplot con personalización
    sns.boxplot(x=data[columna], 
                color="skyblue",   # Color de la caja
                width=0.5,         # Ancho de la caja
                fliersize=8,       # Tamaño de los puntos atípicos
                linewidth=1.5)     # Grosor de las líneas del boxplot

    # Personalizar títulos y etiquetas
    plt.title(f'Distribución de {columna}', fontsize=16, fontweight='bold')  # Título del gráfico
    plt.xlabel(columna, fontsize=14)  # Etiqueta en el eje x

    # Ajustar el diseño
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()
    
def Grafico_pastel_objetivo(df, target_col, colors=None):
    """
    Genera un gráfico de pastel con la proporción y el número absoluto
    de la variable objetivo.

    Parámetros:
    df : pandas.DataFrame
        DataFrame que contiene los datos.
    target_col : str
        Nombre de la columna objetivo.
    colors : list, opcional
        Lista de colores para el gráfico.
    """

    # Conteos absolutos
    conteos = df[target_col].value_counts()
    total = conteos.sum()

    # Si no se especifican colores
    if colors is None:
        colors = ["#cfe2f3", "#d9ead3"]

    # Función para mostrar porcentaje + valor absoluto
    def autopct_format(pct):
        valor = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({valor})"

    # Crear gráfico
    plt.figure()
    plt.pie(
        conteos,
        labels=conteos.index,
        autopct=autopct_format,
        colors=colors
    )

    plt.title(f"Proporción de la variable objetivo: {target_col}")
    plt.show()


def diagnostico_balance_clases_binaria(df, target, umbral_estratificar=1.5, verbose=True):
    """
    Diagnostica el balance de clases antes del split.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame completo.
    target : str
        Nombre de la variable objetivo.
    umbral_estratificar : float
        IR a partir del cual se recomienda estratificar.
    verbose : bool
        Si True imprime el diagnóstico.

    Retorna
    -------
    dict con estadísticas del balance.
    """

    if target not in df.columns:
        raise ValueError(f"La columna '{target}' no existe en el DataFrame.")

    y = df[target].dropna()

    clases, conteos = np.unique(y, return_counts=True)
    n_total = conteos.sum()
    proporciones = conteos / n_total

    n_max = conteos.max()
    n_min = conteos.min()
    IR = n_max / n_min if n_min > 0 else np.inf

    # Clasificación del nivel de desbalance
    if IR < 1.5:
        etiqueta = "Balance razonable (IR < 1.5)"
    elif IR < 3:
        etiqueta = "Desbalance moderado (1.5 ≤ IR < 3)"
    else:
        etiqueta = "Desbalance severo (IR ≥ 3)"

    recomendar_estratificar = IR >= umbral_estratificar

    if verbose:
        print(f"\n[Diagnóstico de balance] Variable objetivo: '{target}'")
        print(f"Total muestras: {n_total}\n")

        for c, n_c, p_c in zip(clases, conteos, proporciones):
            print(f"Clase {c}: {n_c} ({p_c:.1%})")

        print(f"\nIR (Imbalance Ratio) = {IR:.3f} → {etiqueta}")

        if recomendar_estratificar:
            print("\nSe recomienda estratificar: usar stratify=y en train_test_split.")
            

def detect_outliers_iqr_indices(df, col):
    """
    Detecta outliers en una columna usando el método IQR y devuelve los índices de los outliers.
    
    Parámetros:
    df : pd.DataFrame
        DataFrame que contiene la columna a analizar.
    col : str
        Nombre de la columna numérica.
    
    Retorna:
    dict
        Diccionario con:
        - outlier_indices: lista de índices de los outliers
        - n_outliers: número de outliers
        - prop_outliers: proporción de outliers
        - q1, q3, iqr, lim_inf, lim_sup
    """
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    
    if s.empty:
        return {
            'outlier_indices': [],
            'n_outliers': 0,
            'prop_outliers': 0.0,
            'q1': np.nan,
            'q3': np.nan,
            'iqr': np.nan,
            'lim_inf': np.nan,
            'lim_sup': np.nan
        }
    
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - 1.5*iqr, q3 + 1.5*iqr
    
    outlier_mask = (s < lim_inf) | (s > lim_sup)
    outlier_indices = s[outlier_mask].index.tolist()
    
    n_out = len(outlier_indices)
    prop_out = n_out / len(s)
    
    return {
        'outlier_indices': outlier_indices,
        'n_outliers': n_out,
        'prop_outliers': round(prop_out, 3),
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'lim_inf': lim_inf,
        'lim_sup': lim_sup
    }