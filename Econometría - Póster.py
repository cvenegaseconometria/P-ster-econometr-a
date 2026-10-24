
# ============================================================
# IDENTIFICACIÓN
# Carga, limpieza, gráficos de la serie original y primera
# diferencia, FAC y FACP
# ============================================================

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from IPython.display import display
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch
)
from scipy.stats import jarque_bera

# ------------------------------------------------------------
# 1. CARGAR Y LIMPIAR LOS DATOS
# ------------------------------------------------------------

archivo = "anex-IPC-sep2026.xlsx"

datos = pd.read_excel(
    archivo,
    sheet_name="11",
    header=7
)

datos = datos.iloc[:, :6].copy()

datos.columns = [
    "anio",
    "mes",
    "indice",
    "variacion_mensual",
    "variacion_anio_corrido",
    "variacion_anual"
]

datos["anio"] = pd.to_numeric(datos["anio"], errors="coerce")
datos["indice"] = pd.to_numeric(datos["indice"], errors="coerce")

datos = datos.dropna(subset=["anio", "indice"]).copy()

meses = {
    "enero": 1, "febrero": 2, "marzo": 3,
    "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9,
    "octubre": 10, "noviembre": 11, "diciembre": 12
}

datos["numero_mes"] = (
    datos["mes"].astype(str).str.strip().str.lower().map(meses)
)

datos = datos.dropna(subset=["numero_mes"]).copy()

datos["fecha"] = pd.to_datetime({
    "year": datos["anio"].astype(int),
    "month": datos["numero_mes"].astype(int),
    "day": 1
})

datos = (
    datos.sort_values("fecha")
    .drop_duplicates(subset="fecha", keep="last")
    .set_index("fecha")
)

# Frecuencia mensual explícita
y = datos["indice"].astype(float).asfreq("MS")
y = y.dropna()

# Primera diferencia
dy = y.diff().dropna()

print("IDENTIFICACIÓN DE LA SERIE")
print(f"Primera observación: {y.index.min():%Y-%m}")
print(f"Última observación: {y.index.max():%Y-%m}")
print(f"Número de observaciones: {len(y)}")

# ------------------------------------------------------------
# 2. GRÁFICA DE LA SERIE ORIGINAL
# ------------------------------------------------------------

plt.figure(figsize=(13, 5))
plt.plot(y.index, y, label="IPC de energéticos")
plt.title("Serie original del IPC de energéticos")
plt.xlabel("Fecha")
plt.ylabel("Índice")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 3. FAC Y FACP DE LA SERIE ORIGINAL
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

plot_acf(y, lags=36, zero=False, ax=axes[0])
axes[0].set_title("FAC de la serie original")

plot_pacf(y, lags=36, zero=False, method="ywm", ax=axes[1])
axes[1].set_title("FACP de la serie original")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 4. GRÁFICA DE LA PRIMERA DIFERENCIA
# ------------------------------------------------------------

plt.figure(figsize=(13, 5))
plt.plot(dy.index, dy, label="Primera diferencia")
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.title("Primera diferencia del IPC de energéticos")
plt.xlabel("Fecha")
plt.ylabel("Variación mensual del índice")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. FAC Y FACP DE LA PRIMERA DIFERENCIA
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

plot_acf(dy, lags=36, zero=False, ax=axes[0])
axes[0].set_title("FAC de la primera diferencia")

plot_pacf(dy, lags=36, zero=False, method="ywm", ax=axes[1])
axes[1].set_title("FACP de la primera diferencia")

plt.tight_layout()
plt.show()


# ============================================================
# ESTIMACIÓN
# Comparación de los 15 modelos ARIMA(p,1,q)
# ============================================================

# ------------------------------------------------------------
# 6. ESTIMAR LOS 15 MODELOS Y COMPARAR AIC Y BIC
# ------------------------------------------------------------

resultados_modelos = []
ajustes = {}

for p in range(3):
    for q in range(5):

        orden = (p, 1, q)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                ajuste = ARIMA(
                    y,
                    order=orden,
                    trend="n"
                ).fit()

            nombre = f"ARIMA({p},1,{q})"
            ajustes[orden] = ajuste

            resultados_modelos.append({
                "Modelo": nombre,
                "p": p,
                "d": 1,
                "q": q,
                "AIC": ajuste.aic,
                "BIC": ajuste.bic,
                "Convergencia": ajuste.mle_retvals.get(
                    "converged", True
                )
            })

        except Exception as error:
            print(f"No se pudo estimar ARIMA{orden}: {error}")

tabla_modelos = pd.DataFrame(resultados_modelos)
tabla_modelos = tabla_modelos.sort_values("AIC").reset_index(drop=True)

print("\nCOMPARACIÓN DE MODELOS: ORDENADOS POR AIC")
display(tabla_modelos.round(4))

print("\nMEJOR MODELO SEGÚN AIC:")
print(tabla_modelos.loc[0, "Modelo"])

print("\nMEJOR MODELO SEGÚN BIC:")
print(tabla_modelos.loc[tabla_modelos["BIC"].idxmin(), "Modelo"])


# ------------------------------------------------------------
# 7. EXTRAER LOS MODELOS ARIMA(1,1,1) Y ARIMA(2,1,1)
# ------------------------------------------------------------

ajuste_111 = ajustes[(1, 1, 1)]
ajuste_211 = ajustes[(2, 1, 1)]

print("\nRESUMEN ARIMA(1,1,1)")
print(ajuste_111.summary())

print("\nRESUMEN ARIMA(2,1,1)")
print(ajuste_211.summary())


# ============================================================
# VALIDACIÓN
# Diagnóstico de residuos, significancia, normalidad,
# heteroscedasticidad y estacionariedad
# ============================================================

# ------------------------------------------------------------
# 8. GRÁFICA, FAC Y FACP DE LOS RESIDUOS DEL ARIMA(2,1,1)
# ------------------------------------------------------------

# Excluir los dos primeros residuos en los diagnósticos
# para mantener el criterio usado anteriormente.
residuos_211 = ajuste_211.resid.iloc[2:].dropna()

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

axes[0, 0].plot(
    residuos_211.index,
    residuos_211
)
axes[0, 0].axhline(0, color="black", linestyle="--")
axes[0, 0].set_title("Residuos del ARIMA(2,1,1)")
axes[0, 0].set_xlabel("Fecha")
axes[0, 0].set_ylabel("Residuo")

plot_acf(
    residuos_211,
    lags=36,
    zero=False,
    ax=axes[0, 1]
)
axes[0, 1].set_title("FAC de los residuos ARIMA(2,1,1)")

plot_pacf(
    residuos_211,
    lags=36,
    zero=False,
    method="ywm",
    ax=axes[1, 0]
)
axes[1, 0].set_title("FACP de los residuos ARIMA(2,1,1)")

sm.qqplot(
    residuos_211,
    line="s",
    ax=axes[1, 1]
)
axes[1, 1].set_title("Gráfico Q-Q de los residuos ARIMA(2,1,1)")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 9. PRUEBA LJUNG-BOX PARA AMBOS MODELOS
# ------------------------------------------------------------

# Para ARIMA(1,1,1): p + q = 2
# Para ARIMA(2,1,1): p + q = 3

residuos_111 = ajuste_111.resid.iloc[2:].dropna()

for nombre, residuos, grados_libertad in [
    ("ARIMA(1,1,1)", residuos_111, 2),
    ("ARIMA(2,1,1)", residuos_211, 3)
]:
    print(f"\nLJUNG-BOX: {nombre}")

    prueba_lb = acorr_ljungbox(
        residuos,
        lags=[6, 12, 18, 24],
        model_df=grados_libertad,
        return_df=True
    )

    display(prueba_lb.round(4))

    if (prueba_lb["lb_pvalue"] < 0.05).any():
        print("Hay evidencia de autocorrelación en al menos uno de los rezagos.")
    else:
        print("No se rechaza la ausencia de autocorrelación en los rezagos evaluados.")


# ------------------------------------------------------------
# 10. ESTIMACIÓN DE PARÁMETROS DEL ARIMA(2,1,1)
# ------------------------------------------------------------

tabla_parametros = pd.DataFrame({
    "Parámetro": ajuste_211.params.index,
    "Estimación": ajuste_211.params.values,
    "Error estándar": ajuste_211.bse.values,
    "Estadístico z": ajuste_211.tvalues,
    "Valor p": ajuste_211.pvalues
})

tabla_parametros["Significancia (5 %)"] = (
    tabla_parametros["Valor p"]
    .apply(lambda p: "Significativo" if p < 0.05 else "No significativo")
)

print("\nPARÁMETROS ARIMA(2,1,1)")
display(tabla_parametros.round(4))


# ------------------------------------------------------------
# 11. PRUEBA DE NORMALIDAD JARQUE-BERA
# ------------------------------------------------------------

jb = jarque_bera(residuos_211)

print("\nJARQUE-BERA: RESIDUOS ARIMA(2,1,1)")
print(f"Estadístico JB: {jb.statistic:.4f}")
print(f"Valor p: {jb.pvalue:.6g}")

if jb.pvalue < 0.05:
    print("Se rechaza la hipótesis nula de normalidad.")
else:
    print("No se rechaza la hipótesis nula de normalidad.")


# ------------------------------------------------------------
# 12. PRUEBA DE HETEROSCEDASTICIDAD ARCH
# ------------------------------------------------------------

arch = het_arch(residuos_211, nlags=12)

print("\nPRUEBA ARCH: RESIDUOS ARIMA(2,1,1)")
print(f"LM: {arch[0]:.4f}")
print(f"Valor p LM: {arch[1]:.6f}")
print(f"Estadístico F: {arch[2]:.4f}")
print(f"Valor p F: {arch[3]:.6f}")

if arch[1] < 0.05:
    print("Hay evidencia de heteroscedasticidad condicional tipo ARCH.")
else:
    print("No se encuentra evidencia suficiente de efectos ARCH.")


# ------------------------------------------------------------
# 13. DICKEY-FULLER AUMENTADA (ADF)
# Serie original, primera diferencia y residuos de ambos modelos
# ------------------------------------------------------------

def prueba_adf(serie, nombre):
    serie = pd.Series(serie).dropna()

    resultado = adfuller(
        serie,
        regression="c",
        autolag="AIC"
    )

    print(f"\nADF: {nombre}")
    print(f"Estadístico: {resultado[0]:.4f}")
    print(f"Valor p: {resultado[1]:.6f}")
    print(f"Rezagos utilizados: {resultado[2]}")

    if resultado[1] < 0.05:
        print("Se rechaza la hipótesis nula de raíz unitaria.")
    else:
        print("No se rechaza la hipótesis nula de raíz unitaria.")

prueba_adf(y, "Serie original")
prueba_adf(dy, "Primera diferencia")
prueba_adf(residuos_111, "Residuos ARIMA(1,1,1)")
prueba_adf(residuos_211, "Residuos ARIMA(2,1,1)")


# ============================================================
# PRONÓSTICO
# Evaluación histórica desde 2023 y pronóstico futuro desde 2026
# ============================================================

# ------------------------------------------------------------
# 14. PRONÓSTICO HISTÓRICO: ENERO 2023 - SEPTIEMBRE 2026
# Entrenamiento hasta diciembre de 2022
# ------------------------------------------------------------

entrenamiento = y.loc[: "2022-12-01"]
observados = y.loc["2023-01-01":"2026-09-01"]

modelo_historico = ARIMA(
    entrenamiento,
    order=(2, 1, 1),
    trend="n"
).fit()

pred_historico = modelo_historico.get_forecast(
    steps=len(observados)
)

tabla_historica = pred_historico.summary_frame(alpha=0.05)
tabla_historica.index = observados.index
tabla_historica.index.name = "Fecha"

comparacion = pd.DataFrame({
    "Observado": observados,
    "Pronóstico": tabla_historica["mean"],
    "IC inferior 95 %": tabla_historica["mean_ci_lower"],
    "IC superior 95 %": tabla_historica["mean_ci_upper"]
})

errores = comparacion["Observado"] - comparacion["Pronóstico"]

mae = np.mean(np.abs(errores))
rmse = np.sqrt(np.mean(errores**2))
mape = np.mean(
    np.abs(errores / comparacion["Observado"])
) * 100

print("\nPRONÓSTICO HISTÓRICO: ENERO 2023 - SEPTIEMBRE 2026")
print(f"Número de observaciones: {len(observados)}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f} %")

display(comparacion.round(4))

plt.figure(figsize=(14, 6))
plt.plot(
    entrenamiento.index,
    entrenamiento,
    label="Entrenamiento hasta diciembre de 2022"
)
plt.plot(
    observados.index,
    observados,
    label="Observado",
    color="black"
)
plt.plot(
    comparacion.index,
    comparacion["Pronóstico"],
    label="Pronóstico histórico",
    color="darkorange"
)
plt.fill_between(
    comparacion.index,
    comparacion["IC inferior 95 %"],
    comparacion["IC superior 95 %"],
    alpha=0.2,
    color="darkorange",
    label="Intervalo de predicción 95 %"
)
plt.axvline(
    pd.Timestamp("2023-01-01"),
    linestyle="--",
    color="gray"
)
plt.title("Pronóstico histórico del IPC: ARIMA(2,1,1)")
plt.xlabel("Fecha")
plt.ylabel("Índice IPC")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 15. PRONÓSTICO FUTURO: OCTUBRE 2026 - SEPTIEMBRE 2027
# Reestimación con toda la serie disponible
# ------------------------------------------------------------

modelo_final = ARIMA(
    y,
    order=(2, 1, 1),
    trend="n"
).fit()

pred_futuro = modelo_final.get_forecast(steps=12)
tabla_futura = pred_futuro.summary_frame(alpha=0.05)

fechas_futuras = pd.date_range(
    start=y.index[-1] + pd.offsets.MonthBegin(1),
    periods=12,
    freq="MS"
)

tabla_futura.index = fechas_futuras
tabla_futura.index.name = "Fecha"

pronostico_futuro = tabla_futura.rename(columns={
    "mean": "Pronóstico",
    "mean_se": "Error estándar",
    "mean_ci_lower": "IC inferior 95 %",
    "mean_ci_upper": "IC superior 95 %"
})

print("\nPRONÓSTICO FUTURO: 12 MESES")
display(pronostico_futuro.round(4))

plt.figure(figsize=(14, 6))
plt.plot(
    y.index,
    y,
    label="IPC observado",
    color="steelblue"
)
plt.plot(
    pronostico_futuro.index,
    pronostico_futuro["Pronóstico"],
    label="Pronóstico futuro",
    color="darkorange"
)
plt.fill_between(
    pronostico_futuro.index,
    pronostico_futuro["IC inferior 95 %"],
    pronostico_futuro["IC superior 95 %"],
    alpha=0.2,
    color="darkorange",
    label="Intervalo de predicción 95 %"
)
plt.axvline(
    y.index[-1],
    linestyle="--",
    color="gray",
    label="Último dato observado"
)
plt.title("Pronóstico futuro del IPC de energéticos: ARIMA(2,1,1)")
plt.xlabel("Fecha")
plt.ylabel("Índice IPC")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 14. PRONÓSTICO HISTÓRICO: ENERO 2025 - ÚLTIMO DATO OBSERVADO
# Entrenamiento hasta diciembre de 2024
# Intervalos mediante bootstrap de residuos
# ------------------------------------------------------------

def bootstrap_arima211(modelo, historia, pasos, n_sim=5000, semilla=123):
    """
    Simula trayectorias de un ARIMA(2,1,1) mediante bootstrap
    de residuos empíricos.
    """
    rng = np.random.default_rng(semilla)

    parametros = modelo.params
    phi1 = parametros["ar.L1"]
    phi2 = parametros["ar.L2"]
    theta1 = parametros["ma.L1"]

    diferencias = np.diff(np.asarray(historia, dtype=float))
    residuos = np.asarray(modelo.resid, dtype=float)
    residuos = residuos[2:]
    residuos = residuos[np.isfinite(residuos)]
    residuos = residuos - residuos.mean()

    if len(diferencias) < 2 or len(residuos) == 0:
        raise ValueError("No hay suficientes datos para el bootstrap.")

    trayectorias = np.zeros((n_sim, pasos))

    for b in range(n_sim):
        nivel = float(historia.iloc[-1])
        d1 = diferencias[-1]
        d2 = diferencias[-2]
        e1 = float(residuos[-1])

        innovaciones = rng.choice(residuos, size=pasos, replace=True)

        for h in range(pasos):
            e = innovaciones[h]

            nueva_diferencia = (
                phi1 * d1
                + phi2 * d2
                + e
                + theta1 * e1
            )

            nivel += nueva_diferencia
            trayectorias[b, h] = nivel

            d2 = d1
            d1 = nueva_diferencia
            e1 = e

    return trayectorias


entrenamiento = y.loc[:"2024-12-01"]
observados = y.loc["2025-01-01":]

if len(entrenamiento) == 0 or len(observados) == 0:
    raise ValueError(
        "La serie no contiene datos suficientes para evaluar "
        "el pronóstico histórico desde enero de 2025."
    )

modelo_historico = ARIMA(
    entrenamiento,
    order=(2, 1, 1),
    trend="n"
).fit()

pred_historico = modelo_historico.get_forecast(
    steps=len(observados)
)

trayectorias_historicas = bootstrap_arima211(
    modelo_historico,
    entrenamiento,
    pasos=len(observados),
    n_sim=5000,
    semilla=123
)

limite_inferior = np.percentile(
    trayectorias_historicas, 2.5, axis=0
)
limite_superior = np.percentile(
    trayectorias_historicas, 97.5, axis=0
)

comparacion = pd.DataFrame({
    "Observado": observados.to_numpy(),
    "Pronóstico": pred_historico.predicted_mean.to_numpy(),
    "IC inferior 95 %": limite_inferior,
    "IC superior 95 %": limite_superior
}, index=observados.index)

comparacion.index.name = "Fecha"

errores = comparacion["Observado"] - comparacion["Pronóstico"]

mae = np.mean(np.abs(errores))
rmse = np.sqrt(np.mean(errores**2))

mape = np.mean(
    np.abs(errores / comparacion["Observado"])
) * 100

print("\nPRONÓSTICO HISTÓRICO: ENERO 2025 - ÚLTIMO DATO")
print(f"Número de observaciones: {len(observados)}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f} %")
print("Intervalos de predicción: bootstrap de residuos, 95 %")

display(comparacion.round(4))

plt.figure(figsize=(14, 6))

plt.plot(
    entrenamiento.index,
    entrenamiento,
    label="Entrenamiento hasta diciembre de 2024"
)

plt.plot(
    observados.index,
    observados,
    label="Observado",
    color="black"
)

plt.plot(
    comparacion.index,
    comparacion["Pronóstico"],
    label="Pronóstico histórico",
    color="darkorange"
)

plt.fill_between(
    comparacion.index,
    comparacion["IC inferior 95 %"],
    comparacion["IC superior 95 %"],
    alpha=0.2,
    color="darkorange",
    label="Intervalo bootstrap 95 %"
)

plt.axvline(
    pd.Timestamp("2025-01-01"),
    linestyle="--",
    color="gray"
)

plt.title("Pronóstico histórico del IPC: ARIMA(2,1,1) con bootstrap")
plt.xlabel("Fecha")
plt.ylabel("Índice IPC")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 15. PRONÓSTICO FUTURO: 12 MESES
# Reestimación con toda la serie disponible
# Intervalos mediante bootstrap de residuos
# ------------------------------------------------------------

modelo_final = ARIMA(
    y,
    order=(2, 1, 1),
    trend="n"
).fit()

pred_futuro = modelo_final.get_forecast(steps=12)

fechas_futuras = pd.date_range(
    start=y.index[-1] + pd.offsets.MonthBegin(1),
    periods=12,
    freq="MS"
)

trayectorias_futuras = bootstrap_arima211(
    modelo_final,
    y,
    pasos=12,
    n_sim=5000,
    semilla=456
)

limite_inferior_futuro = np.percentile(
    trayectorias_futuras, 2.5, axis=0
)
limite_superior_futuro = np.percentile(
    trayectorias_futuras, 97.5, axis=0
)

pronostico_futuro = pd.DataFrame({
    "Pronóstico": pred_futuro.predicted_mean.to_numpy(),
    "IC inferior 95 %": limite_inferior_futuro,
    "IC superior 95 %": limite_superior_futuro
}, index=fechas_futuras)

pronostico_futuro.index.name = "Fecha"

print("\nPRONÓSTICO FUTURO: 12 MESES")
print("Intervalos de predicción: bootstrap de residuos, 95 %")

display(pronostico_futuro.round(4))

plt.figure(figsize=(14, 6))

plt.plot(
    y.index,
    y,
    label="IPC observado",
    color="steelblue"
)

plt.plot(
    pronostico_futuro.index,
    pronostico_futuro["Pronóstico"],
    label="Pronóstico futuro",
    color="darkorange"
)

plt.fill_between(
    pronostico_futuro.index,
    pronostico_futuro["IC inferior 95 %"],
    pronostico_futuro["IC superior 95 %"],
    alpha=0.2,
    color="darkorange",
    label="Intervalo bootstrap 95 %"
)

plt.axvline(
    y.index[-1],
    linestyle="--",
    color="gray",
    label="Último dato observado"
)

plt.title("Pronóstico futuro del IPC de energéticos: ARIMA(2,1,1) con bootstrap")
plt.xlabel("Fecha")
plt.ylabel("Índice IPC")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
# MODELO ARIMA (1,1,1) CON DERIVA 
# %%
# ================================================================
# ECONOMETRÍA II
# ARIMA(1,1,1) CON DERIVA + BOOTSTRAP DE RESIDUOS
# ================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera


# ================================================================
# 0. CONFIGURACIÓN
# ================================================================

B = 10000       # Número de trayectorias bootstrap
H = 12          # Horizonte de pronóstico
SEMILLA = 12345
ALPHA = 0.05

rng = np.random.default_rng(SEMILLA)

if not isinstance(y, pd.Series):
    raise TypeError("y debe ser una pandas.Series.")

y = y.copy().dropna().astype(float)
y.index = pd.DatetimeIndex(y.index).to_period("M").to_timestamp()
y = y.sort_index()
y.name = "indice"

if y.index.has_duplicates:
    raise ValueError("La serie contiene fechas mensuales duplicadas.")

fechas_esperadas = pd.date_range(
    y.index.min(), y.index.max(), freq="MS"
)

if not y.index.equals(fechas_esperadas):
    raise ValueError(
        "Hay meses faltantes o fechas irregulares. "
        "Corrige la serie antes de estimar."
    )

print("=" * 70)
print("SERIE ANALIZADA")
print("=" * 70)
print("Inicio:", y.index.min().date())
print("Final: ", y.index.max().date())
print("Observaciones:", len(y))


# ================================================================
# 1. ESTIMACIÓN PRINCIPAL
# ================================================================

modelo = ARIMA(
    y,
    order=(1, 1, 1),
    trend="t"
).fit()

print("\nRESUMEN DEL MODELO")
print(modelo.summary())


# ================================================================
# 2. PARÁMETROS Y CRITERIOS DE INFORMACIÓN
# ================================================================

ic = modelo.conf_int(alpha=ALPHA)

tabla_parametros = pd.DataFrame({
    "Coeficiente": modelo.params,
    "Error estándar": modelo.bse,
    "Estadístico z": modelo.tvalues,
    "Valor p": modelo.pvalues,
    "IC 95 % inferior": ic.iloc[:, 0],
    "IC 95 % superior": ic.iloc[:, 1]
})

print("\nPARÁMETROS")
display(tabla_parametros.round(5))

tabla_criterios = pd.DataFrame({
    "Indicador": [
        "Log-verosimilitud", "AIC", "BIC", "HQIC", "Observaciones"
    ],
    "Valor": [
        modelo.llf, modelo.aic, modelo.bic,
        modelo.hqic, modelo.nobs
    ]
})

print("\nCRITERIOS DE INFORMACIÓN")
display(tabla_criterios.round(4))


# ================================================================
# 3. RESIDUOS
# ================================================================

residuos = pd.Series(
    np.asarray(modelo.resid),
    index=y.index,
    name="residuo"
).dropna()

# Primer residuo separado para el análisis de sensibilidad.
residuos_sin_primero = residuos.iloc[1:]

print("\nRESUMEN DE RESIDUOS")
display(residuos.describe().to_frame().round(5))

print("\nPRIMEROS 10 RESIDUOS")
display(residuos.head(10).to_frame().round(5))

print("\nPrimer residuo:", residuos.iloc[0])
print(
    "Mediana absoluta de los residuos restantes:",
    residuos_sin_primero.abs().median()
)


# ================================================================
# 4. GRÁFICAS DE RESIDUOS, FAC Y PACF
# ================================================================

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(residuos.index, residuos, color="black", linewidth=0.9)
ax.axhline(0, color="gray", linewidth=1)
ax.set_title("Residuos del ARIMA(1,1,1) con deriva")
ax.set_xlabel("Fecha")
ax.set_ylabel("Residuo")
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(11, 8))

plot_acf(
    residuos_sin_primero, lags=24, zero=False, ax=axes[0]
)
axes[0].set_title("FAC de residuos (excluyendo el primero)")

plot_pacf(
    residuos_sin_primero,
    lags=24,
    zero=False,
    method="ywm",
    ax=axes[1]
)
axes[1].set_title("PACF de residuos (excluyendo el primero)")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].hist(
    residuos_sin_primero,
    bins="auto",
    density=True,
    edgecolor="black",
    alpha=0.75
)

x_grid = np.linspace(
    residuos_sin_primero.min(),
    residuos_sin_primero.max(),
    300
)

axes[0].plot(
    x_grid,
    stats.norm.pdf(
        x_grid,
        residuos_sin_primero.mean(),
        residuos_sin_primero.std(ddof=1)
    ),
    linewidth=2
)

axes[0].set_title("Histograma de residuos")
axes[0].set_xlabel("Residuo")
axes[0].set_ylabel("Densidad")

qqplot(residuos_sin_primero, line="45", fit=True, ax=axes[1])
axes[1].set_title("Q-Q plot de residuos")

plt.tight_layout()
plt.show()


# ================================================================
# 5. LJUNG-BOX HASTA EL REZAGO 24
# ================================================================

def decision_pvalor(p):
    if pd.isna(p):
        return "No disponible"
    return "Rechazar H0" if p < 0.05 else "No rechazar H0"

rezagos = list(range(1, 25))

lb_completo = acorr_ljungbox(
    residuos, lags=rezagos, model_df=2, return_df=True
)

lb_sin_primero = acorr_ljungbox(
    residuos_sin_primero,
    lags=rezagos,
    model_df=2,
    return_df=True
)

lb_completo["Decisión"] = lb_completo["lb_pvalue"].apply(
    decision_pvalor
)
lb_sin_primero["Decisión"] = lb_sin_primero["lb_pvalue"].apply(
    decision_pvalor
)

print("\nLJUNG-BOX: MUESTRA COMPLETA")
display(lb_completo.round(4))

print("\nLJUNG-BOX: SIN EL PRIMER RESIDUO")
display(lb_sin_primero.round(4))


# ================================================================
# 6. JARQUE-BERA
# ================================================================

jb_stat, jb_p, asimetria, curtosis = jarque_bera(
    residuos_sin_primero
)

print("\nJARQUE-BERA")
print(f"Estadístico: {jb_stat:.4f}")
print(f"Valor p:     {jb_p:.6f}")
print(f"Asimetría:   {asimetria:.4f}")
print(f"Curtosis:    {curtosis:.4f}")
print("Decisión:", decision_pvalor(jb_p))


# ================================================================
# 7. ARCH-LM
# ================================================================

arch_lm, arch_p, arch_f, arch_f_p = het_arch(
    residuos_sin_primero, nlags=12
)

print("\nARCH-LM (12 REZAGOS)")
print(f"Estadístico LM: {arch_lm:.4f}")
print(f"Valor p LM:     {arch_p:.6f}")
print(f"Estadístico F:  {arch_f:.4f}")
print(f"Valor p F:      {arch_f_p:.6f}")
print("Decisión LM:", decision_pvalor(arch_p))
print("Decisión F: ", decision_pvalor(arch_f_p))


# ================================================================
# 8. ESTABILIDAD E INVERTIBILIDAD
# ================================================================

print("\nRAÍCES AR:", modelo.arroots)
print("Módulos AR:", np.abs(modelo.arroots))
print("RAÍCES MA:", modelo.maroots)
print("Módulos MA:", np.abs(modelo.maroots))

print(
    "AR estable:",
    bool(np.all(np.abs(modelo.arroots) > 1))
)
print(
    "MA invertible:",
    bool(np.all(np.abs(modelo.maroots) > 1))
)


# ================================================================
# 9. PRONÓSTICO ESTÁNDAR DE REFERENCIA
# ================================================================

pred_estandar = modelo.get_forecast(steps=H)
frame_estandar = pred_estandar.summary_frame(alpha=ALPHA)

print("\nPRONÓSTICO ESTÁNDAR DE REFERENCIA")
display(
    frame_estandar[
        ["mean", "mean_ci_lower", "mean_ci_upper"]
    ].round(3)
)


# ================================================================
# 10. BOOTSTRAP DE RESIDUOS PARA EL PRONÓSTICO FUTURO
# ================================================================

# Se centra la distribución empírica de residuos.
# Se excluye el primer residuo porque corresponde al efecto de
# inicialización de la serie integrada y no es comparable con
# los errores ordinarios posteriores.

errores_boot = residuos_sin_primero.to_numpy().copy()
errores_boot = errores_boot - errores_boot.mean()

if len(errores_boot) < 30:
    raise ValueError("Hay muy pocos residuos para el bootstrap.")

# Extraer parámetros estimados.
parametros = dict(zip(modelo.param_names, modelo.params))

c = float(parametros["x1"])
phi = float(parametros["ar.L1"])
theta = float(parametros["ma.L1"])

# Última diferencia observada y último residuo estimado.
ultima_diferencia = float(y.diff().dropna().iloc[-1])
ultimo_residuo = float(residuos.iloc[-1])

# Remuestreo de errores con reemplazo.
# Cada fila representa una trayectoria y cada columna un horizonte.
innovaciones = rng.choice(
    errores_boot,
    size=(B, H),
    replace=True
)

trayectorias = np.zeros((B, H))

diferencia_anterior = np.full(B, ultima_diferencia)
error_anterior = np.full(B, ultimo_residuo)
nivel_anterior = float(y.iloc[-1])

for h in range(H):
    error_nuevo = innovaciones[:, h]

    # Ecuación de la serie diferenciada:
    # Δy_t = deriva + phi*Δy_(t-1) + e_t + theta*e_(t-1)
    diferencia_simulada = (
        c
        + phi * diferencia_anterior
        + error_nuevo
        + theta * error_anterior
    )

    nivel_anterior = nivel_anterior + diferencia_simulada
    trayectorias[:, h] = nivel_anterior

    diferencia_anterior = diferencia_simulada
    error_anterior = error_nuevo

# Resumen empírico de las trayectorias simuladas.
media_boot = np.mean(trayectorias, axis=0)
mediana_boot = np.median(trayectorias, axis=0)

inferior_boot = np.quantile(trayectorias, ALPHA / 2, axis=0)
superior_boot = np.quantile(trayectorias, 1 - ALPHA / 2, axis=0)

fechas_futuras = pd.date_range(
    start=y.index[-1] + pd.offsets.MonthBegin(1),
    periods=H,
    freq="MS"
)

tabla_bootstrap = pd.DataFrame({
    "Media bootstrap": media_boot,
    "Mediana bootstrap": mediana_boot,
    "IC bootstrap 95 % inferior": inferior_boot,
    "IC bootstrap 95 % superior": superior_boot
}, index=fechas_futuras)

tabla_bootstrap.index.name = "Fecha"

print("\nPRONÓSTICO CON BOOTSTRAP DE RESIDUOS")
print(f"Trayectorias simuladas: {B}")
display(tabla_bootstrap.round(3))

print("\nComparación de trayectorias centrales:")
tabla_comparacion = pd.DataFrame({
    "Pronóstico estándar": np.asarray(
        frame_estandar["mean"]
    ),
    "Media bootstrap": media_boot,
    "Mediana bootstrap": mediana_boot
}, index=fechas_futuras)

display(tabla_comparacion.round(3))


# ================================================================
# 11. GRÁFICA DEL PRONÓSTICO CON INTERVALO BOOTSTRAP
# ================================================================

fig, ax = plt.subplots(figsize=(13, 6))

ax.plot(
    y.index,
    y,
    color="black",
    linewidth=1.5,
    label="IPC energético observado"
)

ax.plot(
    fechas_futuras,
    media_boot,
    color="red",
    linewidth=2,
    label="Media de las trayectorias bootstrap"
)

ax.fill_between(
    fechas_futuras,
    inferior_boot,
    superior_boot,
    color="red",
    alpha=0.18,
    label="Intervalo bootstrap del 95 %"
)

ax.set_title(
    "IPC energético: pronóstico ARIMA(1,1,1) con bootstrap"
)
ax.set_xlabel("Fecha")
ax.set_ylabel("Índice")
ax.legend()
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.show()


# ================================================================
# 12. DETALLE DE LOS ÚLTIMOS 36 MESES + PRONÓSTICO
# ================================================================

y_reciente = y.iloc[-min(36, len(y)):]

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    y_reciente.index,
    y_reciente,
    color="black",
    linewidth=1.5,
    label="IPC observado"
)

ax.plot(
    fechas_futuras,
    media_boot,
    color="red",
    linewidth=2,
    label="Media bootstrap"
)

ax.fill_between(
    fechas_futuras,
    inferior_boot,
    superior_boot,
    color="red",
    alpha=0.18,
    label="IC bootstrap 95 %"
)

ax.set_title("Detalle del pronóstico del IPC energético")
ax.set_xlabel("Fecha")
ax.set_ylabel("Índice")
ax.legend()
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.show()


# ================================================================
# 13. VALIDACIÓN RETROSPECTIVA FUERA DE MUESTRA
# ================================================================

# Se reserva el último año. La estimación de validación es
# independiente de la estimación principal sobre toda la muestra.

if len(y) <= 36:
    raise ValueError("No hay suficientes datos para validar 12 meses.")

y_train = y.iloc[:-12]
y_test = y.iloc[-12:]

modelo_validacion = ARIMA(
    y_train,
    order=(1, 1, 1),
    trend="t"
).fit()

pred_validacion = modelo_validacion.get_forecast(steps=12)
frame_validacion = pred_validacion.summary_frame(alpha=ALPHA)

y_real = y_test.to_numpy()
y_hat = frame_validacion["mean"].to_numpy()

errores = y_real - y_hat

mae = np.mean(np.abs(errores))
rmse = np.sqrt(np.mean(errores ** 2))

mascara = y_real != 0
mape = (
    np.mean(np.abs(errores[mascara] / y_real[mascara])) * 100
    if np.any(mascara)
    else np.nan
)

tabla_validacion = pd.DataFrame({
    "Observado": y_real,
    "Pronóstico": y_hat,
    "Error": errores,
    "Error absoluto": np.abs(errores)
}, index=y_test.index)

tabla_validacion.index.name = "Fecha"

print("\nVALIDACIÓN RETROSPECTIVA: ÚLTIMOS 12 MESES")
display(tabla_validacion.round(3))

tabla_metricas = pd.DataFrame({
    "Métrica": ["MAE", "RMSE", "MAPE (%)"],
    "Valor": [mae, rmse, mape]
})

print("\nMÉTRICAS DE VALIDACIÓN")
display(tabla_metricas.round(4))


# ================================================================
# 14. GRÁFICA DE VALIDACIÓN RETROSPECTIVA
# ================================================================

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    y_train.index,
    y_train,
    color="black",
    linewidth=1.2,
    label="Muestra de estimación"
)

ax.plot(
    y_test.index,
    y_test,
    color="black",
    linewidth=1.8,
    label="Valores observados de evaluación"
)

ax.plot(
    y_test.index,
    y_hat,
    color="red",
    linewidth=2,
    label="Pronóstico retrospectivo"
)

ax.set_title("Validación retrospectiva ARIMA(1,1,1) con deriva")
ax.set_xlabel("Fecha")
ax.set_ylabel("Índice")
ax.legend()
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.show()


# ================================================================
# 15. EXPORTAR RESULTADOS A EXCEL
# ================================================================

archivo_salida = "Resultados_ARIMA111_deriva_BOOTSTRAP.xlsx"

with pd.ExcelWriter(archivo_salida) as writer:
    tabla_parametros.to_excel(writer, sheet_name="Parametros")
    tabla_criterios.to_excel(
        writer, sheet_name="AIC_BIC", index=False
    )
    residuos.to_frame().to_excel(
        writer, sheet_name="Residuos_completos"
    )
    residuos_sin_primero.to_frame().to_excel(
        writer, sheet_name="Residuos_sin_primero"
    )
    lb_completo.to_excel(
        writer, sheet_name="LjungBox_completo"
    )
    lb_sin_primero.to_excel(
        writer, sheet_name="LjungBox_sin_primero"
    )
    tabla_bootstrap.to_excel(
        writer, sheet_name="Pronostico_bootstrap"
    )
    tabla_comparacion.to_excel(
        writer, sheet_name="Comparacion_pronosticos"
    )
    tabla_validacion.to_excel(
        writer, sheet_name="Validacion"
    )
    tabla_metricas.to_excel(
        writer, sheet_name="Metricas", index=False
    )

print("\n" + "=" * 70)
print("ANÁLISIS FINALIZADO")
print("=" * 70)
print("Archivo Excel:", archivo_salida)
print("Modelo principal: ARIMA(1,1,1) con deriva")
print("Bootstrap: remuestreo de residuos centrados")
print("Trayectorias:", B)
print("Horizonte:", H, "meses")
print("Intervalos bootstrap: percentiles 2.5 % y 97.5 %")

