# Para activar el entorno virtual antes de ejecutar este script, usar:
#PS C:\Users\pbarg> 
# cd "C:\Users\pbarg\Documents\UNIVERSIDAD\4\TFG\VisualStudioCode"
#PS C:\Users\pbarg\Documents\UNIVERSIDAD\4\TFG\VisualStudioCode> 
#.venv\Scripts\Activate.ps1

# =============================================================================
# TFG: ANALISIS PREDICTIVO E INGENIERIA CUANTITATIVA PARA OPTIMIZACION
#      DE CARTERAS - DAX40
# =============================================================================
# ARCHIVO: DAX_Analisis_del_Dato.py
# PREREQUISITO: Ejecutar DAX_Ingenieria_del_Dato.py primero (secciones 1-12)
#               para generar 7_dataset_final_completo.xlsx
#
# Este script implementa las secciones 14-19 (Analisis del Dato):
#   Seccion 14: Marco Teorico
#   Seccion 15: Carga de Datos y Preparacion (division train/test 80/20)
#   Seccion 16: Feature Engineering avanzado para ML
#   Seccion 17: Modelo Supervisado (GradientBoosting + MLP Ensemble)
#   Seccion 18: Modelo No Supervisado (HMM - Deteccion de Regimenes)
#   Seccion 19: Metricas de Adecuacion de los Modelos (3+)
#   Seccion 19.4: Guardar artefactos (modelos, parametros, datos) para Negocio
#
# PREREQUISITO PARA DAX_Analisis_de_Negocio.py:
#   Ejecutar este script genera /modelos/ con todos los artefactos necesarios
#   para que DAX_Analisis_de_Negocio.py ejecute las secciones 20-23
#   (comparacion de estrategias) sin re-entrenar los modelos.
#
# MODELOS IMPLEMENTADOS:
#   SUPERVISADO: MLPRegressor
#       -> Predice retornos esperados de cada activo (mu_pred)
#   NO SUPERVISADO: Hidden Markov Model (HMM) para deteccion de regimenes
#       -> Clasifica estado del mercado (bull / bear / neutral)
#   COMBINACION: Optimizador Markowitz ML-Enhanced que integra ambos
#       -> mu = blend(prediccion_ML, media_historica) segun confianza regimen
#       -> Sigma = EWMA + Ledoit-Wolf, condicionada por regimen HMM
#
# BENCHMARK Y DAX40:
#   El DAX40 se usa EXCLUSIVAMENTE como benchmark de comparacion (buy-and-hold).
#   NO se incluye como feature del modelo por riesgo de multicolinealidad extrema
#   (el indice es la media ponderada de los mismos activos). Ver justificacion
#   completa en DAX_Ingenieria_del_Dato.py, Seccion 3.
# =============================================================================

# 0. IMPORTACION DE LIBRERIAS
import os
import sys
import io
import warnings
import numpy as np
import pandas as pd
import datetime
import subprocess

# Configurar stdout para UTF-8 en Windows
if sys.platform == 'win32': 
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from scipy import stats
from scipy.optimize import minimize as scipy_minimize
from scipy.special import logsumexp as _logsumexp

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor   # reemplaza GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.covariance import LedoitWolf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance        # para HistGBR (sin feature_importances_ MDI)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

# Directorio de trabajo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mapeo UNICO ticker -> sector (fuente de verdad para todo el pipeline).
# Replica exacta de DAX_Ingenieria_del_Dato.py. Si un ticker entra o sale
# del DAX40, se modifica SOLO aqui (y en el script de ingenieria).
TICKER_SECTOR = {
    'ADS.DE':  'Consumo',       'ZAL.DE':  'Consumo',
    'AIR.DE':  'Industrial',    'MTX.DE':  'Industrial',
    'RHM.DE':  'Industrial',    'SY1.DE':  'Industrial',
    'BNR.DE':  'Industrial',    'HEI.DE':  'Industrial',
    'ALV.DE':  'Finanzas',      'MUV2.DE': 'Finanzas',
    'DBK.DE':  'Finanzas',      'HNR1.DE': 'Finanzas',
    'DB1.DE':  'Finanzas',      'CBK.DE':  'Finanzas',
    'BAS.DE':  'Quimica',       'HEN3.DE': 'Quimica',
    'BEI.DE':  'Quimica',
    'BAYN.DE': 'Salud',         'MRK.DE':  'Salud',
    'FRE.DE':  'Salud',         'FME.DE':  'Salud',
    'QIA.DE':  'Salud',         'SHL.DE':  'Salud',
    'BMW.DE':  'Automocion',    'MBG.DE':  'Automocion',
    'VOW3.DE': 'Automocion',    'CON.DE':  'Automocion',
    'DTG.DE':  'Automocion',    'PAH3.DE': 'Automocion',
    'P911.DE': 'Automocion',
    'SAP.DE':  'Tecnologia',    'SIE.DE':  'Tecnologia',
    'IFX.DE':  'Tecnologia',
    'RWE.DE':  'Energia',       'EOAN.DE': 'Energia',
    'ENR.DE':  'Energia',
    'DTE.DE':  'Telecom',
    'VNA.DE':  'Inmobiliario',
    'DHL.DE':  'Logistica',
}


# =============================================================================
# IMPLEMENTACION PROPIA: GaussianHMM (Hidden Markov Model)
# =============================================================================
# Implementacion nativa en NumPy/SciPy. No requiere hmmlearn (sin wheels
# para Python 3.14). Algoritmos: Baum-Welch (EM), Viterbi, Forward-Backward.
# Referencia: Rabiner, L. (1989). A tutorial on HMMs and selected applications
#             in speech recognition. Proc. IEEE, 77(2), 257-286.
# =============================================================================

class GaussianHMM:
    """
    Hidden Markov Model con emisiones gaussianas multivariantes.

    Implementacion propia sin dependencias externas (hmmlearn).
    Usa log-aritmetica (scipy.special.logsumexp) para estabilidad numerica.

    Parametros
    ----------
    n_components : int
        Numero de estados ocultos (regimenes).
    n_iter : int
        Iteraciones maximas de Baum-Welch (EM).
    tol : float
        Criterio de convergencia (delta log-likelihood).
    n_init : int
        Numero de inicializaciones aleatorias (se retiene la mejor).
    random_state : int or None
        Semilla para reproducibilidad.
    """

    def __init__(self, n_components=3, n_iter=100, tol=1e-4,
                 n_init=5, random_state=42):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        # Atributos aprendidos (se rellenan en fit)
        self.startprob_ = None   # (K,)
        self.transmat_ = None    # (K, K)
        self.means_ = None       # (K, d)
        self.covars_ = None      # (K, d, d)

    # ------------------------------------------------------------------
    # Distribucion de emision
    # ------------------------------------------------------------------
    def _log_emission(self, X):
        """Log P(x_t | estado=k) para cada t y k.  Forma: (T, K)."""
        T, d = X.shape
        K = self.n_components
        log_prob = np.empty((T, K))
        for k in range(K):
            diff = X - self.means_[k]                             # (T, d)
            cov_k = self.covars_[k] + 1e-6 * np.eye(d)           # regulariz.
            L = np.linalg.cholesky(cov_k)                         # (d, d)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            solved = np.linalg.solve(L, diff.T)                   # (d, T)
            log_prob[:, k] = -0.5 * (d * np.log(2 * np.pi)
                                     + log_det
                                     + np.sum(solved ** 2, axis=0))
        return log_prob

    # ------------------------------------------------------------------
    # Forward / Backward  (log-espacio, vectorizado sobre K)
    # ------------------------------------------------------------------
    def _forward(self, log_B):
        """Algoritmo forward.  Devuelve log_alpha (T, K)."""
        T, K = log_B.shape
        log_alpha = np.empty((T, K))
        log_alpha[0] = np.log(self.startprob_ + 1e-300) + log_B[0]
        log_A = np.log(self.transmat_ + 1e-300)                   # (K, K)
        for t in range(1, T):
            # (K,1) + (K,K) -> (K,K);  logsumexp por columna -> (K,)
            log_alpha[t] = (_logsumexp(log_alpha[t - 1, :, None] + log_A,
                                       axis=0)
                            + log_B[t])
        return log_alpha

    def _backward(self, log_B):
        """Algoritmo backward.  Devuelve log_beta (T, K)."""
        T, K = log_B.shape
        log_beta = np.empty((T, K))
        log_beta[T - 1] = 0.0                                     # log(1)
        log_A = np.log(self.transmat_ + 1e-300)
        for t in range(T - 2, -1, -1):
            # (K,K) + (K,) + (K,) -> (K,K);  logsumexp por fila -> (K,)
            log_beta[t] = _logsumexp(log_A
                                     + log_B[t + 1][None, :]
                                     + log_beta[t + 1][None, :],
                                     axis=1)
        return log_beta

    # ------------------------------------------------------------------
    # Entrenamiento: Baum-Welch (EM)
    # ------------------------------------------------------------------
    def fit(self, X):
        """Entrena el HMM con el algoritmo Baum-Welch (EM).

        Ejecuta n_init inicializaciones y retiene la de mayor log-likelihood.
        """
        T, d = X.shape
        K = self.n_components
        best_score = -np.inf
        best_params = None

        for init_i in range(self.n_init):
            rng = np.random.RandomState(
                (self.random_state + init_i) if self.random_state is not None
                else None)

            # --- Inicializacion ---
            idx = rng.choice(T, K, replace=False)
            means = X[idx].copy()
            data_cov = np.cov(X.T) + 1e-6 * np.eye(d)
            covars = np.array([data_cov.copy() for _ in range(K)])
            startprob = np.ones(K) / K
            diag_val = 0.7 + 0.2 * rng.random()
            transmat = np.full((K, K), (1 - diag_val) / max(K - 1, 1))
            np.fill_diagonal(transmat, diag_val)

            # Asignar temporalmente para usar _log_emission / _forward / _backward
            self.startprob_ = startprob
            self.transmat_ = transmat
            self.means_ = means
            self.covars_ = covars

            prev_ll = -np.inf
            for _ in range(self.n_iter):
                # --- E-step ---
                log_B = self._log_emission(X)
                log_alpha = self._forward(log_B)
                log_beta = self._backward(log_B)

                # gamma_t(k) = P(s_t=k | O)
                log_gamma = log_alpha + log_beta
                log_gamma -= _logsumexp(log_gamma, axis=1, keepdims=True)
                gamma = np.exp(log_gamma)                          # (T, K)

                # xi_t(i,j) = P(s_t=i, s_{t+1}=j | O)
                log_A = np.log(transmat + 1e-300)
                # Vectorizado: (T-1, K, 1) + (K, K) + (T-1, 1, K) + (T-1, 1, K)
                log_xi = (log_alpha[:-1, :, None]
                          + log_A[None, :, :]
                          + log_B[1:, None, :]
                          + log_beta[1:, None, :])                 # (T-1, K, K)
                log_xi -= _logsumexp(
                    log_xi.reshape(T - 1, -1), axis=1
                )[:, None, None]
                xi = np.exp(log_xi)                                # (T-1, K, K)

                # --- M-step ---
                startprob = gamma[0] + 1e-300
                startprob /= startprob.sum()

                xi_sum = xi.sum(axis=0) + 1e-300                   # (K, K)
                transmat = xi_sum / xi_sum.sum(axis=1, keepdims=True)

                for k in range(K):
                    gk = gamma[:, k]                               # (T,)
                    gk_sum = gk.sum() + 1e-300
                    means[k] = (gk[:, None] * X).sum(axis=0) / gk_sum
                    diff = X - means[k]                            # (T, d)
                    covars[k] = ((gk[:, None, None]
                                  * (diff[:, :, None] * diff[:, None, :]))
                                 .sum(axis=0) / gk_sum
                                 + 1e-6 * np.eye(d))

                self.startprob_ = startprob
                self.transmat_ = transmat
                self.means_ = means
                self.covars_ = covars

                # Convergencia
                ll = _logsumexp(log_alpha[-1])
                if abs(ll - prev_ll) < self.tol:
                    break
                prev_ll = ll

            final_ll = _logsumexp(self._forward(self._log_emission(X))[-1])
            if final_ll > best_score:
                best_score = final_ll
                best_params = (startprob.copy(), transmat.copy(),
                               means.copy(), covars.copy())

        # Restaurar mejores parametros
        self.startprob_, self.transmat_, self.means_, self.covars_ = best_params
        return self

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------
    def predict(self, X):
        """Decodificacion de Viterbi: secuencia de estados mas probable."""
        log_B = self._log_emission(X)
        T, K = log_B.shape
        log_A = np.log(self.transmat_ + 1e-300)

        V = np.empty((T, K))
        bp = np.zeros((T, K), dtype=int)
        V[0] = np.log(self.startprob_ + 1e-300) + log_B[0]

        for t in range(1, T):
            scores = V[t - 1, :, None] + log_A                    # (K, K)
            bp[t] = scores.argmax(axis=0)
            V[t] = scores[bp[t], np.arange(K)] + log_B[t]

        path = np.empty(T, dtype=int)
        path[T - 1] = V[T - 1].argmax()
        for t in range(T - 2, -1, -1):
            path[t] = bp[t + 1, path[t + 1]]
        return path

    def predict_proba(self, X):
        """Probabilidades posteriores P(s_t=k | O) (forward-backward)."""
        log_B = self._log_emission(X)
        log_alpha = self._forward(log_B)
        log_beta = self._backward(log_B)
        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)

    def score(self, X):
        """Log-likelihood total de la secuencia observada."""
        log_B = self._log_emission(X)
        log_alpha = self._forward(log_B)
        return float(_logsumexp(log_alpha[-1]))

    def bic(self, X):
        """Bayesian Information Criterion (menor = mejor)."""
        T, d = X.shape
        K = self.n_components
        # Parametros libres: startprob + transmat + means + full covars
        n_params = ((K - 1)
                    + K * (K - 1)
                    + K * d
                    + K * d * (d + 1) // 2)
        return n_params * np.log(T) - 2.0 * self.score(X)


def construir_sectores(tickers_disponibles):
    """Construye dict {sector: [tickers]} filtrando solo tickers presentes."""
    sectores = {}
    for ticker in tickers_disponibles:
        sector = TICKER_SECTOR.get(ticker)
        if sector:
            sectores.setdefault(sector, []).append(ticker)
    return {s: sorted(ts) for s, ts in sorted(sectores.items())}


# =============================================================================
# SECCION 14: MARCO TEORICO
# =============================================================================
#
# 14.1 TEORIA MODERNA DE CARTERAS (Markowitz, 1952)
# -------------------------------------------------
# Harry Markowitz formalizo la optimizacion de carteras como un problema de
# optimizacion cuadratica: maximizar el retorno esperado E[r_p] para un nivel
# de riesgo sigma_p dado, donde:
#   E[r_p] = w' * mu        (retorno esperado de la cartera)
#   sigma_p = sqrt(w' * Sigma * w)  (volatilidad de la cartera)
#   sujeto a: sum(w) = 1, w_i >= 0
#
# La frontera eficiente es el conjunto de carteras que maximizan E[r_p]
# para cada nivel de sigma_p. La cartera de maximo Sharpe Ratio
# S = (E[r_p] - r_f) / sigma_p es la solucion optima.
#
# 14.2 LIMITACIONES DEL MODELO CLASICO
# -------------------------------------
# Markowitz presenta limitaciones bien documentadas en la literatura:
#
#   1. ESTIMACION DE RETORNOS ESPERADOS (mu):
#      La media historica de retornos es un estimador muy ruidoso del retorno
#      futuro esperado. Merton (1980) demostro que se necesitan siglos de datos
#      para estimar mu con precision razonable. Este es el "talon de Aquiles"
#      de Markowitz: pequenos errores en mu producen carteras radicalmente
#      diferentes (Chopra & Ziemba, 1993).
#
#   2. ESTIMACION DE LA COVARIANZA (Sigma):
#      La covarianza muestral con ventana rodante:
#      a) Da igual peso a observaciones de hace 252 dias que a las de ayer
#      b) Es inestable con N activos / T observaciones cercano a 1
#      c) No captura cambios de regimen (la correlacion entre activos
#         AUMENTA drasticamente en crisis - Longin & Solnik, 2001)
#
#   3. ASUNCION DE ESTACIONARIEDAD:
#      Markowitz asume que mu y Sigma son constantes. Los mercados financieros
#      exhiben cambios de regimen (Hamilton, 1989), clustering de volatilidad
#      (Engle, 1982), y dependencias no lineales. El EDA de la Seccion 12
#      confirmo estos fenomenos en los datos del DAX40:
#      - Test ARCH significativo (p<0.01) en todas las empresas
#      - Autocorrelacion significativa (Ljung-Box) en retornos
#      - Descomposicion STL con componente estacional significativa
#
# 14.3 MACHINE LEARNING EN OPTIMIZACION DE CARTERAS
# --------------------------------------------------
# La literatura reciente propone tres vias de mejora:
#
#   A) MEJORA DE mu (retornos esperados):
#      Usar ML para predecir retornos en lugar de la media historica.
#      - Gu, Kelly & Xiu (2020, Review of Financial Studies): demuestran que
#        modelos de ML superan al benchmark historico en prediccion de retornos.
#      - GradientBoosting ha mostrado ser particularmente efectivo en datos
#        tabulares financieros (Leippold, Wang & Zhou, 2021).
#
#   B) MEJORA DE Sigma (covarianza):
#      - Ledoit & Wolf (2004): shrinkage reduce error de estimacion.
#      - EWMA (RiskMetrics, 1996): ponderacion exponencial da mas peso a
#        observaciones recientes, capturando mejor la dinamica temporal.
#      - Combinacion EWMA + Ledoit-Wolf + condicionamiento por regimen:
#        estado del arte en gestion cuantitativa de carteras.
#
#   C) DETECCION DE REGIMENES:
#      Modelos no supervisados para detectar estados del mercado:
#      - Hamilton (1989): Hidden Markov Models para switching regimes
#      - Hidden Markov Models (HMM): modelan transiciones temporales
#        entre regimenes con asignacion "soft" probabilistica
#      - Aplicacion: ajustar agresividad de la cartera segun el regimen
#
# 14.4 JUSTIFICACION DE MODELOS ELEGIDOS
# ----------------------------------------
#
# MODELO SUPERVISADO: GradientBoostingRegressor
#   - El EDA mostro autocorrelacion en retornos -> features temporales son utiles
#   - Heterocedasticidad (ARCH) -> los arboles no asumen varianza constante
#   - Distribuciones no-gaussianas (kurtosis >3) -> arboles no asumen normalidad
#   - Superior a MLP/LSTM en datos tabulares con features ingenieriles
#     (Grinsztajn, Oyallon & Varoquaux, 2022, NeurIPS)
#   - Interpretable via feature importance (ventaja para el TFG)
#
# MODELO NO SUPERVISADO: Hidden Markov Model (HMM)
#   - El EDA mostro clusters de volatilidad (ARCH) -> regimenes de mercado
#   - HMM proporciona probabilidades posteriores (soft clustering)
#     via algoritmo forward-backward, no solo etiquetas duras como K-Means
#   - Permite calcular P(regimen=bull|datos), P(regimen=bear|datos)
#   - Modela TRANSICIONES TEMPORALES entre regimenes (propiedad de Markov):
#     un dia BEAR tiene alta probabilidad de ser seguido por otro dia BEAR
#   - El algoritmo de Viterbi decodifica la secuencia GLOBALMENTE optima
#     de regimenes, reduciendo el "flickering" espurio entre estados
#   - Alineado con Hamilton (1989): referencia seminal para switching regimes
#   - Implementacion propia en NumPy/SciPy (Baum-Welch + Viterbi)
#
# COMBINACION (ML-Enhanced Markowitz):
#   mu_enhanced = alpha * mu_ML + (1 - alpha) * mu_historico
#   Sigma_enhanced = EWMA(halflife=63) + Ledoit-Wolf shrinkage
#   alpha = f(confianza_regimen): mayor en regimenes estables, menor en crisis
#   Restricciones: sum(w)=1, w_i>=0, w_i<=0.30, turnover penalty
#
# REFERENCIAS:
#   [1] Markowitz, H. (1952). Portfolio Selection. J. Finance, 7(1), 77-91.
#   [2] Ledoit, O. & Wolf, M. (2004). Honey, I shrunk the sample covariance matrix.
#   [3] Gu, S., Kelly, B. & Xiu, D. (2020). Empirical Asset Pricing via ML.
#   [4] Hamilton, J. (1989). A new approach to the economic analysis of
#       nonstationary time series. Econometrica, 57(2), 357-384.
#   [5] Nystrup, P. et al. (2018). Dynamic portfolio optimization across
#       hidden market regimes. Quantitative Finance, 18(1), 83-95.

print("="*70)
print("TFG DAX40: ANALISIS DEL DATO - OPTIMIZACION ML-ENHANCED")
print("="*70)


# =============================================================================
# SECCION 15: CARGA DE DATOS Y PREPARACION
# =============================================================================

print(f"\n" + "="*70)
print("SECCION 15: CARGA DE DATOS PREPROCESADOS")
print("="*70)

ruta_dataset_csv = os.path.join(BASE_DIR, '7_dataset_final_completo.csv')
ruta_dataset_xlsx = os.path.join(BASE_DIR, '7_dataset_final_completo.xlsx')

if not os.path.exists(ruta_dataset_csv) and not os.path.exists(ruta_dataset_xlsx):
    print(f"  [ERROR] No se encontro el dataset")
    print(f"  Ejecutar primero DAX_Ingenieria_del_Dato.py")
    sys.exit(1)

print(f"\n  [15.1] Cargando dataset preprocesado...")
if os.path.exists(ruta_dataset_csv):
    dataset_long = pd.read_csv(ruta_dataset_csv)
    print(f"    (Cargado desde CSV - lectura rapida)")
else:
    dataset_long = pd.read_excel(ruta_dataset_xlsx)
    dataset_long.to_csv(ruta_dataset_csv, index=False)
    print(f"    (Cargado desde Excel, CSV generado para proximas ejecuciones)")
dataset_long['Date'] = pd.to_datetime(dataset_long['Date'])

# Extraer metadatos
seleccionadas = sorted(dataset_long['Ticker'].unique().tolist())
n_activos = len(seleccionadas)

print(f"    Shape: {dataset_long.shape}")
print(f"    Activos seleccionados ({n_activos}): {seleccionadas}")
print(f"    Periodo: {dataset_long['Date'].min().date()} a {dataset_long['Date'].max().date()}")
print(f"    Columnas: {dataset_long.columns.tolist()}")

# Verificar columnas necesarias
cols_necesarias = ['Date', 'Ticker', 'Adj Close', 'Log_Return_D', 'Log_Return_Wins']
cols_macro = ['EURUSD_Close', 'STOXX50_Close', 'ORO_Close']
# MODIFICACIÓN NUEVA: Añadir VIX y TNX como features macro si estan disponibles.
# VIX captura volatilidad implicita (miedo del mercado), TNX captura tipos de interes.
cols_macro_ret = ['EURUSD_Log_Return_D', 'STOXX50_Log_Return_D', 'ORO_Log_Return_D',
                  'VIX_Log_Return_D', 'TNX_Log_Return_D']
cols_benchmark = ['DAX40_Close']

for col in cols_necesarias:
    if col not in dataset_long.columns:
        raise ValueError(f"Columna requerida '{col}' no encontrada en dataset")

# Verificar macro features disponibles
cols_macro_exist = [c for c in cols_macro_ret if c in dataset_long.columns]
print(f"    Features macro disponibles: {cols_macro_exist}")

# =============================================================================
# 15.2 USO DEL DAX40 COMO BENCHMARK (NO COMO FEATURE)
# =============================================================================
#
# El DAX40 se usa SOLO como benchmark buy-and-hold (ver justificacion
# detallada en Seccion 3 de DAX_Ingenieria_del_Dato.py).
# Motivos principales de exclusion como feature:
#   - Multicolinealidad extrema (~0.85-0.95 con cartera de activos DAX)
#   - Look-ahead bias: DAX40[t] incorpora precios de los activos a predecir
#   - Las features macro (STOXX50, ORO, VIX, TNX) son genuinamente exogenas

# Extraer benchmark DAX40 para comparacion posterior
dax40_benchmark = None
if 'DAX40_Close' in dataset_long.columns:
    # DAX40_Close es igual para todos los tickers (se repite por fecha)
    primer_ticker = seleccionadas[0]
    _dax40_data = dataset_long[dataset_long['Ticker'] == primer_ticker][['Date', 'DAX40_Close']].copy()
    _dax40_data = _dax40_data.set_index('Date').sort_index().dropna()
    dax40_benchmark = _dax40_data['DAX40_Close']
    print(f"\n  [15.2] DAX40 benchmark extraido: {len(dax40_benchmark)} observaciones")
    print(f"    -> Se usara SOLO como referencia buy-and-hold en backtest")
    print(f"    -> Se EXCLUYE de features del modelo ML (multicolinealidad)")
else:
    print(f"\n  [15.2] [WARN] DAX40_Close no encontrado, benchmark no disponible")

# =============================================================================
# 15.3 DIVISION TEMPORAL TRAIN / TEST (80/20)
# =============================================================================

fecha_min = dataset_long['Date'].min()
fecha_max = dataset_long['Date'].max()
rango_dias = (fecha_max - fecha_min).days
fecha_corte = fecha_min + pd.Timedelta(days=int(rango_dias * 0.80))

mask_train = dataset_long['Date'] <= fecha_corte
mask_test = dataset_long['Date'] > fecha_corte

dataset_train = dataset_long[mask_train].copy().reset_index(drop=True)
dataset_test = dataset_long[mask_test].copy().reset_index(drop=True)

print(f"\n  [15.3] Division temporal 80/20:")
print(f"    Train: {dataset_train['Date'].min().date()} a {dataset_train['Date'].max().date()} "
      f"({len(dataset_train):,} obs, {mask_train.sum()/len(dataset_long)*100:.1f}%)")
print(f"    Test : {dataset_test['Date'].min().date()} a {dataset_test['Date'].max().date()} "
      f"({len(dataset_test):,} obs, {mask_test.sum()/len(dataset_long)*100:.1f}%)")

# Pivotear retornos a formato matricial (fechas x activos)
retornos_train_wide = (
    dataset_train.pivot(index='Date', columns='Ticker', values='Log_Return_D')
    .sort_index()[seleccionadas].dropna()
)
retornos_test_wide = (
    dataset_test.pivot(index='Date', columns='Ticker', values='Log_Return_D')
    .sort_index()[seleccionadas].dropna()
)

print(f"    Retornos train (wide): {retornos_train_wide.shape}")
print(f"    Retornos test (wide):  {retornos_test_wide.shape}")

# OPTIMIZACIÓN: definir retornos_full aqui una sola vez; se reutiliza en Sec. 18, 20, 21
retornos_full = pd.concat([retornos_train_wide, retornos_test_wide]).sort_index()
print(f"    Retornos full (wide):  {retornos_full.shape}")


# =============================================================================
# SECCION 16: FEATURE ENGINEERING AVANZADO PARA ML
# =============================================================================
#
# OBJETIVO:
#   Construir un conjunto rico de features que capture la informacion temporal,
#   de volatilidad y macroeconomica necesaria para predecir retornos.
#
# JUSTIFICACION:
#   El EDA (seccion 12) revelo:
#   1. Autocorrelacion significativa -> features de lags y momentum
#   2. Heterocedasticidad (ARCH) -> features de volatilidad realizada
#   3. Correlaciones variables -> features cross-sectional
#   4. Impacto macro -> features EURUSD, STOXX50, ORO
#
# FEATURES POR ACTIVO (i):
#   - ret_lag_1..5:     Retornos pasados (1-5 dias)
#   - mom_5, mom_22:    Momentum a 5 y 22 dias (suma retornos)
#   - vol_22, vol_66:   Volatilidad realizada 22d y 66d (std retornos)
#   - skew_22:          Asimetria 22d (riesgo de cola)
#   - rel_strength_22:  Fuerza relativa vs media cross-sectional
#
# FEATURES GLOBALES (compartidas):
#   - eurusd_ret_1d..5d:  Retornos EUR/USD recientes
#   - stoxx_ret_1d..5d:   Retornos EuroStoxx 50 recientes
#   - oro_ret_1d..5d:     Retornos Oro recientes
#   - market_vol_22:      Volatilidad media del mercado (proxy VIX europeo)
#   - dispersion_22:      Dispersion cross-sectional (divergencia entre activos)

print(f"\n" + "="*70)
print("SECCION 16: FEATURE ENGINEERING AVANZADO")
print("="*70)


def construir_features(retornos_wide, macro_df, tickers, ventana_lag=5, volume_wide=None):
    """
    Construye features temporales, de volatilidad y macro para cada activo.

    Args:
        retornos_wide: DataFrame (fechas x tickers) de retornos diarios
        macro_df: DataFrame con columnas macro por fecha
        tickers: lista de tickers
        ventana_lag: numero de lags individuales
        volume_wide: DataFrame (fechas x tickers) de volumen (opcional)

    Returns:
        dict: {ticker: DataFrame de features (fechas x n_features)}
        target_dict: {ticker: Series de retorno siguiente dia}
    """
    features_dict = {}
    target_dict = {}

    # Features cross-sectional
    market_ret = retornos_wide.mean(axis=1)      # retorno medio del mercado
    market_vol22 = market_ret.rolling(22).std()   # volatilidad de mercado
    dispersion22 = retornos_wide.std(axis=1).rolling(22).mean()  # dispersion

    for ticker in tickers:
        if ticker not in retornos_wide.columns:
            continue

        ret = retornos_wide[ticker]
        feats = pd.DataFrame(index=retornos_wide.index)

        # --- Lags de retornos ---
        for lag in range(1, ventana_lag + 1):
            feats[f'ret_lag_{lag}'] = ret.shift(lag)

        # --- Momentum ---
        feats['mom_5'] = ret.rolling(5).sum()
        feats['mom_22'] = ret.rolling(22).sum()
        feats['mom_66'] = ret.rolling(66).sum()

        # --- Volatilidad realizada ---
        feats['vol_22'] = ret.rolling(22).std()
        feats['vol_66'] = ret.rolling(66).std()

        # --- Asimetria y kurtosis rolling ---
        feats['skew_22'] = ret.rolling(22).skew()
        feats['kurt_22'] = ret.rolling(22).kurt()

        # --- Fuerza relativa ---
        feats['rel_strength_22'] = ret.rolling(22).sum() - market_ret.rolling(22).sum()

        # --- Features cross-sectional ---
        feats['market_vol_22'] = market_vol22
        feats['dispersion_22'] = dispersion22
        feats['market_ret_5'] = market_ret.rolling(5).sum()

        # --- Features macro ---
        # MODIFICACIÓN NUEVA: Incluir VIX y TNX en las features macro del modelo.
        # VIX captura miedo del mercado (volatilidad implicita); TNX captura tipos de interes.
        for macro_col in ['EURUSD_Log_Return_D', 'STOXX50_Log_Return_D', 'ORO_Log_Return_D',
                          'VIX_Log_Return_D', 'TNX_Log_Return_D']:
            if macro_col in macro_df.columns:
                macro_name = macro_col.replace('_Log_Return_D', '').lower()
                macro_serie = macro_df[macro_col]
                # Alinear por indice
                macro_aligned = macro_serie.reindex(retornos_wide.index)
                feats[f'{macro_name}_ret_1d'] = macro_aligned.shift(1)
                feats[f'{macro_name}_mom_5'] = macro_aligned.rolling(5).sum()
                feats[f'{macro_name}_vol_22'] = macro_aligned.rolling(22).std()

        # MODIFICACIÓN NUEVA: Features de volumen (liquidez/actividad).
        # El ratio volumen/media detecta rupturas de rango y confirma tendencias.
        # Un volumen anormalmente alto (ratio > 2) indica eventos corporativos.
        if volume_wide is not None and ticker in volume_wide.columns:
            vol_ticker = volume_wide[ticker].reindex(retornos_wide.index)
            vol_ma22 = vol_ticker.rolling(22).mean()
            feats['vol_ratio_22'] = vol_ticker / (vol_ma22 + 1e-8)
            feats['vol_trend_5'] = vol_ticker.rolling(5).mean() / (vol_ma22 + 1e-8)

        # --- Target: retorno del dia siguiente ---
        target = ret.shift(-1)

        features_dict[ticker] = feats
        target_dict[ticker] = target

    return features_dict, target_dict


# Extraer macro features del dataset_train (son iguales para todos los tickers)
_primer_ticker = seleccionadas[0]
macro_train_df = (
    dataset_train[dataset_train['Ticker'] == _primer_ticker]
    [['Date'] + [c for c in cols_macro_ret if c in dataset_train.columns]]
    .set_index('Date').sort_index()
)
macro_test_df = (
    dataset_test[dataset_test['Ticker'] == _primer_ticker]
    [['Date'] + [c for c in cols_macro_ret if c in dataset_test.columns]]
    .set_index('Date').sort_index()
)

# MODIFICACIÓN NUEVA: Extraer volumen para features de liquidez.
# El volumen detecta rupturas de rango y confirma tendencias de precio.
_has_volume = 'Volume' in dataset_train.columns
if _has_volume:
    volume_train_wide = (
        dataset_train.pivot(index='Date', columns='Ticker', values='Volume')
        .sort_index()[seleccionadas]
    )
    volume_test_wide = (
        dataset_test.pivot(index='Date', columns='Ticker', values='Volume')
        .sort_index()[seleccionadas]
    )
    print(f"    [OK] Volumen extraido para features de liquidez")
else:
    volume_train_wide = None
    volume_test_wide = None
    print(f"    [INFO] Columna Volume no disponible, features de volumen desactivadas")

# OPTIMIZACIÓN: una sola llamada sobre el dataset completo (todas las features son
# backward-looking: shift>=1, rolling backward) → sin look-ahead bias y sin
# truncar las ventanas rolling al inicio del periodo de test.
# Se slicean train/test por fecha, equivalente a llamadas separadas pero:
#   1) Ahorra 2 llamadas a construir_features (~10-30s cada una)
#   2) Mejora calidad de features al inicio del test (sin warmup NaN)
print(f"\n  [16.1] Construyendo features sobre dataset completo (unica llamada)...")
macro_full_df = pd.concat([macro_train_df, macro_test_df]).sort_index()
volume_full = pd.concat([volume_train_wide, volume_test_wide]).sort_index() if _has_volume else None
features_all, targets_all = construir_features(
    retornos_full, macro_full_df, seleccionadas,
    volume_wide=volume_full
)
# Slice por fecha: identico a llamadas separadas para fechas dentro de train
features_train = {t: f[f.index <= fecha_corte] for t, f in features_all.items()}
targets_train  = {t: s[s.index <= fecha_corte] for t, s in targets_all.items()}
features_test  = {t: f[f.index >  fecha_corte] for t, f in features_all.items()}
targets_test   = {t: s[s.index >  fecha_corte] for t, s in targets_all.items()}

print(f"\n  [16.2] Features train/test derivados por slicing temporal")

# Mostrar resumen de features
ejemplo_ticker = seleccionadas[0]
n_features_total = features_train[ejemplo_ticker].shape[1]
print(f"\n  [16.3] Resumen de features:")
print(f"    Features por activo: {n_features_total}")
print(f"    Columnas: {features_train[ejemplo_ticker].columns.tolist()}")

# Concatenar todos los activos para entrenamiento global
def _concat_features_targets(features_dict, targets_dict, tickers):
    """Concatena features y targets de todos los activos en un solo dataset."""
    X_list, y_list, meta_list = [], [], []
    for ticker in tickers:
        if ticker not in features_dict:
            continue
        feats = features_dict[ticker].copy()
        tgt = targets_dict[ticker].copy()
        # Alinear y eliminar NaN
        df_combined = feats.copy()
        df_combined['_target'] = tgt
        df_combined = df_combined.dropna()
        if len(df_combined) == 0:
            continue
        X_list.append(df_combined.drop(columns='_target').values)
        y_list.append(df_combined['_target'].values)
        meta_list.extend([(ticker, d) for d in df_combined.index])
    return np.vstack(X_list), np.concatenate(y_list), meta_list

X_train_all, y_train_all, meta_train = _concat_features_targets(
    features_train, targets_train, seleccionadas
)
print(f"\n  [16.4] Dataset de entrenamiento consolidado:")
print(f"    X_train: {X_train_all.shape}")
print(f"    y_train: {y_train_all.shape}")

# Normalizar features
scaler_features = StandardScaler()
X_train_scaled = scaler_features.fit_transform(X_train_all)

print(f"\n    Features normalizadas con StandardScaler")

# GRAFICO 16.1 - Heatmap de correlacion de features
print(f"\n  [16.5] Generando heatmap de correlacion de features...")
_feat_concat = pd.concat([features_train[t].dropna() for t in seleccionadas], axis=0)
_feat_corr = _feat_concat.corr()
_mask_triu = np.triu(np.ones_like(_feat_corr, dtype=bool))
fig_hm, ax_hm = plt.subplots(figsize=(18, 14))
sns.heatmap(_feat_corr, mask=_mask_triu, cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, linewidths=0.3, ax=ax_hm,
            annot=False, cbar_kws={'label': 'Correlacion de Pearson'})
ax_hm.set_title(
    'Heatmap de Correlacion de Features (Train Set)\n'
    'Correlaciones altas (|r|>0.7) indican potencial multicolinealidad',
    fontsize=12, fontweight='bold')
ax_hm.tick_params(axis='x', rotation=90, labelsize=7)
ax_hm.tick_params(axis='y', rotation=0, labelsize=7)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '16_1_heatmap_correlacion_features.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 16_1_heatmap_correlacion_features.png")


# =============================================================================
# SECCION 17: MODELO SUPERVISADO - GRADIENTBOOSTING + MLP ENSEMBLE
# =============================================================================
#
# JUSTIFICACION DETALLADA DEL GRADIENTBOOSTING:
# -----------------------------------------------------------
# 1. NATURALEZA DE LOS DATOS:
#    Los retornos del DAX40 son datos tabulares con features ingenieriles
#    (lags, momentum, volatilidad). Grinsztajn et al. (2022, NeurIPS)
#    demostraron que los modelos de arboles (GradientBoosting, Random Forest)
#    superan consistentemente a las redes neuronales en datos tabulares de
#    dimensionalidad media (<100 features).
#
# 2. PROPIEDADES ESTADISTICAS DETECTADAS EN EL EDA:
#    a) Heterocedasticidad (ARCH): Los arboles de decision NO asumen
#       varianza constante. Cada split adapta sus predicciones al rango
#       local de la variable, capturando naturalmente la volatilidad variable.
#    b) No-gaussianidad (kurtosis > 3): Los arboles son no-parametricos,
#       no asumen distribucion alguna en los datos ni en los errores.
#    c) Autocorrelacion (Ljung-Box): Las features de lags y momentum
#       codifican explicitamente la estructura temporal que los arboles
#       pueden explotar mediante splits condicionales.
#
# 3. VENTAJAS SOBRE ALTERNATIVAS:
#    vs LSTM: LSTM necesita >5000 observaciones para generalizar bien
#       (Lim & Zohren, 2021). Nuestro dataset: ~1800 obs/activo en train.
#       GradientBoosting es mas eficiente con muestras limitadas.
#    vs Random Forest: GradientBoosting optimiza secuencialmente el error
#       residual (boosting), siendo mas preciso que el promediado de RF.
#    vs Regresion Lineal: Relaciones no lineales entre features
#       (ej: momentum alto + volatilidad baja -> retorno positivo,
#       pero momentum alto + volatilidad alta -> retorno incierto).
#
# ARQUITECTURA:
#   Modelo 1 (GradientBoosting): n_estimators=200, max_depth=4, lr=0.05
#   Modelo 2 (MLPRegressor): capas (128, 64, 32), relu, alpha=0.01
#   Ensemble: promedio ponderado de predicciones (70% GB + 30% MLP)
#     Justificacion: GB captura interacciones; MLP captura no-linealidades
#     globales. El ensamble reduce varianza de prediccion.
#
# VALIDACION:
#   TimeSeriesSplit con 5 folds (respeta orden temporal, sin look-ahead bias)

print(f"\n" + "="*70)
print("SECCION 17: MODELO SUPERVISADO (GradientBoosting + MLP Ensemble)")
print("="*70)

# 17.1 Validacion cruzada temporal
print(f"\n  [17.1] Validacion cruzada temporal (TimeSeriesSplit, 5 folds)...")

tscv = TimeSeriesSplit(n_splits=5)
scores_gb_cv = []
scores_mlp_cv = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled)):
    X_tr_cv, X_va_cv = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr_cv, y_va_cv = y_train_all[train_idx], y_train_all[val_idx]

    # HistGradientBoosting (implementacion moderna tipo LightGBM, 10-100x mas rapido)
    gb_cv = HistGradientBoostingRegressor(
        max_iter=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=20, random_state=42
    )
    gb_cv.fit(X_tr_cv, y_tr_cv)
    rmse_gb = np.sqrt(mean_squared_error(y_va_cv, gb_cv.predict(X_va_cv)))
    scores_gb_cv.append(rmse_gb)

    # MLP
    mlp_cv = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), activation='relu', alpha=0.01,
        max_iter=300, batch_size=64, learning_rate='adaptive',
        learning_rate_init=0.001, random_state=42, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=15, verbose=False
    )
    mlp_cv.fit(X_tr_cv, y_tr_cv)
    rmse_mlp = np.sqrt(mean_squared_error(y_va_cv, mlp_cv.predict(X_va_cv)))
    scores_mlp_cv.append(rmse_mlp)

    print(f"    Fold {fold+1}: GB RMSE={rmse_gb:.6f}, MLP RMSE={rmse_mlp:.6f}")

print(f"\n    Media CV -> GB: {np.mean(scores_gb_cv):.6f} +/- {np.std(scores_gb_cv):.6f}")
print(f"    Media CV -> MLP: {np.mean(scores_mlp_cv):.6f} +/- {np.std(scores_mlp_cv):.6f}")

# 17.2 Entrenamiento final sobre todo el conjunto TRAIN
print(f"\n  [17.2] Entrenamiento final sobre todo el train set...")

# MODIFICACIÓN NUEVA: RandomizedSearchCV para GradientBoosting.
# En vez de fijar hiperparametros a mano, se exploran 40 combinaciones
# aleatorias con validacion cruzada temporal (TimeSeriesSplit 3 folds).
# Esto reduce el riesgo de overfitting a un unico juego de parametros
# y documenta que la eleccion es data-driven, no arbitraria.
print(f"    Ejecutando RandomizedSearchCV para GradientBoosting (40 iters, 3 folds)...")

_gb_param_dist = {
    'max_iter':           [100, 200, 300, 500],      # equiv. n_estimators
    'max_depth':          [3, 4, 5, None],            # None = sin limite (controlado por min_samples_leaf)
    'learning_rate':      [0.01, 0.03, 0.05, 0.08, 0.1],
    'min_samples_leaf':   [10, 20, 30, 50],
    'l2_regularization':  [0.0, 0.1, 1.0],           # regularizacion L2 (reemplaza subsample)
    'max_features':       [0.5, 0.7, 0.9, 1.0],      # fraccion de features por split (sklearn >= 1.4)
}

_gb_base = HistGradientBoostingRegressor(random_state=42)
_tscv_tuning = TimeSeriesSplit(n_splits=3)

_gb_search = RandomizedSearchCV(
    _gb_base, _gb_param_dist, n_iter=40,
    cv=_tscv_tuning, scoring='neg_root_mean_squared_error',
    random_state=42, n_jobs=1, verbose=0
)
_gb_search.fit(X_train_scaled, y_train_all)

print(f"    Mejores hiperparametros GB: {_gb_search.best_params_}")
print(f"    Mejor RMSE CV: {-_gb_search.best_score_:.6f}")

model_gb = _gb_search.best_estimator_

model_mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32), activation='relu', alpha=0.01,
    max_iter=500, batch_size=64, learning_rate='adaptive',
    learning_rate_init=0.001, random_state=42, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=20, verbose=False
)
model_mlp.fit(X_train_scaled, y_train_all)

# Predicciones in-sample
y_pred_gb_train = model_gb.predict(X_train_scaled)
y_pred_mlp_train = model_mlp.predict(X_train_scaled)

# Ensemble: pesos adaptativos basados en el RMSE del CV temporal (inverso-RMSE).
# Un modelo con menor error en CV recibe proporcionalmente mas peso en el ensamble.
# Esto es data-driven: si GB y MLP tienen el mismo RMSE obtenemos 50/50;
# si GB es claramente superior (habitual en tabulares) converge a ~70/30.
_inv_rmse_gb = 1.0 / (np.mean(scores_gb_cv) + 1e-10)
_inv_rmse_mlp = 1.0 / (np.mean(scores_mlp_cv) + 1e-10)
_total_inv = _inv_rmse_gb + _inv_rmse_mlp
PESO_GB = float(round(_inv_rmse_gb / _total_inv, 4))
PESO_MLP = float(round(_inv_rmse_mlp / _total_inv, 4))
print(f"    Pesos adaptativos (1/RMSE_CV): GB={PESO_GB:.3f}, MLP={PESO_MLP:.3f}")
y_pred_ensemble_train = PESO_GB * y_pred_gb_train + PESO_MLP * y_pred_mlp_train

print(f"    GB train RMSE:       {np.sqrt(mean_squared_error(y_train_all, y_pred_gb_train)):.6f}")
print(f"    MLP train RMSE:      {np.sqrt(mean_squared_error(y_train_all, y_pred_mlp_train)):.6f}")
print(f"    Ensemble train RMSE: {np.sqrt(mean_squared_error(y_train_all, y_pred_ensemble_train)):.6f}")

# 17.3 Feature Importance (HistGradientBoosting — permutation importance)
# HistGBR no expone feature_importances_ MDI. Se usa permutation_importance:
# mide cuanto sube el RMSE al permutar aleatoriamente cada feature (10 repeticiones).
# Ventaja: no sesgado por cardinalidad; mas fiable que MDI para features continuas.
print(f"\n  [17.3] Importancia de features (HistGBR - permutation importance):")
feature_names = features_train[seleccionadas[0]].columns.tolist()
_n_perm = min(3000, len(X_train_scaled))  # submuestra para velocidad
_perm_result = permutation_importance(
    model_gb, X_train_scaled[:_n_perm], y_train_all[:_n_perm],
    n_repeats=10, random_state=42, n_jobs=1
)
importances = _perm_result.importances_mean
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(f"\n    Top 10 features mas importantes:")
for i, row in importance_df.head(10).iterrows():
    print(f"      {row['Feature']:<25s} {row['Importance']:.6f}")

# GRAFICO 17.1 - Feature Importance
fig_fi, ax_fi = plt.subplots(figsize=(12, 8))
top_n = min(20, len(importance_df))
top_feats = importance_df.head(top_n)
ax_fi.barh(range(top_n), top_feats['Importance'].values[::-1],
           color='steelblue', edgecolor='black', linewidth=0.5)
ax_fi.set_yticks(range(top_n))
ax_fi.set_yticklabels(top_feats['Feature'].values[::-1], fontsize=9)
ax_fi.set_xlabel('Importancia (permutation importance - subida de RMSE al permutar)', fontsize=10)
ax_fi.set_title('Top 20 Features - HistGradientBoosting Supervisado\n'
    '(permutation importance: sin sesgo de cardinalidad vs MDI)',
    fontsize=11, fontweight='bold')
ax_fi.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '17_1_feature_importance_gb.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 17_1_feature_importance_gb.png")

# GRAFICO 17.3 - Permutation Importance del MLP + Comparacion GB vs MLP
print(f"\n  [17.3b] Permutation importance del MLP...")
_perm_result_mlp = permutation_importance(
    model_mlp, X_train_scaled[:_n_perm], y_train_all[:_n_perm],
    n_repeats=10, random_state=42, n_jobs=1
)
importances_mlp = _perm_result_mlp.importances_mean
importance_mlp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance_MLP': importances_mlp
}).sort_values('Importance_MLP', ascending=False)

print(f"\n    Top 10 features MLP:")
for _, _row in importance_mlp_df.head(10).iterrows():
    print(f"      {_row['Feature']:<25s} {_row['Importance_MLP']:.6f}")

top_n_comp = min(15, len(importance_df))
top_gb_comp = importance_df.head(top_n_comp)
top_mlp_comp = importance_mlp_df.head(top_n_comp)

_imp_merged = importance_df[['Feature', 'Importance']].merge(
    importance_mlp_df[['Feature', 'Importance_MLP']], on='Feature', how='outer'
).fillna(0).sort_values('Importance', ascending=False).head(top_n_comp)
_x_comp = np.arange(top_n_comp)
_w = 0.35

# Figura 17_3a: Permutation Importance GradientBoosting
fig_17_3a, ax_17_3a = plt.subplots(figsize=(10, 8))
ax_17_3a.barh(range(top_n_comp), top_gb_comp['Importance'].values[::-1],
              color='steelblue', edgecolor='black', linewidth=0.5)
ax_17_3a.set_yticks(range(top_n_comp))
ax_17_3a.set_yticklabels(top_gb_comp['Feature'].values[::-1], fontsize=8)
ax_17_3a.set_title(
    'Permutation Importance — GradientBoosting (Top features)\n'
    '(Incremento de RMSE al permutar aleatoriamente cada feature — 10 repeticiones)',
    fontsize=12, fontweight='bold')
ax_17_3a.set_xlabel('Permutation Importance (incremento RMSE)')
ax_17_3a.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '17_3a_permutation_importance_gb.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 17_3a_permutation_importance_gb.png")

# Figura 17_3b: Permutation Importance MLP
fig_17_3b, ax_17_3b = plt.subplots(figsize=(10, 8))
ax_17_3b.barh(range(top_n_comp), top_mlp_comp['Importance_MLP'].values[::-1],
              color='darkorange', edgecolor='black', linewidth=0.5)
ax_17_3b.set_yticks(range(top_n_comp))
ax_17_3b.set_yticklabels(top_mlp_comp['Feature'].values[::-1], fontsize=8)
ax_17_3b.set_title(
    'Permutation Importance — MLP (Red Neuronal, Top features)\n'
    '(Incremento de RMSE al permutar aleatoriamente cada feature — 10 repeticiones)',
    fontsize=12, fontweight='bold')
ax_17_3b.set_xlabel('Permutation Importance (incremento RMSE)')
ax_17_3b.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '17_3b_permutation_importance_mlp.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 17_3b_permutation_importance_mlp.png")

# Figura 17_3c: Comparacion directa GB vs MLP
fig_17_3c, ax_17_3c = plt.subplots(figsize=(10, 8))
ax_17_3c.barh(_x_comp + _w / 2, _imp_merged['Importance'].values[::-1],
              _w, label='GB', color='steelblue', alpha=0.8, edgecolor='black', lw=0.5)
ax_17_3c.barh(_x_comp - _w / 2, _imp_merged['Importance_MLP'].values[::-1],
              _w, label='MLP', color='darkorange', alpha=0.8, edgecolor='black', lw=0.5)
ax_17_3c.set_yticks(_x_comp)
ax_17_3c.set_yticklabels(_imp_merged['Feature'].values[::-1], fontsize=8)
ax_17_3c.set_title(
    'Comparacion de Permutation Importance: GradientBoosting vs MLP\n'
    '(barras paralelas por feature — azul=GB, naranja=MLP)',
    fontsize=12, fontweight='bold')
ax_17_3c.set_xlabel('Permutation Importance (incremento RMSE)')
ax_17_3c.legend(fontsize=9)
ax_17_3c.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '17_3c_comparacion_importancia_gb_vs_mlp.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 17_3c_comparacion_importancia_gb_vs_mlp.png")

# GRAFICO 17.2a - Convergencia del Loss del MLP
fig_conv_a, ax_conv_a = plt.subplots(figsize=(9, 5))
ax_conv_a.plot(model_mlp.loss_curve_, linewidth=1.5, color='steelblue', label='Train Loss')
ax_conv_a.set_title(
    'Curva de Convergencia del MLP — Loss por Iteracion de Entrenamiento\n'
    '(escala logaritmica: la curva decreciente indica aprendizaje correcto)',
    fontsize=12, fontweight='bold')
ax_conv_a.set_xlabel('Iteracion')
ax_conv_a.set_ylabel('Loss (MSE)')
ax_conv_a.set_yscale('log')
ax_conv_a.grid(alpha=0.3)
ax_conv_a.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '17_2a_convergencia_mlp_loss.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 17_2a_convergencia_mlp_loss.png")

# GRAFICO 17.2b - Validacion cruzada temporal por fold
x_folds = range(1, len(scores_gb_cv)+1)
fig_conv_b, ax_conv_b = plt.subplots(figsize=(9, 5))
ax_conv_b.plot(x_folds, scores_gb_cv, 'o-', label=f'GB (media={np.mean(scores_gb_cv):.5f})',
               color='darkgreen', linewidth=1.5)
ax_conv_b.plot(x_folds, scores_mlp_cv, 's-', label=f'MLP (media={np.mean(scores_mlp_cv):.5f})',
               color='darkorange', linewidth=1.5)
ax_conv_b.set_title(
    'Validacion Cruzada Temporal (TimeSeriesSplit, 5 folds) — RMSE por Fold\n'
    '(GB vs MLP: menor RMSE = mejor prediccion fuera de muestra)',
    fontsize=12, fontweight='bold')
ax_conv_b.set_xlabel('Fold')
ax_conv_b.set_ylabel('RMSE')
ax_conv_b.legend(fontsize=9)
ax_conv_b.grid(alpha=0.3)
ax_conv_b.set_xticks(list(x_folds))
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '17_2b_cv_folds_rmse.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 17_2b_cv_folds_rmse.png")


# =============================================================================
# SECCION 18: MODELO NO SUPERVISADO - HMM DETECCION DE REGIMENES
# =============================================================================
#
# JUSTIFICACION DETALLADA DEL HMM (Hidden Markov Model):
# -----------------------------------------------------------
# 1. MOTIVACION FINANCIERA:
#    Los mercados financieros no se comportan de forma uniforme. Existen
#    "regimenes" claramente diferenciados:
#    - BULL (alcista): retornos positivos, baja volatilidad, correlaciones bajas
#    - BEAR (bajista): retornos negativos, alta volatilidad, correlaciones altas
#    - NEUTRO (transicion): volatilidad moderada, sin tendencia clara
#
#    El EDA (seccion 12.3) confirmo heterocedasticidad significativa (ARCH)
#    en todos los activos -> la varianza NO es constante -> existen regimenes.
#
# 2. POR QUE HMM Y NO GMM NI K-MEANS:
#    a) K-Means produce asignacion "dura" (cluster 0 o 1, nada entre medias).
#       HMM produce probabilidades posteriores via forward-backward:
#       P(bull)=0.75, P(neutral)=0.20, P(bear)=0.05 -> transiciones SUAVES.
#    b) GMM trata cada observacion como INDEPENDIENTE. HMM modela la
#       ESTRUCTURA TEMPORAL: la probabilidad de estado hoy depende del
#       estado de ayer (propiedad de Markov). Esto es clave en finanzas
#       porque los regimenes son PERSISTENTES (un dia BEAR suele ir
#       seguido de otro dia BEAR).
#    c) El algoritmo de Viterbi del HMM decodifica la secuencia de estados
#       GLOBALMENTE optima, reduciendo el "flickering" espurio entre
#       regimenes que produce el GMM (cambios BULL-BEAR-BULL en 3 dias).
#    d) La MATRIZ DE TRANSICION es un parametro APRENDIDO del modelo
#       (no calculada empiricamente post-hoc como con GMM), lo que
#       permite analizar rigurosamente la dinamica entre regimenes.
#    e) Fundamentacion teorica clasica: Hamilton (1989) introdujo los
#       modelos Markov-switching especificamente para regimenes economicos.
#
# 3. IMPLEMENTACION:
#    Se usa una implementacion propia de GaussianHMM en NumPy/SciPy
#    (no depende de hmmlearn, que no tiene wheels para Python 3.14).
#    Algoritmos: Baum-Welch (EM) para entrenamiento, Viterbi para
#    decodificacion, forward-backward para probabilidades posteriores.
#
# FEATURES PARA DETECCION DE REGIMENES:
#   - Retorno medio del mercado (ultimos 22 dias)
#   - Volatilidad promedio del mercado (ultimos 22 dias)
#   - Correlacion media entre activos (ultimos 22 dias)
#   - Estas 3 features caracterizan completamente el "estado" del mercado.
#
# NUMERO DE COMPONENTES: 3
#   Justificacion: Nystrup et al. (2018) y Ang & Bekaert (2002) demuestran
#   que 2-3 regimenes son optimos para mercados de renta variable.
#   Se valida con BIC: se prueba k=2,3,4,5 y se elige k con menor BIC.

print(f"\n" + "="*70)
print("SECCION 18: MODELO NO SUPERVISADO (HMM - Deteccion de Regimenes)")
print("="*70)

# 18.1 Construir features de regimen sobre TODO el dataset
print(f"\n  [18.1] Construyendo features de regimen...")

# OPTIMIZACIÓN: reutilizar retornos_full (ya definido en Sec. 15) en lugar de
# reconstruir desde dataset_long. Resultado identico, evita un pivot+dropna.
# Features de regimen (rolling 22 dias)
regime_features = pd.DataFrame(index=retornos_full.index)
regime_features['market_ret_22'] = retornos_full.mean(axis=1).rolling(22).mean()
regime_features['market_vol_22'] = retornos_full.mean(axis=1).rolling(22).std()

# Correlacion media entre activos (proxy de "contagio")
def _rolling_corr_mean(df, window=22):
    """Calcula correlacion media cross-sectional con ventana rolling.

    OPTIMIZACIÓN: elimina el bucle Python fecha-a-fecha (O(T) lookups en
    MultiIndex) usando un reshape vectorizado sobre la salida de rolling().corr().
    rolling().corr() devuelve (T*n_cols, n_cols); reshape a (T, n_cols, n_cols)
    y extraemos la media del triangulo superior con indexacion booleana numpy.
    Speedup tipico: 100-500x frente al bucle original.
    """
    n_cols = df.shape[1]
    mask_upper = np.triu(np.ones((n_cols, n_cols), dtype=bool), k=1)
    # shape (T*n_cols, n_cols) → (T, n_cols, n_cols)
    corr_3d = df.rolling(window).corr().to_numpy().reshape(len(df), n_cols, n_cols)
    upper_means = corr_3d[:, mask_upper].mean(axis=1)  # (T,)
    result = pd.Series(upper_means, index=df.index, dtype=float)
    result.iloc[:window - 1] = np.nan
    return result

regime_features['avg_corr_22'] = _rolling_corr_mean(retornos_full, 22)
regime_features = regime_features.dropna()

print(f"    Features de regimen: {regime_features.shape}")
print(f"    Columnas: {regime_features.columns.tolist()}")

# 18.2 Seleccion de numero optimo de componentes (BIC)
print(f"\n  [18.2] Seleccion de K optimo (BIC)...")

# MODIFICACIÓN NUEVA: Entrenar HMM SOLO sobre datos de entrenamiento para evitar
# look-ahead bias. Antes se entrenaba sobre todo el dataset (train+test), lo que
# significaba que el modelo "veia" regimenes futuros al ajustar sus parametros.
# Ahora: fit scaler y HMM solo en train, luego transform/predict sobre todo.
regime_feature_cols = ['market_ret_22', 'market_vol_22', 'avg_corr_22']
regime_train_mask = regime_features.index <= fecha_corte
regime_train_data = regime_features[regime_train_mask][regime_feature_cols].values

scaler_regime = StandardScaler()
regime_scaled_train = scaler_regime.fit_transform(regime_train_data)

bic_scores = {}
for k in [2, 3, 4, 5]:
    hmm_test = GaussianHMM(n_components=k, n_iter=200,
                            random_state=42, n_init=5)
    hmm_test.fit(regime_scaled_train)
    bic_scores[k] = hmm_test.bic(regime_scaled_train)
    print(f"    k={k}: BIC={bic_scores[k]:.2f}")

k_optimo = min(bic_scores, key=bic_scores.get)
print(f"    [ELEGIDO] K={k_optimo} (menor BIC)")
print(f"    [INFO] HMM entrenado SOLO sobre train ({regime_train_mask.sum()} obs, sin look-ahead)")

# 18.3 Entrenar HMM definitivo
print(f"\n  [18.3] Entrenando HMM (Baum-Welch) con K={k_optimo}...")

hmm_model = GaussianHMM(
    n_components=k_optimo, n_iter=300,
    random_state=42, n_init=10
)
hmm_model.fit(regime_scaled_train)

# Transform sobre todo el dataset y predict (para visualizacion)
regime_scaled = scaler_regime.transform(regime_features[regime_feature_cols].values)

# Asignar regimenes (Viterbi: secuencia globalmente optima)
regimenes = hmm_model.predict(regime_scaled)
# Probabilidades posteriores (Forward-Backward)
proba_regimenes = hmm_model.predict_proba(regime_scaled)

# Agregar al DataFrame
regime_features['Regimen'] = regimenes

# 18.4 Caracterizar regimenes
print(f"\n  [18.4] Caracterizacion de regimenes:")
print(f"  {'Regimen':<10} {'N obs':>8} {'Ret medio':>12} {'Vol media':>12} {'Corr media':>12} {'Label':>10}")
print(f"  " + "-"*68)

# Ordenar regimenes por retorno medio (bull = mayor ret, bear = menor)
regimen_stats = {}
for r in range(k_optimo):
    mask_r = regime_features['Regimen'] == r
    stats_r = {
        'n_obs': mask_r.sum(),
        'ret_mean': regime_features.loc[mask_r, 'market_ret_22'].mean(),
        'vol_mean': regime_features.loc[mask_r, 'market_vol_22'].mean(),
        'corr_mean': regime_features.loc[mask_r, 'avg_corr_22'].mean(),
    }
    regimen_stats[r] = stats_r

# Asignar etiquetas descriptivas
sorted_by_ret = sorted(regimen_stats.keys(), key=lambda r: regimen_stats[r]['ret_mean'])
regimen_labels = {}
if k_optimo == 3:
    regimen_labels[sorted_by_ret[0]] = 'BEAR'
    regimen_labels[sorted_by_ret[1]] = 'NEUTRO'
    regimen_labels[sorted_by_ret[2]] = 'BULL'
elif k_optimo == 2:
    regimen_labels[sorted_by_ret[0]] = 'BEAR'
    regimen_labels[sorted_by_ret[1]] = 'BULL'
else:
    for i, r in enumerate(sorted_by_ret):
        regimen_labels[r] = f'R{i}'

for r in range(k_optimo):
    s = regimen_stats[r]
    label = regimen_labels[r]
    print(f"  {r:<10} {s['n_obs']:>8} {s['ret_mean']:>12.6f} {s['vol_mean']:>12.6f} "
          f"{s['corr_mean']:>12.4f} {label:>10}")

# 18.5 Matriz de transicion APRENDIDA por el HMM
# (Ventaja sobre GMM: la matriz es un PARAMETRO del modelo, no una estimacion
#  post-hoc a partir de predicciones. Es mas rigurosa estadisticamente.)
print(f"\n  [18.5] Matriz de transicion APRENDIDA por el HMM:")
transition_matrix_norm = hmm_model.transmat_

print(f"  De\\A  ", end="")
for r in range(k_optimo):
    print(f"  {regimen_labels[r]:>8}", end="")
print()
for r_from in range(k_optimo):
    print(f"  {regimen_labels[r_from]:<8}", end="")
    for r_to in range(k_optimo):
        print(f"  {transition_matrix_norm[r_from, r_to]:>8.3f}", end="")
    print()

print(f"\n    [INFO] Diagonal alta = persistencia de regimen (esperado en mercados)")
print(f"    [INFO] A diferencia del GMM, esta matriz es un parametro optimizado del modelo")

# Preparacion compartida para las tres figuras de regimenes
market_ret_daily = retornos_full.mean(axis=1)
market_ret_daily_aligned = market_ret_daily.reindex(regime_features.index)
market_cum = (1 + market_ret_daily_aligned).cumprod()

_cmap_reg = plt.cm.RdBu
colores_regimen_ext = {}
for _rank, r in enumerate(sorted_by_ret):
    _t = _rank / max(k_optimo - 1, 1)
    _t = 0.05 + _t * 0.90
    colores_regimen_ext[r] = _cmap_reg(_t)

# Figura 18_1a: Retornos acumulados con fondo de regimenes detectados
fig_18_1a, ax_18_1a = plt.subplots(figsize=(16, 5))
ax_18_1a.plot(regime_features.index, market_cum.values, color='black', linewidth=1)
for r in range(k_optimo):
    mask_r = regime_features['Regimen'].values == r
    ax_18_1a.fill_between(regime_features.index, market_cum.min(), market_cum.max(),
                           where=mask_r, alpha=0.25, color=colores_regimen_ext[r],
                           label=f'{regimen_labels[r]} ({regimen_stats[r]["n_obs"]} dias)')
ax_18_1a.set_ylabel('Retorno Acumulado')
ax_18_1a.set_xlabel('Fecha')
ax_18_1a.set_title(
    'Retornos Acumulados del Mercado DAX40 con Regimenes HMM Superpuestos (algoritmo Viterbi)\n'
    '(fondo coloreado = regimen detectado; rojo=BEAR, gris=NEUTRO, azul=BULL)',
    fontsize=12, fontweight='bold')
ax_18_1a.legend(fontsize=8, loc='upper left')
ax_18_1a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '18_1a_retornos_acumulados_regimenes.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 18_1a_retornos_acumulados_regimenes.png")

# Figura 18_1b: Probabilidades de regimen (area apilada, forward-backward)
_colores_stack = [colores_regimen_ext[r] for r in range(k_optimo)]
_labels_stack  = [f'P({regimen_labels[r]})' for r in range(k_optimo)]
fig_18_1b, ax_18_1b = plt.subplots(figsize=(16, 5))
ax_18_1b.stackplot(
    regime_features.index,
    [proba_regimenes[:, r] for r in range(k_optimo)],
    labels=_labels_stack, colors=_colores_stack, alpha=0.75)
ax_18_1b.set_ylabel('Probabilidad')
ax_18_1b.set_xlabel('Fecha')
ax_18_1b.set_title(
    'Probabilidades Posteriores de Regimen HMM — Area Apilada (algoritmo forward-backward)\n'
    '(suma total = 1 en cada instante; transiciones suaves = efecto del modelo Markov)',
    fontsize=12, fontweight='bold')
ax_18_1b.legend(fontsize=8, loc='upper left')
ax_18_1b.grid(alpha=0.3)
ax_18_1b.set_ylim([0.0, 1.0])
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '18_1b_probabilidades_regimenes_area.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 18_1b_probabilidades_regimenes_area.png")

# Figura 18_1c: Volatilidad del mercado coloreada por regimen
fig_18_1c, ax_18_1c = plt.subplots(figsize=(16, 5))
ax_18_1c.plot(regime_features.index, regime_features['market_vol_22'].values,
              color='gray', linewidth=0.8, alpha=0.5)
for r in range(k_optimo):
    mask_r = regime_features['Regimen'].values == r
    fechas_r = regime_features.index[mask_r]
    vol_r = regime_features.loc[mask_r, 'market_vol_22'].values
    ax_18_1c.scatter(fechas_r, vol_r, s=8, alpha=0.6,
                     color=colores_regimen_ext[r], label=regimen_labels[r])
ax_18_1c.set_ylabel('Volatilidad 22d')
ax_18_1c.set_xlabel('Fecha')
ax_18_1c.set_title(
    'Volatilidad del Mercado DAX40 Coloreada por Regimen HMM\n'
    '(picos en BEAR confirman heterocedasticidad; cada punto coloreado por regimen asignado)',
    fontsize=12, fontweight='bold')
ax_18_1c.legend(fontsize=8)
ax_18_1c.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '18_1c_volatilidad_por_regimen.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 18_1c_volatilidad_por_regimen.png")

# Figura 18_2a: Seleccion de K optimo por Criterio BIC
ks = list(bic_scores.keys())
bics = list(bic_scores.values())
fig_18_2a, ax_18_2a = plt.subplots(figsize=(8, 5))
ax_18_2a.plot(ks, bics, 'o-', color='steelblue', linewidth=2, markersize=10)
ax_18_2a.axvline(k_optimo, color='red', linestyle='--', label=f'K optimo={k_optimo} (BIC minimo)')
ax_18_2a.set_xlabel('Numero de Componentes (K)')
ax_18_2a.set_ylabel('BIC (menor = mejor)')
ax_18_2a.set_title(
    f'Seleccion del Numero Optimo de Regimenes HMM por Criterio BIC\n'
    f'(K={k_optimo} minimiza BIC={bic_scores[k_optimo]:.1f} — balance complejidad/ajuste)',
    fontsize=12, fontweight='bold')
ax_18_2a.legend()
ax_18_2a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '18_2a_seleccion_k_bic.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 18_2a_seleccion_k_bic.png")

# Figura 18_2b: Matriz de transicion aprendida por el HMM
xlabels = [regimen_labels[r] for r in range(k_optimo)]
fig_18_2b, ax_18_2b = plt.subplots(figsize=(7, 6))
im_tr = ax_18_2b.imshow(transition_matrix_norm, cmap='YlOrRd', vmin=0, vmax=1)
ax_18_2b.set_xticks(range(k_optimo))
ax_18_2b.set_yticks(range(k_optimo))
ax_18_2b.set_xticklabels(xlabels, fontsize=10)
ax_18_2b.set_yticklabels(xlabels, fontsize=10)
ax_18_2b.set_xlabel('Regimen Destino')
ax_18_2b.set_ylabel('Regimen Origen')
ax_18_2b.set_title(
    'Matriz de Transicion Aprendida por el HMM (Baum-Welch)\n'
    '(diagonal alta = regimenes persistentes — propiedad clave de los mercados)',
    fontsize=12, fontweight='bold')
for i in range(k_optimo):
    for j in range(k_optimo):
        ax_18_2b.text(j, i, f'{transition_matrix_norm[i,j]:.2f}',
                      ha='center', va='center', fontsize=12, fontweight='bold')
plt.colorbar(im_tr, ax=ax_18_2b, label='Probabilidad de transicion')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '18_2b_matriz_transicion_hmm.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 18_2b_matriz_transicion_hmm.png")


# =============================================================================
# SECCION 19: METRICAS DE ADECUACION DE LOS MODELOS (3+)
# =============================================================================
#
# DEFINICION Y JUSTIFICACION DE METRICAS:
#
# A) MODELO SUPERVISADO (GradientBoosting + MLP):
#
#   [M1] RMSE (Root Mean Squared Error):
#     RMSE = sqrt(mean((y_real - y_pred)^2))
#     Penaliza errores grandes cuadraticamente. Unidades: mismas que retorno.
#     Umbral: <0.015 aceptable para retornos diarios (~1.5% error).
#
#   [M2] MAE (Mean Absolute Error):
#     MAE = mean(|y_real - y_pred|)
#     Error medio absoluto, mas robusto a outliers que RMSE.
#     Interpretacion directa en puntos porcentuales de retorno.
#
#   [M3] R-cuadrado (Coeficiente de Determinacion):
#     R2 = 1 - SS_res/SS_tot
#     Proporcion de varianza explicada por el modelo.
#     En finanzas, R2 > 0.02 ya es significativo economicamente
#     (Gu, Kelly & Xiu, 2020). Retornos diarios tienen baja predictibilidad.
#
#   [M4] Directional Accuracy (Precision Direccional):
#     DA = % de dias donde sign(y_pred) == sign(y_real)
#     Metrica financiera critica: no importa la magnitud del error si
#     acertamos la DIRECCION (positivo/negativo). DA > 50% indica
#     capacidad de timing. Referencia: mercado eficiente -> DA = 50%.
#
# B) MODELO NO SUPERVISADO (HMM):
#
#   [M5] BIC (Bayesian Information Criterion):
#     BIC = k*ln(n) - 2*ln(L)  donde L = likelihood, k = parametros, n = obs
#     Penaliza complejidad, favorece parsimonia. Menor = mejor.
#     Se usa para elegir K optimo.
#
#   [M6] Silhouette Score:
#     s(i) = (b(i) - a(i)) / max(a(i), b(i))
#     a(i) = distancia media intra-cluster, b(i) = distancia media al cluster
#     mas cercano. Rango: [-1, 1], >0.25 aceptable, >0.50 bueno.
#
#   [M7] Log-Likelihood:
#     ln(P(datos|modelo)). Mayor = mejor ajuste a los datos observados.

print(f"\n" + "="*70)
print("SECCION 19: METRICAS DE ADECUACION DE LOS MODELOS")
print("="*70)

# --- Preparar datos TEST para evaluacion ---
X_test_all, y_test_all, meta_test = _concat_features_targets(
    features_test, targets_test, seleccionadas
)
X_test_scaled = scaler_features.transform(X_test_all)

# A) Metricas del modelo supervisado
print(f"\n  [19.1] METRICAS MODELO SUPERVISADO:")
print(f"  " + "-"*70)

y_pred_gb_test = model_gb.predict(X_test_scaled)
y_pred_mlp_test = model_mlp.predict(X_test_scaled)
y_pred_ensemble_test = PESO_GB * y_pred_gb_test + PESO_MLP * y_pred_mlp_test

# M1: RMSE
rmse_gb_test = np.sqrt(mean_squared_error(y_test_all, y_pred_gb_test))
rmse_mlp_test = np.sqrt(mean_squared_error(y_test_all, y_pred_mlp_test))
rmse_ens_test = np.sqrt(mean_squared_error(y_test_all, y_pred_ensemble_test))

print(f"\n  [M1] RMSE (Root Mean Squared Error):")
print(f"       GradientBoosting:  {rmse_gb_test:.6f}")
print(f"       MLP:               {rmse_mlp_test:.6f}")
print(f"       Ensemble (70/30):  {rmse_ens_test:.6f}")

# M2: MAE
mae_gb_test = mean_absolute_error(y_test_all, y_pred_gb_test)
mae_mlp_test = mean_absolute_error(y_test_all, y_pred_mlp_test)
mae_ens_test = mean_absolute_error(y_test_all, y_pred_ensemble_test)

print(f"\n  [M2] MAE (Mean Absolute Error):")
print(f"       GradientBoosting:  {mae_gb_test:.6f}")
print(f"       MLP:               {mae_mlp_test:.6f}")
print(f"       Ensemble (70/30):  {mae_ens_test:.6f}")

# M3: R2
r2_gb_test = r2_score(y_test_all, y_pred_gb_test)
r2_mlp_test = r2_score(y_test_all, y_pred_mlp_test)
r2_ens_test = r2_score(y_test_all, y_pred_ensemble_test)

print(f"\n  [M3] R-cuadrado (Coeficiente de Determinacion):")
print(f"       GradientBoosting:  {r2_gb_test:.6f}")
print(f"       MLP:               {r2_mlp_test:.6f}")
print(f"       Ensemble (70/30):  {r2_ens_test:.6f}")
print(f"       Nota: R2 > 0.02 en retornos diarios es economicamente significativo")

# M4: Directional Accuracy
da_gb = np.mean(np.sign(y_test_all) == np.sign(y_pred_gb_test)) * 100
da_mlp = np.mean(np.sign(y_test_all) == np.sign(y_pred_mlp_test)) * 100
da_ens = np.mean(np.sign(y_test_all) == np.sign(y_pred_ensemble_test)) * 100

print(f"\n  [M4] Directional Accuracy (Precision Direccional):")
print(f"       GradientBoosting:  {da_gb:.2f}%")
print(f"       MLP:               {da_mlp:.2f}%")
print(f"       Ensemble (70/30):  {da_ens:.2f}%")
print(f"       Referencia: mercado eficiente = 50% (al azar)")

# B) Metricas del modelo no supervisado
print(f"\n  [19.2] METRICAS MODELO NO SUPERVISADO (HMM):")
print(f"  " + "-"*70)

# M5: BIC
bic_optimo = bic_scores[k_optimo]
print(f"\n  [M5] BIC (Bayesian Information Criterion): {bic_optimo:.2f}")
print(f"       K elegido: {k_optimo} (menor BIC entre k=2,3,4,5)")

# M6: Silhouette Score
from sklearn.metrics import silhouette_score
sil_score = silhouette_score(regime_scaled, regimenes, metric='euclidean')
print(f"\n  [M6] Silhouette Score: {sil_score:.4f}")
print(f"       Interpretacion: {'BUENO' if sil_score > 0.25 else 'MODERADO'} "
      f"(>0.25 aceptable, >0.50 bueno)")

# M7: Log-Likelihood
log_lik = hmm_model.score(regime_scaled)
print(f"\n  [M7] Log-Likelihood: {log_lik:.2f}")
print(f"       Mayor = mejor ajuste a los datos observados")

residuos = y_test_all - y_pred_ensemble_test

# Figura 19_1a: Scatter real vs prediccion del Ensemble
lim = max(abs(y_test_all.min()), abs(y_test_all.max()), abs(y_pred_ensemble_test.min()), abs(y_pred_ensemble_test.max()))
fig_19_1a, ax_19_1a = plt.subplots(figsize=(8, 7))
ax_19_1a.scatter(y_test_all, y_pred_ensemble_test, alpha=0.15, s=10, color='steelblue')
ax_19_1a.plot([-lim, lim], [-lim, lim], 'r--', linewidth=2, label='Prediccion perfecta (y=x)')
ax_19_1a.set_xlabel('Retorno Real')
ax_19_1a.set_ylabel('Retorno Predicho (Ensemble GB+MLP)')
ax_19_1a.set_title(
    f'Retorno Real vs Predicho — Ensemble Supervisado (GB+MLP) en Test Set\n'
    f'R\u00b2={r2_ens_test:.4f} | Directional Accuracy={da_ens:.1f}% | RMSE={rmse_ens_test:.6f}',
    fontsize=12, fontweight='bold')
ax_19_1a.legend()
ax_19_1a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '19_1a_scatter_real_vs_predicho.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  [OK] Guardado: 19_1a_scatter_real_vs_predicho.png")

# Figura 19_1b: Distribucion de residuos del Ensemble
fig_19_1b, ax_19_1b = plt.subplots(figsize=(8, 6))
ax_19_1b.hist(residuos, bins=60, alpha=0.7, color='steelblue', edgecolor='black', density=True)
ax_19_1b.axvline(0, color='red', linewidth=2, linestyle='--', label='Cero (perfecto)')
ax_19_1b.axvline(residuos.mean(), color='darkgreen', linewidth=2, label=f'Media={residuos.mean():.5f}')
ax_19_1b.set_xlabel('Residuo (Real - Predicho)')
ax_19_1b.set_ylabel('Densidad')
ax_19_1b.set_title(
    f'Distribucion de Residuos del Ensemble GB+MLP — Test Set\n'
    f'Desviacion tipica={residuos.std():.5f} (residuos centrados en cero = sin sesgo sistematico)',
    fontsize=12, fontweight='bold')
ax_19_1b.legend(fontsize=8)
ax_19_1b.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '19_1b_distribucion_residuos.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 19_1b_distribucion_residuos.png")

# Figura 19_1c: Serie temporal de residuos (primeras 500 obs)
fig_19_1c, ax_19_1c = plt.subplots(figsize=(12, 4))
ax_19_1c.plot(residuos[:500], linewidth=0.8, color='gray', alpha=0.7)
ax_19_1c.axhline(0, color='red', linewidth=1, linestyle='--')
ax_19_1c.set_xlabel('Observacion (indice en el test set)')
ax_19_1c.set_ylabel('Residuo (Real - Predicho)')
ax_19_1c.set_title(
    'Serie Temporal de Residuos del Ensemble (primeras 500 observaciones del test set)\n'
    '(patron aleatorio alrededor de cero = ausencia de sesgo temporal sistematico)',
    fontsize=12, fontweight='bold')
ax_19_1c.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '19_1c_serie_temporal_residuos.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 19_1c_serie_temporal_residuos.png")

# Figura 19_1d: Comparativa de metricas por modelo (GB, MLP, Ensemble)
modelos = ['GB', 'MLP', 'Ensemble']
metricas_vals = {
    'RMSE': [rmse_gb_test, rmse_mlp_test, rmse_ens_test],
    'MAE': [mae_gb_test, mae_mlp_test, mae_ens_test],
    'DA (%)': [da_gb/100, da_mlp/100, da_ens/100],
}
x_pos = np.arange(len(modelos))
width = 0.25
fig_19_1d, ax_19_1d = plt.subplots(figsize=(9, 6))
for i, (metrica, vals) in enumerate(metricas_vals.items()):
    ax_19_1d.bar(x_pos + i*width, vals, width, label=metrica, alpha=0.7)
ax_19_1d.set_xticks(x_pos + width)
ax_19_1d.set_xticklabels(modelos)
ax_19_1d.set_ylabel('Valor de la metrica')
ax_19_1d.set_title(
    'Comparativa de Metricas de Evaluacion por Modelo — Test Set\n'
    '(RMSE y MAE: menor = mejor; DA%: mayor = mejor)',
    fontsize=12, fontweight='bold')
ax_19_1d.legend(fontsize=9)
ax_19_1d.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '19_1d_comparativa_metricas_modelos.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 19_1d_comparativa_metricas_modelos.png")

# GRAFICO 19.2 - Predicciones vs Realidad POR TICKER (individual)
# MODIFICACIÓN NUEVA: Permite evaluar en qué activos el modelo predice mejor.
# Es clave para el tribunal: si el modelo solo funciona con 3-4 tickers,
# la cartera optimizada estará sesgada. Ideal: todos los scatters similares.
if meta_test:
    n_sel = len(seleccionadas)
    n_cols_pt = min(5, n_sel)
    n_rows_pt = (n_sel + n_cols_pt - 1) // n_cols_pt
    fig_pt, axes_pt = plt.subplots(n_rows_pt, n_cols_pt, figsize=(4*n_cols_pt, 3.5*n_rows_pt))
    fig_pt.suptitle('Predicciones vs Realidad — Por Ticker (Test Set)\n'
                    'Diagonal perfecta = prediccion ideal. Nube dispersa = ruido dominante.',
                    fontsize=13, fontweight='bold', y=1.01)
    axes_flat = axes_pt.flatten() if n_sel > 1 else [axes_pt]
    for idx_t, ticker in enumerate(seleccionadas):
        ax_t = axes_flat[idx_t]
        ticker_mask = np.array([m[0] == ticker for m in meta_test])
        if ticker_mask.sum() < 10:
            ax_t.set_title(ticker, fontsize=9)
            ax_t.text(0.5, 0.5, 'Datos\ninsuf.', ha='center', va='center', transform=ax_t.transAxes)
            continue
        y_real_t = y_test_all[ticker_mask]
        y_pred_t = y_pred_ensemble_test[ticker_mask]
        ax_t.scatter(y_real_t, y_pred_t, alpha=0.3, s=8, color='steelblue')
        lim_min = min(y_real_t.min(), y_pred_t.min()) * 1.1
        lim_max = max(y_real_t.max(), y_pred_t.max()) * 1.1
        ax_t.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=0.8, alpha=0.6)
        _corr_t = np.corrcoef(y_real_t, y_pred_t)[0, 1] if len(y_real_t) > 2 else 0
        ax_t.set_title(f'{ticker} (r={_corr_t:.3f})', fontsize=9)
        ax_t.tick_params(labelsize=7)
        ax_t.grid(alpha=0.2)
    # Ocultar ejes sobrantes
    for idx_t in range(n_sel, len(axes_flat)):
        axes_flat[idx_t].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '19_2_predicciones_por_ticker.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Guardado: 19_2_predicciones_por_ticker.png")

# GRAFICO 19.3 - Diagnostico completo de residuos (QQ + ACF + Ljung-Box)
print(f"\n  [19.4] Diagnostico avanzado de residuos (QQ, ACF, Ljung-Box)...")

def _acf_manual(x, nlags=40):
    """ACF via numpy sin dependencia de statsmodels."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    c0 = np.dot(x, x) / n
    vals = [1.0]
    for lag in range(1, nlags + 1):
        vals.append(np.dot(x[lag:], x[:-lag]) / (n * c0 + 1e-300))
    return np.array(vals)

def _ljung_box(x, lags=20):
    """Ljung-Box Q: H0 = ruido blanco. Bajo H0 ~ chi2(m). p<0.05 -> autocorrelacion."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    acf_v = _acf_manual(x, nlags=lags)
    q_stats, p_vals = [], []
    for m in range(1, lags + 1):
        q = n * (n + 2) * np.sum(acf_v[1:m + 1] ** 2 / (n - np.arange(1, m + 1)))
        p = 1.0 - stats.chi2.cdf(q, df=m)
        q_stats.append(q)
        p_vals.append(p)
    return np.array(q_stats), np.array(p_vals)

_nlags_acf = 40
_acf_res   = _acf_manual(residuos, nlags=_nlags_acf)
_lb_q, _lb_p = _ljung_box(residuos, lags=20)
_conf_acf  = 1.96 / np.sqrt(len(residuos))

_kurt_res = stats.kurtosis(residuos)
_skew_res = stats.skew(residuos)

# Figura 19_3a: QQ-Plot de residuos vs Normal teorica
(osm, osr), (slope_qq, intercept_qq, r_qq) = stats.probplot(residuos, dist='norm')
_x_line = np.array([osm.min(), osm.max()])
fig_19_3a, ax_19_3a = plt.subplots(figsize=(8, 7))
ax_19_3a.scatter(osm, osr, alpha=0.25, s=10, color='steelblue', label='Residuos')
ax_19_3a.plot(_x_line, slope_qq * _x_line + intercept_qq, 'r-', linewidth=2, label='Normal teorica')
ax_19_3a.set_xlabel('Cuantiles Teoricos (Normal estandar)')
ax_19_3a.set_ylabel('Cuantiles de Muestra (residuos)')
ax_19_3a.set_title(
    f'QQ-Plot de Residuos del Ensemble vs Distribucion Normal (R={r_qq:.4f})\n'
    f'Colas gruesas por encima/debajo de la linea roja = distribucion leptocurtica (tipica en finanzas)',
    fontsize=12, fontweight='bold')
ax_19_3a.legend(fontsize=8)
ax_19_3a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '19_3a_qqplot_residuos.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 19_3a_qqplot_residuos.png")

# Figura 19_3b: Histograma de residuos con curva normal ajustada
_x_norm = np.linspace(residuos.min(), residuos.max(), 300)
fig_19_3b, ax_19_3b = plt.subplots(figsize=(8, 6))
ax_19_3b.hist(residuos, bins=70, density=True, alpha=0.6, color='steelblue',
              edgecolor='black', linewidth=0.4, label='Residuos (densidad)')
ax_19_3b.plot(_x_norm, stats.norm.pdf(_x_norm, residuos.mean(), residuos.std()),
              'r-', linewidth=2, label='Normal ajustada')
ax_19_3b.axvline(0, color='darkgreen', linestyle='--', linewidth=1.5, label='Cero')
ax_19_3b.set_xlabel('Residuo (Real - Predicho)')
ax_19_3b.set_ylabel('Densidad')
ax_19_3b.set_title(
    f'Histograma de Residuos del Ensemble con Curva Normal Superpuesta\n'
    f'Kurtosis={_kurt_res:.2f}  Skewness={_skew_res:.3f} (exceso de kurtosis = colas gruesas)',
    fontsize=12, fontweight='bold')
ax_19_3b.legend(fontsize=8)
ax_19_3b.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '19_3b_histograma_residuos.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 19_3b_histograma_residuos.png")

# Figura 19_3c: ACF de residuos (test de autocorrelacion)
_lags_x = np.arange(_nlags_acf + 1)
fig_19_3c, ax_19_3c = plt.subplots(figsize=(10, 5))
ax_19_3c.bar(_lags_x, _acf_res, color='steelblue', alpha=0.7, edgecolor='black', lw=0.4)
ax_19_3c.axhline( _conf_acf, color='red', linestyle='--', linewidth=1.2, label=f'+/-{_conf_acf:.4f} (IC 95%)')
ax_19_3c.axhline(-_conf_acf, color='red', linestyle='--', linewidth=1.2)
ax_19_3c.axhline(0, color='black', linewidth=0.8)
ax_19_3c.set_xlabel('Lag (dias)')
ax_19_3c.set_ylabel('Autocorrelacion (ACF)')
ax_19_3c.set_title(
    f'ACF de Residuos del Ensemble — Test de Independencia ({_nlags_acf} lags)\n'
    f'(barras dentro de bandas rojas = residuos no autocorrelacionados, deseable para un buen modelo)',
    fontsize=12, fontweight='bold')
ax_19_3c.legend(fontsize=8)
ax_19_3c.grid(alpha=0.3)
ax_19_3c.set_xlim([-0.5, _nlags_acf + 0.5])
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '19_3c_acf_residuos.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 19_3c_acf_residuos.png")

# Figura 19_3d: Ljung-Box p-valores por lag
_lags_lb = np.arange(1, len(_lb_p) + 1)
_colors_lb = ['#e74c3c' if p < 0.05 else 'steelblue' for p in _lb_p]
fig_19_3d, ax_19_3d = plt.subplots(figsize=(10, 5))
ax_19_3d.bar(_lags_lb, _lb_p, color=_colors_lb, alpha=0.85, edgecolor='black', lw=0.4)
ax_19_3d.axhline(0.05, color='red', linestyle='--', linewidth=1.5, label='alpha=0.05')
ax_19_3d.set_xlabel('Lag m')
ax_19_3d.set_ylabel('p-valor del test Ljung-Box')
ax_19_3d.set_title(
    'Test Ljung-Box sobre Residuos del Ensemble — p-valores por Lag\n'
    '(rojo: p<0.05 rechaza H\u2080 de ruido blanco; azul: residuos independientes en ese lag)',
    fontsize=12, fontweight='bold')
ax_19_3d.legend(fontsize=8)
ax_19_3d.grid(alpha=0.3)
ax_19_3d.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '19_3d_ljungbox_pvalores.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 19_3d_ljungbox_pvalores.png")

# GRAFICO 19.4 - Rolling RMSE 60 dias (Train vs Test)
print(f"\n  [19.5] Rolling RMSE 60 dias (Train vs Test)...")

_se_by_date_train = {}
for (ticker, date), y_r, y_p in zip(meta_train, y_train_all, y_pred_ensemble_train):
    _se_by_date_train.setdefault(date, []).append((y_r - y_p) ** 2)
_se_train_s = pd.Series(
    {d: np.mean(v) for d, v in _se_by_date_train.items()}
).sort_index()
_rolling_rmse_train = np.sqrt(_se_train_s.rolling(60, min_periods=20).mean())

_se_by_date_test = {}
for (ticker, date), y_r, y_p in zip(meta_test, y_test_all, y_pred_ensemble_test):
    _se_by_date_test.setdefault(date, []).append((y_r - y_p) ** 2)
_se_test_s = pd.Series(
    {d: np.mean(v) for d, v in _se_by_date_test.items()}
).sort_index()
_rolling_rmse_test = np.sqrt(_se_test_s.rolling(60, min_periods=20).mean())

fig_rrmse, ax_rrmse = plt.subplots(figsize=(15, 5))
ax_rrmse.plot(_rolling_rmse_train.index, _rolling_rmse_train.values,
              color='steelblue', linewidth=1.5, label='Train (rolling 60d)')
ax_rrmse.plot(_rolling_rmse_test.index, _rolling_rmse_test.values,
              color='darkorange', linewidth=2.0, label='Test (rolling 60d)')
ax_rrmse.axvline(fecha_corte, color='red', linestyle='--', linewidth=1.5,
                 label=f'Corte train/test ({fecha_corte.date()})')
ax_rrmse.fill_between(_rolling_rmse_train.index, _rolling_rmse_train.values,
                      alpha=0.12, color='steelblue')
ax_rrmse.fill_between(_rolling_rmse_test.index, _rolling_rmse_test.values,
                      alpha=0.12, color='darkorange')
ax_rrmse.set_xlabel('Fecha')
ax_rrmse.set_ylabel('RMSE (ventana 60 dias)')
ax_rrmse.set_title(
    'Rolling RMSE 60 dias: Train vs Test — Ensemble (GB+MLP)\n'
    'Divergencia alta entre curvas sugiere overfitting; convergencia = buena generalizacion',
    fontsize=11, fontweight='bold')
ax_rrmse.legend(fontsize=10)
ax_rrmse.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '19_4_rolling_rmse.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 19_4_rolling_rmse.png")

# Tabla resumen de metricas
print(f"\n  [19.3] TABLA RESUMEN DE METRICAS:")
print(f"  {'Modelo':<25} {'Metrica':<20} {'Valor':>12} {'Umbral':>12} {'Evaluacion':>12}")
print(f"  " + "="*85)
metricas_resumen = [
    ('GB Supervisado', 'RMSE', rmse_gb_test, '<0.015', 'OK' if rmse_gb_test < 0.015 else 'ALTO'),
    ('GB Supervisado', 'MAE', mae_gb_test, '<0.010', 'OK' if mae_gb_test < 0.010 else 'ALTO'),
    ('GB Supervisado', 'R2', r2_gb_test, '>0.02', 'OK' if r2_gb_test > 0.02 else 'BAJO'),
    ('GB Supervisado', 'Dir. Accuracy (%)', da_gb, '>50%', 'OK' if da_gb > 50 else 'BAJO'),
    ('MLP Supervisado', 'RMSE', rmse_mlp_test, '<0.015', 'OK' if rmse_mlp_test < 0.015 else 'ALTO'),
    ('MLP Supervisado', 'R2', r2_mlp_test, '>0.02', 'OK' if r2_mlp_test > 0.02 else 'BAJO'),
    ('Ensemble', 'RMSE', rmse_ens_test, '<0.015', 'OK' if rmse_ens_test < 0.015 else 'ALTO'),
    ('Ensemble', 'R2', r2_ens_test, '>0.02', 'OK' if r2_ens_test > 0.02 else 'BAJO'),
    ('Ensemble', 'Dir. Accuracy (%)', da_ens, '>50%', 'OK' if da_ens > 50 else 'BAJO'),
    ('HMM No Supervisado', 'BIC', bic_optimo, 'Menor', '--'),
    ('HMM No Supervisado', 'Silhouette', sil_score, '>0.25', 'OK' if sil_score > 0.25 else 'BAJO'),
    ('HMM No Supervisado', 'Log-Likelihood', log_lik, 'Mayor', '--'),
]

for m in metricas_resumen:
    print(f"  {m[0]:<25} {m[1]:<20} {m[2]:>12.4f} {m[3]:>12} {m[4]:>12}")


# =============================================================================
# SECCION 19.4: PARAMETROS DEL BACKTEST Y FUNCIONES AUXILIARES
# =============================================================================
# Estos parametros y funciones se guardan en los artefactos para que
# DAX_Analisis_de_Negocio.py pueda ejecutar el backtest sin re-entrenar.
# -----------------------------------------------------------------------
# JUSTIFICACION DE LOS PARAMETROS:
#
# VENTANA_TRAIN_BT = 252 dias (~1 ano bursatil):
#   - Un mercado europeo opera ~252 dias/ano, asi que 252d captura exactamente
#     un ciclo estacional completo (earnings Q1-Q4, vencimientos, dividendos).
#   - Ventanas mas cortas (<126d) producen estimaciones de mu y Sigma muy
#     ruidosas (pocos datos, alta varianza del estimador).
#   - Ventanas mas largas (>504d) diluyen el cambio de regimen: si el mercado
#     paso de BULL a BEAR, 2 anos de datos aun ponderan la fase alcista.
#   - 252d es el estandar de facto en la literatura de gestion cuantitativa
#     (DeMiguel et al. 2009; Ledoit & Wolf, 2004; Kolm et al., 2014).
#
# PASO_REOPT = 22 dias (~1 mes bursatil):
#   - Rebalanceo mensual es el consenso en la industria para carteras long-only
#     de renta variable (Chincarini & Kim, 2006).
#   - Rebalanceos mas frecuentes (semanal, 5d) incrementan costes de
#     transaccion sin mejora proporcional en Sharpe (Kritzman et al., 2010).
#   - Rebalanceos mas espaciados (trimestral, 63d) dejan la cartera expuesta
#     demasiado tiempo a pesos suboptimos tras un cambio brusco de regimen.
#   - 22d balancea reactividad ante cambios de mercado con costes controlados.
#
# COSTE_TRANSACCION = 0.001 (10 bps):
#   - Estimacion conservadora para ETFs/acciones liquidas del DAX40 negociadas
#     en Xetra (spread + comision). La Seccion 22.D de DAX_Analisis_de_Negocio.py
#     evalua escenarios de 5/10/20/50 bps para validar la robustez.

VENTANA_TRAIN_BT  = 252   # 1 ano de datos para estimar mu y Sigma
PASO_REOPT        = 22    # Reoptimizar cada ~1 mes
COSTE_TRANSACCION = 0.001 # 10 bps por trade (ida y vuelta)
MIN_PESO_ML       = 0.02  # 2% minimo por activo (diversificacion forzada)
MAX_TURNOVER      = 0.40  # Maximo 40% de rotacion por rebalanceo
RETRAIN_CADA      = 6     # Re-entrenar GB/MLP cada 6 reoptimizaciones (~6 meses)

# Asignacion de alpha, max_peso y halflife por regimen basada en estadisticas.
# Se ordenan los regimenes de peor (menor retorno medio) a mejor (mayor retorno medio)
# y se interpola linealmente entre los extremos Bear/Bull. Esto funciona para
# cualquier K sin depender de las etiquetas de texto.
#
#   ALPHA:    0.15 (peor regimen, minima confianza en ML) -> 0.60 (mejor regimen)
#   MAX_PESO: 0.20 (mas diversificado en crisis)          -> 0.30 (regimen estable)
#   HALFLIFE: 21d  (alta volatilidad, reacciona rapido)   -> 63d  (baja volatilidad)

ALPHA_BEAR    = 0.15;  ALPHA_BULL    = 0.60
MAX_PESO_BEAR = 0.20;  MAX_PESO_BULL = 0.30
HALFLIFE_BEAR = 21;    HALFLIFE_BULL = 63

_vols_regimen = [regimen_stats[r]['vol_mean'] for r in range(k_optimo)]
_vol_min, _vol_max = min(_vols_regimen), max(_vols_regimen)

sorted_by_ret = sorted(regimen_stats.keys(), key=lambda r: regimen_stats[r]['ret_mean'])

ALPHA_POR_REGIMEN    = {}
MAX_PESO_POR_REGIMEN = {}
HALFLIFE_POR_REGIMEN = {}
for rank, r in enumerate(sorted_by_ret):
    frac_ret = rank / max(k_optimo - 1, 1)                                  # 0.0 peor -> 1.0 mejor retorno
    frac_vol = (_vols_regimen[r] - _vol_min) / (_vol_max - _vol_min + 1e-8) # 0.0 baja vol -> 1.0 alta vol
    ALPHA_POR_REGIMEN[r]    = ALPHA_BEAR    + frac_ret * (ALPHA_BULL    - ALPHA_BEAR)
    MAX_PESO_POR_REGIMEN[r] = MAX_PESO_BEAR + frac_ret * (MAX_PESO_BULL - MAX_PESO_BEAR)
    HALFLIFE_POR_REGIMEN[r] = int(HALFLIFE_BULL - frac_vol * (HALFLIFE_BULL - HALFLIFE_BEAR))  # alta vol -> halflife corto

print(f"\n  [19.4a] Parametros del backtest definidos:")
print(f"    VENTANA_TRAIN_BT={VENTANA_TRAIN_BT}d, PASO_REOPT={PASO_REOPT}d, "
      f"COSTE_TRANSACCION={COSTE_TRANSACCION*100:.1f}bps")
print(f"    MIN_PESO_ML={MIN_PESO_ML*100:.0f}%, MAX_TURNOVER={MAX_TURNOVER*100:.0f}%, "
      f"RETRAIN_CADA={RETRAIN_CADA}")
print(f"    Alpha por regimen: { {regimen_labels[r]: ALPHA_POR_REGIMEN[r] for r in range(k_optimo)} }")


def optimizar_markowitz(mu, sigma, n_act, max_peso=1.0, min_peso=0.0):
    """
    Maximizar Sharpe Ratio via SLSQP.
    mu: retornos esperados anualizados (n_act,)
    sigma: covarianza anualizada (n_act, n_act)
    max_peso: peso maximo por activo
    min_peso: peso minimo por activo (diversificacion forzada)
    Con min_peso>0 se garantiza exposicion minima a todos los activos,
    mas realista para carteras institucionales y reduce riesgo de concentracion.
    """
    if min_peso * n_act > 1.0:
        min_peso = 1.0 / n_act * 0.5

    def neg_sharpe(w):
        ret_p = w @ mu
        vol_p = np.sqrt(np.maximum(w @ sigma @ w, 1e-12))
        return -ret_p / (vol_p + 1e-8)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(min_peso, max_peso)] * n_act
    x0 = np.ones(n_act) / n_act

    result = scipy_minimize(neg_sharpe, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 1000, 'ftol': 1e-10})
    if result.success:
        return result.x
    else:
        return x0


def calcular_metricas_cartera(retornos_diarios):
    """Calcula metricas financieras completas de una serie de retornos diarios."""
    ret = np.array(retornos_diarios)
    ret_anual = np.mean(ret) * 252 * 100
    vol_anual = np.std(ret) * np.sqrt(252) * 100
    sharpe = (np.mean(ret) * 252) / (np.std(ret) * np.sqrt(252) + 1e-8)
    ret_neg = ret[ret < 0]
    if len(ret_neg) > 2:
        downside_vol = np.std(ret_neg) * np.sqrt(252)
        sortino = (np.mean(ret) * 252) / (downside_vol + 1e-8)
    else:
        sortino = sharpe
    ret_cum = np.cumprod(1 + ret)
    running_max = np.maximum.accumulate(ret_cum)
    drawdowns = (ret_cum - running_max) / (running_max + 1e-8)
    max_dd = np.min(drawdowns) * 100
    calmar = (np.mean(ret) * 252) / (abs(max_dd / 100) + 1e-8)
    return {
        'Retorno Anual (%)': ret_anual,
        'Volatilidad Anual (%)': vol_anual,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown (%)': max_dd,
        'Calmar Ratio': calmar,
    }


def predecir_retornos_activos(features_df, ret_wide, model, scaler, tickers, fecha_idx,
                               model2=None, peso_model2=0.0):
    """
    Genera predicciones de retorno para cada activo en una fecha dada.
    Si model2 y peso_model2 > 0 se genera un ensemble ponderado:
        pred = (1 - peso_model2) * model + peso_model2 * model2
    """
    predicciones = {}
    for ticker in tickers:
        if ticker not in features_df or fecha_idx not in features_df[ticker].index:
            predicciones[ticker] = 0.0
            continue
        feats = features_df[ticker].loc[:fecha_idx].iloc[-1:]
        if feats.isna().any(axis=1).iloc[0]:
            predicciones[ticker] = 0.0
            continue
        feats_scaled = scaler.transform(feats.values)
        pred = model.predict(feats_scaled)[0]
        if model2 is not None and peso_model2 > 0.0:
            pred2 = model2.predict(feats_scaled)[0]
            pred = (1.0 - peso_model2) * pred + peso_model2 * pred2
        predicciones[ticker] = pred
    return np.array([predicciones[t] for t in tickers])


def _concat_features_targets_before(features_dict, targets_dict, tickers, max_date):
    """Concatena features/targets de todos los activos hasta max_date (sin look-ahead)."""
    X_list, y_list = [], []
    for ticker in tickers:
        if ticker not in features_dict:
            continue
        feats = features_dict[ticker].copy()
        tgt = targets_dict[ticker].copy()
        df_combined = feats.copy()
        df_combined['_target'] = tgt
        df_combined = df_combined.dropna()
        df_combined = df_combined[df_combined.index <= max_date]
        if len(df_combined) == 0:
            continue
        X_list.append(df_combined.drop(columns='_target').values)
        y_list.append(df_combined['_target'].values)
    if not X_list:
        return None, None
    return np.vstack(X_list), np.concatenate(y_list)


# =============================================================================
# SECCION 19.4b: GUARDAR ARTEFACTOS PARA DAX_Analisis_de_Negocio.py
# =============================================================================
# Guarda todos los modelos entrenados y datos necesarios en /modelos/.
# DAX_Analisis_de_Negocio.py carga estos artefactos y ejecuta las secciones
# 20-23 directamente, sin necesidad de re-entrenar (minutos vs decenas de minutos).

print(f"\n" + "="*70)
print("SECCION 19.4b: GUARDANDO ARTEFACTOS ENTRENADOS")
print("="*70)

_modelos_dir = os.path.join(BASE_DIR, 'modelos')
os.makedirs(_modelos_dir, exist_ok=True)

# Modelos sklearn (joblib es mas eficiente que pickle para arrays numpy)
joblib.dump(model_gb,         os.path.join(_modelos_dir, 'modelo_gb.joblib'))
joblib.dump(model_mlp,        os.path.join(_modelos_dir, 'modelo_mlp.joblib'))
joblib.dump(scaler_features,  os.path.join(_modelos_dir, 'scaler_features.joblib'))
joblib.dump(hmm_model,        os.path.join(_modelos_dir, 'hmm_model.joblib'))
joblib.dump(scaler_regime,    os.path.join(_modelos_dir, 'scaler_regime.joblib'))

# Diccionario con datos, parametros y metricas
_artefactos = {
    # Datos de retornos
    'retornos_full':        retornos_full,
    'retornos_train_wide':  retornos_train_wide,
    'retornos_test_wide':   retornos_test_wide,
    'seleccionadas':        seleccionadas,
    'n_activos':            n_activos,
    'fecha_corte':          fecha_corte,
    # Features para el optimizador
    'features_all':         features_all,
    'targets_all':          targets_all,
    # Salidas HMM
    'regime_features':      regime_features,
    'regimenes':            regimenes,
    'proba_regimenes':      proba_regimenes,
    'k_optimo':             k_optimo,
    'regimen_labels':       regimen_labels,
    'regimen_stats':        regimen_stats,
    'bic_scores':           bic_scores,
    # Benchmark
    'dax40_benchmark':      dax40_benchmark,
    # Pesos del ensemble
    'PESO_GB':              PESO_GB,
    'PESO_MLP':             PESO_MLP,
    'scores_gb_cv':         scores_gb_cv,
    'scores_mlp_cv':        scores_mlp_cv,
    # Metricas seccion 19 (para mostrar en Negocio sin re-evaluar)
    'rmse_gb_test':  rmse_gb_test,  'rmse_mlp_test': rmse_mlp_test,  'rmse_ens_test': rmse_ens_test,
    'mae_gb_test':   mae_gb_test,   'mae_mlp_test':  mae_mlp_test,   'mae_ens_test':  mae_ens_test,
    'r2_gb_test':    r2_gb_test,    'r2_mlp_test':   r2_mlp_test,    'r2_ens_test':   r2_ens_test,
    'da_gb':         da_gb,         'da_mlp':         da_mlp,         'da_ens':         da_ens,
    'sil_score':     sil_score,     'log_lik':        log_lik,
    # Parametros del backtest
    'VENTANA_TRAIN_BT':     VENTANA_TRAIN_BT,
    'PASO_REOPT':           PASO_REOPT,
    'COSTE_TRANSACCION':    COSTE_TRANSACCION,
    'MIN_PESO_ML':          MIN_PESO_ML,
    'MAX_TURNOVER':         MAX_TURNOVER,
    'RETRAIN_CADA':         RETRAIN_CADA,
    'ALPHA_POR_REGIMEN':    ALPHA_POR_REGIMEN,
    'MAX_PESO_POR_REGIMEN': MAX_PESO_POR_REGIMEN,
    'HALFLIFE_POR_REGIMEN': HALFLIFE_POR_REGIMEN,
    # Metadatos
    'TICKER_SECTOR':        TICKER_SECTOR,
}
joblib.dump(_artefactos, os.path.join(_modelos_dir, 'artefactos.joblib'))

print(f"\n  [OK] Modelos guardados en: {_modelos_dir}/")
print(f"       modelo_gb.joblib, modelo_mlp.joblib, scaler_features.joblib")
print(f"       hmm_model.joblib, scaler_regime.joblib, artefactos.joblib")
print(f"\n  Ejecutar DAX_Analisis_de_Negocio.py para las secciones 20-23")
print(f"  (comparacion de estrategias) sin re-entrenar modelos.")


print(f"\n" + "="*70)
print("FIN DEL ANALISIS DEL DATO - SECCIONES 14-19 COMPLETADAS")
print("="*70)
print(f"\n  Visualizaciones generadas:")
print(f"    16_1_heatmap_correlacion_features.png       - Heatmap correlacion de features")
print(f"    17_1_feature_importance_gb.png              - Importancia de features (GB)")
print(f"    17_2a_convergencia_mlp_loss.png             - Curva de loss del MLP")
print(f"    17_2b_cv_folds_rmse.png                    - RMSE por fold de CV temporal")
print(f"    17_3a_permutation_importance_gb.png         - Perm. importance GradientBoosting")
print(f"    17_3b_permutation_importance_mlp.png        - Perm. importance MLP")
print(f"    17_3c_comparacion_importancia_gb_vs_mlp.png - Comparacion GB vs MLP por feature")
print(f"    18_1a_retornos_acumulados_regimenes.png     - Retornos + regimenes HMM")
print(f"    18_1b_probabilidades_regimenes_area.png     - Probabilidades regimen (area)")
print(f"    18_1c_volatilidad_por_regimen.png           - Volatilidad coloreada por regimen")
print(f"    18_2a_seleccion_k_bic.png                  - Seleccion de K optimo (BIC)")
print(f"    18_2b_matriz_transicion_hmm.png             - Matriz de transicion aprendida")
print(f"    19_1a_scatter_real_vs_predicho.png          - Scatter real vs predicho Ensemble")
print(f"    19_1b_distribucion_residuos.png             - Histograma de residuos")
print(f"    19_1c_serie_temporal_residuos.png           - Serie temporal de residuos")
print(f"    19_1d_comparativa_metricas_modelos.png      - Comparativa RMSE/MAE/DA modelos")
print(f"    19_2_predicciones_por_ticker.png            - Predicciones por ticker (grid)")
print(f"    19_3a_qqplot_residuos.png                  - QQ-Plot residuos vs Normal")
print(f"    19_3b_histograma_residuos.png              - Histograma residuos + Normal")
print(f"    19_3c_acf_residuos.png                     - ACF de residuos")
print(f"    19_3d_ljungbox_pvalores.png                - Ljung-Box p-valores por lag")
print(f"    19_4_rolling_rmse.png                      - Rolling RMSE 60d train vs test")
print(f"\n  Continuar con DAX_Analisis_de_Negocio.py para comparacion de estrategias.")

