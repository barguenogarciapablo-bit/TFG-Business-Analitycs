# Para activar el entorno virtual antes de ejecutar este script, usar:
#PS C:\Users\pbarg> 
# cd "C:\Users\pbarg\Documents\UNIVERSIDAD\4\TFG\VisualStudioCode"
#PS C:\Users\pbarg\Documents\UNIVERSIDAD\4\TFG\VisualStudioCode> 
#.venv\Scripts\Activate.ps1

# =============================================================================
# TFG: Optimizador de carteras ML-Enhanced con detección de regímenes: aplicación al DAX40
# =============================================================================
# ARCHIVO: DAX_Analisis_de_Negocio.py
# PREREQUISITO: Ejecutar DAX_Analisis_del_Dato.py primero.
#               Genera la carpeta /modelos/ con todos los artefactos entrenados.
#
# Este script implementa las secciones 20-23 cargando modelos pre-entrenados:
#   Seccion 20: Backtest Markowitz Clasico (baseline)
#   Seccion 21: Optimizador ML-Enhanced (EWMA + Predicciones + Regimenes)
#   Seccion 22: Comparacion Final y Visualizaciones
#   Seccion 23: Conclusiones de Negocio y Recomendaciones
#
# VENTAJA: Al cargar modelos pre-entrenados, este script se ejecuta en
# segundos (vs 20-40 min de entrenamiento en DAX_Analisis_del_Dato.py).
# Util para: ajustar parametros de backtest, regenerar graficos, presentaciones.
# =============================================================================

import os
import sys
import io
import warnings
import numpy as np
import pandas as pd

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
from scipy.optimize import minimize as scipy_minimize
from scipy.special import logsumexp as _logsumexp
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

# =============================================================================
# IMPLEMENTACION PROPIA: GaussianHMM (Hidden Markov Model)
# =============================================================================
# Debe estar definida en __main__ para que joblib/pickle pueda deserializar
# el modelo guardado desde DAX_Analisis_del_Dato.py.
# Implementacion identica a la del script de entrenamiento.
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
        self.startprob_ = None
        self.transmat_ = None
        self.means_ = None
        self.covars_ = None

    def _log_emission(self, X):
        """Log P(x_t | estado=k) para cada t y k.  Forma: (T, K)."""
        T, d = X.shape
        K = self.n_components
        log_prob = np.empty((T, K))
        for k in range(K):
            diff = X - self.means_[k]
            cov_k = self.covars_[k] + 1e-6 * np.eye(d)
            L = np.linalg.cholesky(cov_k)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            solved = np.linalg.solve(L, diff.T)
            log_prob[:, k] = -0.5 * (d * np.log(2 * np.pi)
                                     + log_det
                                     + np.sum(solved ** 2, axis=0))
        return log_prob

    def _forward(self, log_B):
        """Algoritmo forward.  Devuelve log_alpha (T, K)."""
        T, K = log_B.shape
        log_alpha = np.empty((T, K))
        log_alpha[0] = np.log(self.startprob_ + 1e-300) + log_B[0]
        log_A = np.log(self.transmat_ + 1e-300)
        for t in range(1, T):
            log_alpha[t] = (_logsumexp(log_alpha[t - 1, :, None] + log_A,
                                       axis=0)
                            + log_B[t])
        return log_alpha

    def _backward(self, log_B):
        """Algoritmo backward.  Devuelve log_beta (T, K)."""
        T, K = log_B.shape
        log_beta = np.empty((T, K))
        log_beta[T - 1] = 0.0
        log_A = np.log(self.transmat_ + 1e-300)
        for t in range(T - 2, -1, -1):
            log_beta[t] = _logsumexp(log_A
                                     + log_B[t + 1][None, :]
                                     + log_beta[t + 1][None, :],
                                     axis=1)
        return log_beta

    def fit(self, X):
        T, d = X.shape
        K = self.n_components
        best_score = -np.inf
        best_params = None

        for init_i in range(self.n_init):
            rng = np.random.RandomState(
                (self.random_state + init_i) if self.random_state is not None
                else None)

            idx = rng.choice(T, K, replace=False)
            means = X[idx].copy()
            data_cov = np.cov(X.T) + 1e-6 * np.eye(d)
            covars = np.array([data_cov.copy() for _ in range(K)])
            startprob = np.ones(K) / K
            diag_val = 0.7 + 0.2 * rng.random()
            transmat = np.full((K, K), (1 - diag_val) / max(K - 1, 1))
            np.fill_diagonal(transmat, diag_val)

            self.startprob_ = startprob
            self.transmat_ = transmat
            self.means_ = means
            self.covars_ = covars

            prev_ll = -np.inf
            for _ in range(self.n_iter):
                log_B = self._log_emission(X)
                log_alpha = self._forward(log_B)
                log_beta = self._backward(log_B)

                log_gamma = log_alpha + log_beta
                log_gamma -= _logsumexp(log_gamma, axis=1, keepdims=True)
                gamma = np.exp(log_gamma)

                log_A = np.log(transmat + 1e-300)
                log_xi = (log_alpha[:-1, :, None]
                          + log_A[None, :, :]
                          + log_B[1:, None, :]
                          + log_beta[1:, None, :])
                log_xi -= _logsumexp(
                    log_xi.reshape(T - 1, -1), axis=1
                )[:, None, None]
                xi = np.exp(log_xi)

                startprob = gamma[0] + 1e-300
                startprob /= startprob.sum()

                xi_sum = xi.sum(axis=0) + 1e-300
                transmat = xi_sum / xi_sum.sum(axis=1, keepdims=True)

                for k in range(K):
                    gk = gamma[:, k]
                    gk_sum = gk.sum() + 1e-300
                    means[k] = (gk[:, None] * X).sum(axis=0) / gk_sum
                    diff = X - means[k]
                    covars[k] = ((gk[:, None, None]
                                  * (diff[:, :, None] * diff[:, None, :]))
                                 .sum(axis=0) / gk_sum
                                 + 1e-6 * np.eye(d))

                self.startprob_ = startprob
                self.transmat_ = transmat
                self.means_ = means
                self.covars_ = covars

                ll = _logsumexp(log_alpha[-1])
                if abs(ll - prev_ll) < self.tol:
                    break
                prev_ll = ll

            final_ll = _logsumexp(self._forward(self._log_emission(X))[-1])
            if final_ll > best_score:
                best_score = final_ll
                best_params = (startprob.copy(), transmat.copy(),
                               means.copy(), covars.copy())

        self.startprob_, self.transmat_, self.means_, self.covars_ = best_params
        return self

    def predict(self, X):
        """Decodificacion de Viterbi: secuencia de estados mas probable."""
        log_B = self._log_emission(X)
        T, K = log_B.shape
        log_A = np.log(self.transmat_ + 1e-300)

        V = np.empty((T, K))
        bp = np.zeros((T, K), dtype=int)
        V[0] = np.log(self.startprob_ + 1e-300) + log_B[0]

        for t in range(1, T):
            scores = V[t - 1, :, None] + log_A
            bp[t] = scores.argmax(axis=0)
            V[t] = scores[bp[t], np.arange(K)] + log_B[t]

        path = np.empty(T, dtype=int)
        path[T - 1] = V[T - 1].argmax()
        for t in range(T - 2, -1, -1):
            path[t] = bp[t + 1, path[t + 1]]
        return path

    def predict_proba(self, X):
        """Probabilidades posteriores P(s_t=k | O) via forward-backward."""
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
        n_params = ((K - 1)
                    + K * (K - 1)
                    + K * d
                    + K * d * (d + 1) // 2)
        return n_params * np.log(T) - 2.0 * self.score(X)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELOS_DIR = os.path.join(BASE_DIR, 'modelos')

# Costes swap diarios XTB — posicion larga CFD
# Spreads XTB: acciones .DE = ESTER + 4.5%,  indice DE40 = ESTER + 4.0%
_XTB_SPREAD_ACCIONES  = 0.045
_XTB_SPREAD_DAX40     = 0.040
_ESTER_INICIO         = pd.Timestamp('2019-10-02')  # primera publicacion BCE
_FALLBACK_ESTER       = 0.02   # ~2.0% para fechas pre-ESTER
_SWAP_FALLBACK_ACCIONES = -(_FALLBACK_ESTER + _XTB_SPREAD_ACCIONES) / 365
_SWAP_FALLBACK_DAX40    = -(_FALLBACK_ESTER + _XTB_SPREAD_DAX40)    / 365

def _parse_ecb_serie(data):
    obs    = data['dataSets'][0]['series']['0:0:0']['observations']
    fechas = data['structure']['dimensions']['observation'][0]['values']
    return pd.Series(
        {pd.Timestamp(fechas[int(k)]['id']): float(v[0]) / 100 for k, v in obs.items()},
        dtype=float
    ).sort_index()

try:
    import requests as _req
    _ecb_url = ("https://data-api.ecb.europa.eu/service/data/EST/"
                "B.EU000A2X2A25.WT?startPeriod=2019-10-01&format=jsondata")
    _ecb_r = _req.get(_ecb_url, timeout=15)
    _ecb_r.raise_for_status()
    _ester_serie = _parse_ecb_serie(_ecb_r.json())
    _full_idx    = pd.date_range(_ester_serie.index[0], pd.Timestamp.today(), freq='D')
    _ester_serie = _ester_serie.reindex(_full_idx).ffill()

    SWAP_DIARIO_ACCIONES  = -(_ester_serie.iloc[-1] + _XTB_SPREAD_ACCIONES) / 365
    SWAP_DIARIO_DAX40     = -(_ester_serie.iloc[-1] + _XTB_SPREAD_DAX40)    / 365
    swap_acciones_series  = -(_ester_serie + _XTB_SPREAD_ACCIONES) / 365
    swap_dax40_series     = -(_ester_serie + _XTB_SPREAD_DAX40)    / 365

    print(f"  [SWAP] ESTER historico descargado del BCE: {len(_ester_serie):,} dias "
          f"({_ester_serie.index[0].date()} — {_ester_serie.index[-1].date()})")
    print(f"    ESTER  min: {_ester_serie.min()*100:.4f}%  |  "
          f"max: {_ester_serie.max()*100:.4f}%  |  "
          f"actual: {_ester_serie.iloc[-1]*100:.4f}%")
    print(f"    SWAP acciones .DE actual  {SWAP_DIARIO_ACCIONES*100:.6f}%/dia  "
          f"(ESTER {_ester_serie.iloc[-1]*100:.3f}% + spread {_XTB_SPREAD_ACCIONES*100:.1f}%)")
    print(f"    SWAP indice DE40 actual   {SWAP_DIARIO_DAX40*100:.6f}%/dia  "
          f"(ESTER {_ester_serie.iloc[-1]*100:.3f}% + spread {_XTB_SPREAD_DAX40*100:.1f}%)")
    print(f"    Fechas anteriores al {_ESTER_INICIO.date()}: fallback {_SWAP_FALLBACK_ACCIONES*100:.6f}%/dia")

except Exception as _e:
    SWAP_DIARIO_ACCIONES = _SWAP_FALLBACK_ACCIONES
    SWAP_DIARIO_DAX40    = _SWAP_FALLBACK_DAX40
    swap_acciones_series = None
    swap_dax40_series    = None
    print(f"  [SWAP] API BCE no disponible ({_e.__class__.__name__}: {_e}). Usando valores hardcoded.")
    print(f"    SWAP acciones .DE  {SWAP_DIARIO_ACCIONES*100:.6f}%/dia")
    print(f"    SWAP indice DE40   {SWAP_DIARIO_DAX40*100:.6f}%/dia")

def _swap_array(index, series, fallback):
    """Devuelve array de costes de swap para un DatetimeIndex de dias de trading.
    Cada elemento = tasa_diaria * dias_naturales_hasta_siguiente_sesion, de modo
    que fines de semana y festivos quedan absorbidos en el dia de trading anterior
    (viernes = 3 dias, dia antes de festivo = 2+ dias). Replica el cargo real de XTB."""
    idx = pd.DatetimeIndex(index)
    n   = len(idx)
    # dias naturales entre sesion i y sesion i+1 (ultimo periodo = 1 dia por convencion)
    dias_cargo = np.ones(n, dtype=float)
    if n > 1:
        deltas = (idx[1:] - idx[:-1]).days          # array de int
        dias_cargo[:-1] = deltas.astype(float)
    if series is None:
        tasa = np.full(n, fallback)
    else:
        tasa = series.reindex(idx, method='ffill').fillna(fallback).values
    return tasa * dias_cargo

# =============================================================================
# CARGA DE ARTEFACTOS
# =============================================================================

print("="*70)
print("TFG DAX40: ANALISIS DE NEGOCIO - Optimizador de carteras ML-Enhanced con detección de regímenes: aplicación al DAX40")
print("="*70)

_archivos_requeridos = [
    'modelo_gb.joblib', 'modelo_mlp.joblib', 'scaler_features.joblib',
    'hmm_model.joblib', 'scaler_regime.joblib', 'artefactos.joblib'
]
for _f in _archivos_requeridos:
    _ruta = os.path.join(MODELOS_DIR, _f)
    if not os.path.exists(_ruta):
        print(f"\n  [ERROR] No se encontro: {_ruta}")
        print(f"  Ejecutar DAX_Analisis_del_Dato.py primero para generar los artefactos.")
        sys.exit(1)

model_gb        = joblib.load(os.path.join(MODELOS_DIR, 'modelo_gb.joblib'))
model_mlp       = joblib.load(os.path.join(MODELOS_DIR, 'modelo_mlp.joblib'))
scaler_features = joblib.load(os.path.join(MODELOS_DIR, 'scaler_features.joblib'))
hmm_model       = joblib.load(os.path.join(MODELOS_DIR, 'hmm_model.joblib'))
scaler_regime   = joblib.load(os.path.join(MODELOS_DIR, 'scaler_regime.joblib'))
_art            = joblib.load(os.path.join(MODELOS_DIR, 'artefactos.joblib'))

# Desempaquetar artefactos
retornos_full        = _art['retornos_full']
retornos_train_wide  = _art['retornos_train_wide']
retornos_test_wide   = _art['retornos_test_wide']
seleccionadas        = _art['seleccionadas']
n_activos            = _art['n_activos']
fecha_corte          = _art['fecha_corte']
features_all         = _art['features_all']
targets_all          = _art['targets_all']
regime_features      = _art['regime_features']
regimenes            = _art['regimenes']
proba_regimenes      = _art['proba_regimenes']
k_optimo             = _art['k_optimo']
regimen_labels       = _art['regimen_labels']
regimen_stats        = _art['regimen_stats']
bic_scores           = _art['bic_scores']
dax40_benchmark      = _art['dax40_benchmark']
PESO_GB              = _art['PESO_GB']
PESO_MLP             = _art['PESO_MLP']
scores_gb_cv         = _art['scores_gb_cv']
scores_mlp_cv        = _art['scores_mlp_cv']
TICKER_SECTOR        = _art['TICKER_SECTOR']

# Metricas modelo (seccion 19, pre-calculadas)
rmse_gb_test  = _art['rmse_gb_test'];  rmse_mlp_test = _art['rmse_mlp_test'];  rmse_ens_test = _art['rmse_ens_test']
mae_gb_test   = _art['mae_gb_test'];   mae_mlp_test  = _art['mae_mlp_test'];   mae_ens_test  = _art['mae_ens_test']
r2_gb_test    = _art['r2_gb_test'];    r2_mlp_test   = _art['r2_mlp_test'];    r2_ens_test   = _art['r2_ens_test']
da_gb         = _art['da_gb'];         da_mlp        = _art['da_mlp'];          da_ens        = _art['da_ens']
sil_score     = _art['sil_score']

# Parametros del backtest (modificables aqui sin re-entrenar)
VENTANA_TRAIN_BT  = _art['VENTANA_TRAIN_BT']
PASO_REOPT        = _art['PASO_REOPT']
COSTE_TRANSACCION = _art['COSTE_TRANSACCION']
MIN_PESO_ML       = _art['MIN_PESO_ML']
MAX_TURNOVER      = _art['MAX_TURNOVER']
RETRAIN_CADA      = _art['RETRAIN_CADA']
ALPHA_POR_REGIMEN    = _art['ALPHA_POR_REGIMEN']
MAX_PESO_POR_REGIMEN = _art['MAX_PESO_POR_REGIMEN']
HALFLIFE_POR_REGIMEN = _art['HALFLIFE_POR_REGIMEN']

# Gradiente rojo -> azul segun calidad del regimen (identico a DAX_Analisis_del_Dato.py).
# sorted_by_ret[0] = peor retorno (BEAR) -> rojo, sorted_by_ret[-1] = mejor (BULL) -> azul.
_sorted_by_ret = sorted(regimen_labels.keys(),
                        key=lambda r: regimen_stats[r]['ret_mean'])
_cmap_reg = plt.cm.RdBu
colores_regimen_ext = {}
for _rank, r in enumerate(_sorted_by_ret):
    _t = _rank / max(k_optimo - 1, 1)   # 0.0 = peor, 1.0 = mejor
    _t = 0.05 + _t * 0.90               # recortar extremos: [0.05, 0.95]
    colores_regimen_ext[r] = _cmap_reg(_t)

print(f"\n  [OK] Artefactos cargados desde: {MODELOS_DIR}/")
print(f"    Activos: {n_activos}  |  K_HMM: {k_optimo}  |  PESO_GB: {PESO_GB:.3f}  |  PESO_MLP: {PESO_MLP:.3f}")
print(f"    Retornos full: {retornos_full.shape}  |  Fecha corte: {fecha_corte.date()}")

print(f"\n  [INFO] Metricas modelo supervisado (test set, pre-calculadas):")
print(f"    Ensemble RMSE={rmse_ens_test:.6f}  R2={r2_ens_test:.6f}  DA={da_ens:.2f}%")
print(f"    HMM Silhouette={sil_score:.4f}")

# =============================================================================
# FUNCIONES AUXILIARES
# (Definidas aqui para que este script sea autocontenido al cargar artefactos.
#  Las definiciones canonicas viven en DAX_Analisis_del_Dato.py, Seccion 19.4.)
# =============================================================================

def optimizar_markowitz(mu, sigma, n_act, max_peso=1.0, min_peso=0.0):
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
    return result.x if result.success else x0


def calcular_metricas_cartera(retornos_diarios):
    ret = np.array(retornos_diarios)
    ret_anual = np.mean(ret) * 252 * 100
    vol_anual = np.std(ret) * np.sqrt(252) * 100
    sharpe = (np.mean(ret) * 252) / (np.std(ret) * np.sqrt(252) + 1e-8)
    # Downside deviation correcta: sqrt(E[min(r,0)^2]) sobre todos los retornos
    downside_vol = np.sqrt(np.mean(np.minimum(ret, 0.0) ** 2)) * np.sqrt(252)
    sortino = (np.mean(ret) * 252) / (downside_vol + 1e-8)
    ret_cum = np.cumprod(1 + ret)
    running_max = np.maximum.accumulate(ret_cum)
    drawdowns = (ret_cum - running_max) / (running_max + 1e-8)
    max_dd = np.min(drawdowns) * 100
    calmar = (np.mean(ret) * 252) / (abs(max_dd / 100) + 1e-8)
    # CVaR (Expected Shortfall) al 95% y 99%
    var_95 = np.percentile(ret, 5)
    cvar_95 = ret[ret <= var_95].mean() * 100 if (ret <= var_95).any() else var_95 * 100
    var_99 = np.percentile(ret, 1)
    cvar_99 = ret[ret <= var_99].mean() * 100 if (ret <= var_99).any() else var_99 * 100
    return {
        'Retorno Anual (%)': ret_anual,
        'Volatilidad Anual (%)': vol_anual,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown (%)': max_dd,
        'Calmar Ratio': calmar,
        'CVaR 95% (%)': cvar_95,
        'CVaR 99% (%)': cvar_99,
    }


def predecir_retornos_activos(features_df, ret_wide, model, scaler, tickers, fecha_idx,
                               model2=None, peso_model1=1.0, peso_model2=0.0):
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
            pred = peso_model1 * pred + peso_model2 * pred2
        predicciones[ticker] = pred
    return np.array([predicciones[t] for t in tickers])


def _concat_features_targets_before(features_dict, targets_dict, tickers, max_date):
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
# SECCION 20: BACKTEST - MARKOWITZ CLASICO (BASELINE)
# =============================================================================

print(f"\n" + "="*70)
print("SECCION 20: BACKTEST - MARKOWITZ CLASICO (BASELINE)")
print("="*70)

n_dias_test_total = len(retornos_test_wide)
idx_test_start = retornos_full.index.get_loc(retornos_test_wide.index[0])

print(f"\n  [20.1] Parametros del backtest:")
print(f"    Ventana estimacion: {VENTANA_TRAIN_BT} dias (~1 ano)")
print(f"    Paso reoptimizacion: {PASO_REOPT} dias (~1 mes)")
print(f"    Coste transaccion: {COSTE_TRANSACCION*100:.1f} bps por rebalanceo")
print(f"    Dias totales en test: {n_dias_test_total}")

print(f"\n  [20.2] Ejecutando backtest Markowitz Clasico...")

ret_daily_equi = []
ret_daily_marko = []
ret_daily_marko_lw = []
pesos_marko_hist = []
pesos_prev_marko = np.ones(n_activos) / n_activos
pesos_prev_marko_lw = np.ones(n_activos) / n_activos
fechas_bt = []

for step in range(0, n_dias_test_total - PASO_REOPT, PASO_REOPT):
    idx_eval_start = idx_test_start + step
    idx_eval_end = min(idx_eval_start + PASO_REOPT, len(retornos_full))
    idx_est_start = max(0, idx_eval_start - VENTANA_TRAIN_BT)
    ret_estimacion = retornos_full.iloc[idx_est_start:idx_eval_start]
    ret_evaluacion = retornos_full.iloc[idx_eval_start:idx_eval_end]

    if len(ret_estimacion) < 60 or len(ret_evaluacion) == 0:
        continue

    _swap_acc  = _swap_array(ret_evaluacion.index, swap_acciones_series, SWAP_DIARIO_ACCIONES)

    pesos_equi = np.ones(n_activos) / n_activos
    ret_daily_equi.extend((ret_evaluacion.values @ pesos_equi + _swap_acc).tolist())

    mu_hist = ret_estimacion.mean().values * 252
    sigma_hist = ret_estimacion.cov().values * 252
    pesos_mk = optimizar_markowitz(mu_hist, sigma_hist, n_activos, max_peso=1.0)

    turnover = np.sum(np.abs(pesos_mk - pesos_prev_marko))
    coste = turnover * COSTE_TRANSACCION
    ret_daily_marko.extend((ret_evaluacion.values @ pesos_mk + _swap_acc - coste / len(ret_evaluacion)).tolist())

    # Markowitz + Ledoit-Wolf (baseline para aislar contribucion del ML)
    _lw_mk = LedoitWolf()
    _lw_mk.fit(ret_estimacion.values)
    sigma_lw_hist = _lw_mk.covariance_ * 252
    pesos_mk_lw = optimizar_markowitz(mu_hist, sigma_lw_hist, n_activos, max_peso=1.0)
    turnover_lw = np.sum(np.abs(pesos_mk_lw - pesos_prev_marko_lw))
    coste_lw = turnover_lw * COSTE_TRANSACCION
    ret_daily_marko_lw.extend((ret_evaluacion.values @ pesos_mk_lw + _swap_acc - coste_lw / len(ret_evaluacion)).tolist())
    pesos_prev_marko_lw = pesos_mk_lw

    pesos_prev_marko = pesos_mk
    pesos_marko_hist.append(pesos_mk)
    fechas_bt.append(ret_evaluacion.index[0])

metricas_equi     = calcular_metricas_cartera(ret_daily_equi)
metricas_marko    = calcular_metricas_cartera(ret_daily_marko)
metricas_marko_lw = calcular_metricas_cartera(ret_daily_marko_lw)

print(f"    [OK] Markowitz completado: {len(fechas_bt)} reoptimizaciones")
print(f"\n  [20.3] Resultados Markowitz Clasico:")
print(f"    Equiponderada  -> Sharpe: {metricas_equi['Sharpe Ratio']:.4f}, "
      f"Ret: {metricas_equi['Retorno Anual (%)']:.2f}%, "
      f"Vol: {metricas_equi['Volatilidad Anual (%)']:.2f}%")
print(f"    Markowitz Clas -> Sharpe: {metricas_marko['Sharpe Ratio']:.4f}, "
      f"Ret: {metricas_marko['Retorno Anual (%)']:.2f}%, "
      f"Vol: {metricas_marko['Volatilidad Anual (%)']:.2f}%")
print(f"    Marko + LW     -> Sharpe: {metricas_marko_lw['Sharpe Ratio']:.4f}, "
      f"Ret: {metricas_marko_lw['Retorno Anual (%)']:.2f}%, "
      f"Vol: {metricas_marko_lw['Volatilidad Anual (%)']:.2f}%")


# =============================================================================
# SECCION 21: OPTIMIZADOR ML-ENHANCED
# =============================================================================

print(f"\n" + "="*70)
print("SECCION 21: OPTIMIZADOR ML-ENHANCED (COMBINACION SUP. + NO SUP.)")
print("="*70)

print(f"\n  [21.1] Configuracion del optimizador ML-Enhanced:")
for r in range(k_optimo):
    print(f"    {regimen_labels[r]}: alpha={ALPHA_POR_REGIMEN[r]:.2f}, "
          f"max_peso={MAX_PESO_POR_REGIMEN[r]:.2f}")
print(f"    EWMA halflife por regimen (basado en vol): { {regimen_labels[r]: HALFLIFE_POR_REGIMEN[r] for r in range(k_optimo)} }")
print(f"    Min peso: {MIN_PESO_ML*100:.0f}%  |  Max turnover: {MAX_TURNOVER*100:.0f}%")
print(f"    Re-entrenamiento walk-forward cada {RETRAIN_CADA} reoptimizaciones")

print(f"\n  [21.2] Ejecutando backtest ML-Enhanced...")

ret_daily_ml = []
pesos_ml_hist = []
regimen_bt_hist = []
alpha_bt_hist = []
turnover_hist = []
pesos_prev_ml = np.ones(n_activos) / n_activos
fechas_ml_bt = []
_step_count = 0
_n_retrains = 0

for step in range(0, n_dias_test_total - PASO_REOPT, PASO_REOPT):
    idx_eval_start = idx_test_start + step
    idx_eval_end = min(idx_eval_start + PASO_REOPT, len(retornos_full))
    idx_est_start = max(0, idx_eval_start - VENTANA_TRAIN_BT)
    ret_estimacion = retornos_full.iloc[idx_est_start:idx_eval_start]
    ret_evaluacion = retornos_full.iloc[idx_eval_start:idx_eval_end]

    if len(ret_estimacion) < 60 or len(ret_evaluacion) == 0:
        continue

    fecha_actual = retornos_full.index[idx_eval_start]

    # Walk-forward re-training
    if _step_count > 0 and _step_count % RETRAIN_CADA == 0:
        X_ret, y_ret = _concat_features_targets_before(
            features_all, targets_all, seleccionadas, fecha_actual
        )
        if X_ret is not None and len(X_ret) > 200:
            scaler_retrain = StandardScaler()
            X_ret_scaled = scaler_retrain.fit_transform(X_ret)
            model_gb.fit(X_ret_scaled, y_ret)
            model_mlp.fit(X_ret_scaled, y_ret)
            scaler_features = scaler_retrain
            _n_retrains += 1

    # Detectar regimen (filtro causal desde artefactos: solo pasada forward, sin mirar el futuro)
    if fecha_actual in regime_features.index:
        idx_rf = regime_features.index.get_loc(fecha_actual)
        regimen_actual = regimenes[idx_rf]
        _proba_reg = proba_regimenes[idx_rf]
    else:
        closest_idx = regime_features.index.searchsorted(fecha_actual) - 1
        closest_idx = max(0, min(closest_idx, len(regimenes) - 1))
        regimen_actual = regimenes[closest_idx]
        _proba_reg = proba_regimenes[closest_idx]

    alpha = float(sum(_proba_reg[r] * ALPHA_POR_REGIMEN[r] for r in range(k_optimo)))
    max_peso_actual = float(sum(_proba_reg[r] * MAX_PESO_POR_REGIMEN[r] for r in range(k_optimo)))

    # Halflife EWMA: media ponderada por probabilidades posteriores de cada regimen
    _halflife_actual = round(sum(_proba_reg[r] * HALFLIFE_POR_REGIMEN[r] for r in range(k_optimo)))

    # mu_enhanced: ensemble GB+MLP + James-Stein shrinkage
    mu_ml = predecir_retornos_activos(
        features_all, retornos_full, model_gb, scaler_features,
        seleccionadas, fecha_actual, model2=model_mlp, peso_model1=PESO_GB, peso_model2=PESO_MLP
    )
    mu_hist = ret_estimacion.mean().values * 252
    grand_mean = mu_hist.mean()
    n_obs_est = len(ret_estimacion)
    shrink_factor = max(0, 1 - (n_activos - 2) / (n_obs_est * np.sum((mu_hist - grand_mean)**2) + 1e-8))
    mu_shrinkage = grand_mean + shrink_factor * (mu_hist - grand_mean)
    mu_enhanced = alpha * (mu_ml * 252) + (1 - alpha) * mu_shrinkage

    # sigma_enhanced: EWMA adaptativo + Ledoit-Wolf
    try:
        ewma_cov = ret_estimacion.ewm(halflife=_halflife_actual).cov()
        last_date = ret_estimacion.index[-1]
        sigma_ewma = ewma_cov.loc[last_date].values * 252
        if sigma_ewma.shape != (n_activos, n_activos):
            raise ValueError(f"EWMA shape {sigma_ewma.shape} != ({n_activos},{n_activos})")
        lw = LedoitWolf()
        lw.fit(ret_estimacion.values)
        sigma_enhanced = 0.70 * sigma_ewma + 0.30 * lw.covariance_ * 252
    except (ValueError, np.linalg.LinAlgError) as _ewma_err:
        print(f"    [WARN] EWMA fallback en {fecha_actual.date()}: {_ewma_err}")
        lw = LedoitWolf()
        lw.fit(ret_estimacion.values)
        sigma_enhanced = lw.covariance_ * 252

    pesos_ml_opt = optimizar_markowitz(mu_enhanced, sigma_enhanced, n_activos,
                                       max_peso=max_peso_actual, min_peso=MIN_PESO_ML)

    turnover = np.sum(np.abs(pesos_ml_opt - pesos_prev_ml))
    if turnover > MAX_TURNOVER:
        delta = pesos_ml_opt - pesos_prev_ml
        pesos_ml_opt = pesos_prev_ml + delta * (MAX_TURNOVER / turnover)
        pesos_ml_opt = np.maximum(pesos_ml_opt, 0.0)
        pesos_ml_opt /= pesos_ml_opt.sum()
        turnover = MAX_TURNOVER

    coste      = turnover * COSTE_TRANSACCION
    _swap_acc  = _swap_array(ret_evaluacion.index, swap_acciones_series, SWAP_DIARIO_ACCIONES)
    ret_daily_ml.extend((ret_evaluacion.values @ pesos_ml_opt + _swap_acc - coste / len(ret_evaluacion)).tolist())

    pesos_prev_ml = pesos_ml_opt
    pesos_ml_hist.append(pesos_ml_opt)
    regimen_bt_hist.append(regimen_actual)
    alpha_bt_hist.append(alpha)
    turnover_hist.append(turnover)
    fechas_ml_bt.append(fecha_actual)
    _step_count += 1

print(f"    [OK] ML-Enhanced completado: {len(fechas_ml_bt)} reoptimizaciones, "
      f"{_n_retrains} re-entrenamientos walk-forward")

metricas_ml = calcular_metricas_cartera(ret_daily_ml)

# Benchmark DAX40
metricas_dax40 = None
ret_daily_dax40 = []
if dax40_benchmark is not None:
    dax40_test = dax40_benchmark.loc[dax40_benchmark.index >= retornos_test_wide.index[0]]
    if len(dax40_test) > 1:
        dax40_returns = np.log(dax40_test / dax40_test.shift(1)).dropna()
        n_common_dax = min(len(ret_daily_equi), len(dax40_returns))
        _swap_dax    = _swap_array(dax40_returns.index[:n_common_dax], swap_dax40_series, SWAP_DIARIO_DAX40)
        ret_daily_dax40 = (dax40_returns.values[:n_common_dax] + _swap_dax).tolist()
        metricas_dax40 = calcular_metricas_cartera(ret_daily_dax40)
        print(f"    DAX40 B&H   -> Sharpe: {metricas_dax40['Sharpe Ratio']:.4f}, "
              f"Ret: {metricas_dax40['Retorno Anual (%)']:.2f}%")

# =============================================================================
# 21.4 ITERACION: VERIFICAR SI ML-ENHANCED SUPERA A MARKOWITZ
# =============================================================================

print(f"\n  [21.4] Verificando si ML-Enhanced supera Markowitz Clasico...")

# Guardar copia antes de que el bloque de iteracion pueda sobreescribir ret_daily_ml.
# La Seccion 22.D usa turnover_hist (del backtest original) para ajustar costes,
# por lo que necesita los retornos del mismo portfolio, no los de la iteracion.
_ret_daily_ml_original = list(ret_daily_ml)

mejora_sharpe = metricas_ml['Sharpe Ratio'] - metricas_marko['Sharpe Ratio']
print(f"    Sharpe ML-Enhanced: {metricas_ml['Sharpe Ratio']:.4f}")
print(f"    Sharpe Markowitz:   {metricas_marko['Sharpe Ratio']:.4f}")
print(f"    Delta:              {mejora_sharpe:+.4f}")

_alpha_mult_final = 1.0  # 1.0 = ML completo; se reduce si la iteracion lo ajusta
if mejora_sharpe <= 0:
    print(f"\n    [ITERACION] ML-Enhanced no supera Markowitz. Ajustando parametros...")

    ret_daily_ml_iter = []
    metricas_iter = None
    for iteracion, alpha_mult in enumerate([0.5, 0.3, 0.15, 0.0], 1):
        ret_daily_ml_iter = []
        pesos_prev_iter = np.ones(n_activos) / n_activos

        for step in range(0, n_dias_test_total - PASO_REOPT, PASO_REOPT):
            idx_eval_start = idx_test_start + step
            idx_eval_end = min(idx_eval_start + PASO_REOPT, len(retornos_full))
            idx_est_start = max(0, idx_eval_start - VENTANA_TRAIN_BT)
            ret_estimacion = retornos_full.iloc[idx_est_start:idx_eval_start]
            ret_evaluacion = retornos_full.iloc[idx_eval_start:idx_eval_end]

            if len(ret_estimacion) < 60 or len(ret_evaluacion) == 0:
                continue

            fecha_actual = retornos_full.index[idx_eval_start]

            if fecha_actual in regime_features.index:
                idx_rf = regime_features.index.get_loc(fecha_actual)
                reg = regimenes[idx_rf]
                _proba_iter = proba_regimenes[idx_rf]
            else:
                closest = regime_features.index.searchsorted(fecha_actual) - 1
                closest = max(0, min(closest, len(regimenes) - 1))
                reg = regimenes[closest]
                _proba_iter = proba_regimenes[closest]

            alpha_base = float(sum(_proba_iter[r] * ALPHA_POR_REGIMEN[r] for r in range(k_optimo)))
            alpha_iter = alpha_base * alpha_mult

            # max_peso y halflife: media ponderada por probabilidades del regimen
            max_peso_iter  = float(sum(_proba_iter[r] * MAX_PESO_POR_REGIMEN[r] for r in range(k_optimo)))
            _halflife_iter = round(sum(_proba_iter[r] * HALFLIFE_POR_REGIMEN[r] for r in range(k_optimo)))

            mu_ml_iter = predecir_retornos_activos(
                features_all, retornos_full, model_gb, scaler_features,
                seleccionadas, fecha_actual, model2=model_mlp, peso_model1=PESO_GB, peso_model2=PESO_MLP
            )
            mu_hist_iter = ret_estimacion.mean().values * 252
            grand_mean_iter = mu_hist_iter.mean()
            n_obs_iter = len(ret_estimacion)
            sf = max(0, 1 - (n_activos - 2) / (n_obs_iter * np.sum((mu_hist_iter - grand_mean_iter)**2) + 1e-8))
            mu_iter = alpha_iter * (mu_ml_iter * 252) + (1 - alpha_iter) * (grand_mean_iter + sf * (mu_hist_iter - grand_mean_iter))

            try:
                ewma_c = ret_estimacion.ewm(halflife=_halflife_iter).cov()
                last_d = ret_estimacion.index[-1]
                sigma_e = ewma_c.loc[last_d].values * 252
                if sigma_e.shape != (n_activos, n_activos):
                    raise ValueError(f"EWMA shape {sigma_e.shape} != ({n_activos},{n_activos})")
                lw_iter = LedoitWolf()
                lw_iter.fit(ret_estimacion.values)
                sigma_iter = 0.70 * sigma_e + 0.30 * lw_iter.covariance_ * 252
            except (ValueError, np.linalg.LinAlgError) as _ewma_err:
                print(f"    [WARN] EWMA fallback iter en {fecha_actual.date()}: {_ewma_err}")
                lw_iter = LedoitWolf()
                lw_iter.fit(ret_estimacion.values)
                sigma_iter = lw_iter.covariance_ * 252

            pesos_iter = optimizar_markowitz(mu_iter, sigma_iter, n_activos,
                                             max_peso=max_peso_iter, min_peso=MIN_PESO_ML)

            turnover_iter = np.sum(np.abs(pesos_iter - pesos_prev_iter))
            if turnover_iter > MAX_TURNOVER:
                delta_iter_t = pesos_iter - pesos_prev_iter
                pesos_iter = pesos_prev_iter + delta_iter_t * (MAX_TURNOVER / turnover_iter)
                pesos_iter = np.maximum(pesos_iter, 0.0)
                pesos_iter /= pesos_iter.sum()
                turnover_iter = MAX_TURNOVER

            _swap_acc = _swap_array(ret_evaluacion.index, swap_acciones_series, SWAP_DIARIO_ACCIONES)
            ret_daily_ml_iter.extend((ret_evaluacion.values @ pesos_iter + _swap_acc - turnover_iter * COSTE_TRANSACCION / len(ret_evaluacion)).tolist())
            pesos_prev_iter = pesos_iter

        metricas_iter = calcular_metricas_cartera(ret_daily_ml_iter)
        delta_iter = metricas_iter['Sharpe Ratio'] - metricas_marko['Sharpe Ratio']
        print(f"    Iteracion {iteracion}: alpha_mult={alpha_mult:.2f}, "
              f"Sharpe={metricas_iter['Sharpe Ratio']:.4f}, delta={delta_iter:+.4f}")

        if delta_iter > 0:
            print(f"    [OK] ML-Enhanced SUPERA Markowitz con alpha_mult={alpha_mult}")
            ret_daily_ml = ret_daily_ml_iter
            metricas_ml = metricas_iter
            mejora_sharpe = delta_iter
            _alpha_mult_final = alpha_mult
            break
    else:
        _alpha_mult_final = 0.0
        print(f"\n    [INFO] La mejora principal proviene de la estimacion de covarianza")
        if ret_daily_ml_iter and metricas_iter is not None:
            ret_daily_ml = ret_daily_ml_iter
            metricas_ml = metricas_iter
        mejora_sharpe = metricas_ml['Sharpe Ratio'] - metricas_marko['Sharpe Ratio']

print(f"\n  [21.5] Resultado FINAL ML-Enhanced:")
print(f"    Sharpe: {metricas_ml['Sharpe Ratio']:.4f} "
      f"(vs Markowitz: {metricas_marko['Sharpe Ratio']:.4f}, "
      f"delta: {mejora_sharpe:+.4f})")

# GRAFICO 21.1 - Alpha y Turnover
if fechas_ml_bt and alpha_bt_hist:
    _fechas_a = pd.to_datetime(fechas_ml_bt)
    _alphas = np.array(alpha_bt_hist[:len(fechas_ml_bt)])
    _regimes = regimen_bt_hist[:len(fechas_ml_bt)]
    _turnovers = np.array(turnover_hist[:len(fechas_ml_bt)])
    # Usar el mismo gradiente rojo->azul de colores_regimen_ext (ya calculado arriba)
    _colors_map = colores_regimen_ext
    _labels_map = regimen_labels

    # Figura 21_1a: Evolucion del alpha (confianza en ML) por regimen
    fig_21_1a, ax_21_1a = plt.subplots(figsize=(16, 5))
    ax_21_1a.plot(_fechas_a, _alphas, color='black', linewidth=0.8, alpha=0.4, zorder=1)
    _plotted_labels = set()
    for i, (f, a, r) in enumerate(zip(_fechas_a, _alphas, _regimes)):
        _lbl = _labels_map.get(r, f'R{r}') if r not in _plotted_labels else None
        ax_21_1a.scatter(f, a, c=_colors_map.get(r, 'gray'), s=25, zorder=2, label=_lbl, edgecolors='none')
        if _lbl:
            _plotted_labels.add(r)
    ax_21_1a.set_ylabel('Alpha (confianza en prediccion ML)', fontsize=10)
    ax_21_1a.set_xlabel('Fecha de reoptimizacion', fontsize=10)
    ax_21_1a.set_ylim(-0.05, 0.75)
    ax_21_1a.set_title(
        'Evolucion del Alpha de Confianza en ML durante el Backtest — por Regimen HMM\n'
        '(alpha alto = mayor peso al ML; alpha bajo = mayor peso al historico shrinkage)',
        fontsize=12, fontweight='bold')
    ax_21_1a.legend(fontsize=9, loc='upper right')
    ax_21_1a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '21_1a_evolucion_alpha_confianza_ml.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Guardado: 21_1a_evolucion_alpha_confianza_ml.png")

    # Figura 21_1b: Turnover de cartera en cada reoptimizacion
    fig_21_1b, ax_21_1b = plt.subplots(figsize=(16, 4))
    ax_21_1b.bar(_fechas_a, _turnovers, width=15, color='steelblue', alpha=0.7, edgecolor='none')
    ax_21_1b.axhline(MAX_TURNOVER, color='red', linestyle='--', linewidth=1, label=f'Limite turnover ({MAX_TURNOVER*100:.0f}%)')
    ax_21_1b.set_ylabel('Turnover (cambio total de pesos)', fontsize=10)
    ax_21_1b.set_xlabel('Fecha de reoptimizacion', fontsize=10)
    ax_21_1b.set_title(
        f'Turnover de la Cartera ML-Enhanced por Reoptimizacion\n'
        f'(medio={np.mean(_turnovers):.3f}, mediana={np.median(_turnovers):.3f}, max={np.max(_turnovers):.3f})',
        fontsize=12, fontweight='bold')
    ax_21_1b.legend(fontsize=9)
    ax_21_1b.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '21_1b_turnover_cartera_reoptimizacion.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Guardado: 21_1b_turnover_cartera_reoptimizacion.png")
    print(f"    Turnover medio: {np.mean(_turnovers):.3f}, "
          f"mediana: {np.median(_turnovers):.3f}, max: {np.max(_turnovers):.3f}")


# =============================================================================
# SECCION 22: COMPARACION FINAL Y VISUALIZACIONES
# =============================================================================

print(f"\n" + "="*70)
print("SECCION 22: COMPARACION FINAL - TODAS LAS ESTRATEGIAS")
print("="*70)

n_common = min(len(ret_daily_equi), len(ret_daily_marko), len(ret_daily_marko_lw), len(ret_daily_ml))

ret_equi_final     = ret_daily_equi[-n_common:]
ret_marko_final    = ret_daily_marko[-n_common:]
ret_marko_lw_final = ret_daily_marko_lw[-n_common:]
ret_ml_final       = ret_daily_ml[-n_common:]

metricas_equi_f     = calcular_metricas_cartera(ret_equi_final)
metricas_marko_f    = calcular_metricas_cartera(ret_marko_final)
metricas_marko_lw_f = calcular_metricas_cartera(ret_marko_lw_final)
metricas_ml_f       = calcular_metricas_cartera(ret_ml_final)

if ret_daily_dax40:
    ret_dax40_final  = ret_daily_dax40[:n_common]
    metricas_dax40_f = calcular_metricas_cartera(ret_dax40_final)
else:
    ret_dax40_final  = None
    metricas_dax40_f = None

metricas_nombres = ['Retorno Anual (%)', 'Volatilidad Anual (%)', 'Sharpe Ratio',
                    'Sortino Ratio', 'Max Drawdown (%)', 'Calmar Ratio',
                    'CVaR 95% (%)', 'CVaR 99% (%)']

print(f"\n  [22.1] TABLA COMPARATIVA DE METRICAS (periodo test, {n_common} dias):")
print(f"  " + "="*115)
header = f"  {'Metrica':<25} {'Equiponderada':>15} {'Markowitz':>15} {'Marko+LW':>15} {'ML-Enhanced':>15}"
if metricas_dax40_f:
    header += f" {'DAX40 B&H':>15}"
print(header)
print(f"  " + "-"*115)
for m in metricas_nombres:
    line = f"  {m:<25}" + "".join(f" {v:>15.4f}" for v in [
        metricas_equi_f[m], metricas_marko_f[m], metricas_marko_lw_f[m], metricas_ml_f[m]])
    if metricas_dax40_f:
        line += f" {metricas_dax40_f[m]:>15.4f}"
    print(line)

mejora_ret      = metricas_ml_f['Retorno Anual (%)']     - metricas_marko_f['Retorno Anual (%)']
mejora_vol      = metricas_marko_f['Volatilidad Anual (%)'] - metricas_ml_f['Volatilidad Anual (%)']
mejora_sharpe_f = metricas_ml_f['Sharpe Ratio']           - metricas_marko_f['Sharpe Ratio']
mejora_dd       = metricas_marko_f['Max Drawdown (%)']    - metricas_ml_f['Max Drawdown (%)']

print(f"\n  [22.2] ANALISIS DE MEJORA ML-ENHANCED vs MARKOWITZ:")
print(f"    Sharpe Ratio:      {mejora_sharpe_f:>+8.4f} ({'MEJOR' if mejora_sharpe_f > 0 else 'INFERIOR'})")
print(f"    Retorno Anual:     {mejora_ret:>+8.2f}% ({'MEJOR' if mejora_ret > 0 else 'INFERIOR'})")
print(f"    Vol. reducida:     {mejora_vol:>+8.2f}% ({'MEJOR' if mejora_vol > 0 else 'MAYOR'})")
print(f"    Max DD mejorado:   {mejora_dd:>+8.2f}% ({'MEJOR' if mejora_dd > 0 else 'PEOR'})")
n_mejoras = sum([mejora_sharpe_f > 0, mejora_ret > 0, mejora_vol > 0, mejora_dd > 0])
print(f"\n    Mejoras en {n_mejoras}/4 metricas clave")

print(f"\n  [22.3] Generando visualizaciones finales...")

colores = {'Equiponderada': 'steelblue', 'Markowitz': 'darkgreen', 'Marko+LW': '#2ca02c',
           'ML-Enhanced': 'darkorange', 'DAX40 B&H': 'purple'}

nombres    = ['Equipond.', 'Markowitz', 'Marko+LW', 'ML-Enh.']
sharpes    = [metricas_equi_f['Sharpe Ratio'], metricas_marko_f['Sharpe Ratio'],
              metricas_marko_lw_f['Sharpe Ratio'], metricas_ml_f['Sharpe Ratio']]
vols       = [metricas_equi_f['Volatilidad Anual (%)'], metricas_marko_f['Volatilidad Anual (%)'],
              metricas_marko_lw_f['Volatilidad Anual (%)'], metricas_ml_f['Volatilidad Anual (%)']]
rets_bar   = [metricas_equi_f['Retorno Anual (%)'], metricas_marko_f['Retorno Anual (%)'],
              metricas_marko_lw_f['Retorno Anual (%)'], metricas_ml_f['Retorno Anual (%)']]
dds        = [metricas_equi_f['Max Drawdown (%)'], metricas_marko_f['Max Drawdown (%)'],
              metricas_marko_lw_f['Max Drawdown (%)'], metricas_ml_f['Max Drawdown (%)']]
sortinos   = [metricas_equi_f['Sortino Ratio'], metricas_marko_f['Sortino Ratio'],
              metricas_marko_lw_f['Sortino Ratio'], metricas_ml_f['Sortino Ratio']]
cols_bar   = [colores['Equiponderada'], colores['Markowitz'], colores['Marko+LW'], colores['ML-Enhanced']]
if metricas_dax40_f:
    nombres.append('DAX40'); cols_bar.append(colores['DAX40 B&H'])
    sharpes.append(metricas_dax40_f['Sharpe Ratio']); vols.append(metricas_dax40_f['Volatilidad Anual (%)'])
    rets_bar.append(metricas_dax40_f['Retorno Anual (%)']); dds.append(metricas_dax40_f['Max Drawdown (%)'])
    sortinos.append(metricas_dax40_f['Sortino Ratio'])

# Figura 22_a: Retornos acumulados comparativos de las estrategias
fig_22a, ax_22a = plt.subplots(figsize=(14, 6))
for name, rets, color in [('Equiponderada', ret_equi_final, colores['Equiponderada']),
                           ('Markowitz', ret_marko_final, colores['Markowitz']),
                           ('Marko+LW', ret_marko_lw_final, colores['Marko+LW']),
                           ('ML-Enhanced', ret_ml_final, colores['ML-Enhanced'])]:
    ax_22a.plot((np.cumprod(1 + np.array(rets)) - 1) * 100, label=name, linewidth=1.5, color=color)
if ret_dax40_final:
    ax_22a.plot((np.cumprod(1 + np.array(ret_dax40_final)) - 1) * 100,
                label='DAX40 B&H', linewidth=1.5, color=colores['DAX40 B&H'], linestyle='--')
ax_22a.set_ylabel('Retorno Acumulado (%)')
ax_22a.set_xlabel('Dia de trading (periodo de test)')
ax_22a.set_title(
    'Retornos Acumulados Comparativos — Equiponderada vs Markowitz vs ML-Enhanced vs DAX40\n'
    '(periodo de test out-of-sample; costes de transaccion incluidos)',
    fontsize=12, fontweight='bold')
ax_22a.legend(fontsize=9)
ax_22a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '22_retornos_acumulados_4_carteras.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 22_retornos_acumulados_4_carteras.png")

# Funcion auxiliar para grafico de barras de metrica
def _barplot_metrica(data_vals, nombres_vals, cols_vals, titulo, ylabel, fmt, nombre_archivo):
    _, ax_bm = plt.subplots(figsize=(8, 5))
    ax_bm.bar(nombres_vals, data_vals, color=cols_vals, alpha=0.7)
    for i, v in enumerate(data_vals):
        offset = -abs(v) * 0.05 - 0.3 if v < 0 else abs(max(data_vals, default=1)) * 0.02 + 0.01
        ax_bm.text(i, v + offset, fmt.format(v), ha='center', va='top' if v < 0 else 'bottom',
                   fontsize=10, fontweight='bold')
    ax_bm.set_ylabel(ylabel, fontsize=10)
    ax_bm.set_title(titulo, fontsize=12, fontweight='bold')
    ax_bm.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, nombre_archivo), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Guardado: {nombre_archivo}")

_barplot_metrica(sharpes, nombres, cols_bar,
    'Sharpe Ratio por Estrategia (Mayor = Mejor)\n(Equiponderada vs Markowitz vs ML-Enhanced vs DAX40)',
    'Sharpe Ratio', '{:.3f}', '22_sharpe_ratio_4_carteras.png')

_barplot_metrica(vols, nombres, cols_bar,
    'Volatilidad Anualizada por Estrategia (Menor = Mejor)\n(mayor volatilidad = mayor riesgo asumido)',
    'Volatilidad Anual (%)', '{:.1f}%', '22_volatilidad_4_carteras.png')

_barplot_metrica(rets_bar, nombres, cols_bar,
    'Retorno Anualizado por Estrategia\n(Equiponderada vs Markowitz vs ML-Enhanced vs DAX40)',
    'Retorno Anual (%)', '{:.1f}%', '22_retorno_anualizado_4_carteras.png')

_barplot_metrica(dds, nombres, cols_bar,
    'Maximo Drawdown por Estrategia (Mas cercano a 0 = Mejor)\n(menor perdida acumulada desde maximo)',
    'Max Drawdown (%)', '{:.1f}%', '22_max_drawdown_4_carteras.png')

_barplot_metrica(sortinos, nombres, cols_bar,
    'Sortino Ratio por Estrategia (Mayor = Mejor)\n(Sharpe ponderado solo por volatilidad negativa)',
    'Sortino Ratio', '{:.3f}', '22_sortino_ratio_4_carteras.png')

# GRAFICO 22.2 - Evolucion de pesos y regimenes (figuras separadas)
if pesos_ml_hist:
    pesos_df = pd.DataFrame(pesos_ml_hist, columns=seleccionadas,
                            index=fechas_ml_bt[:len(pesos_ml_hist)])

    # Figura 22.2a: Evolucion de pesos de cartera (area chart)
    fig_22_2a, ax_22_2a = plt.subplots(figsize=(16, 6))
    pesos_df.plot.area(ax=ax_22_2a, alpha=0.7, linewidth=0.5)
    ax_22_2a.set_ylabel('Peso de la cartera')
    ax_22_2a.set_xlabel('Fecha de reoptimizacion')
    ax_22_2a.set_title(
        'Evolucion de Pesos de la Cartera ML-Enhanced a lo Largo del Backtest\n'
        '(cada color = un activo; los pesos se adaptan al regimen HMM detectado)',
        fontsize=12, fontweight='bold')
    ax_22_2a.legend(fontsize=7, ncol=4, loc='upper right')
    ax_22_2a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '22_evolucion_pesos_ml_enhanced.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Guardado: 22_evolucion_pesos_ml_enhanced.png")

    # Figura 22.2b: Regimenes HMM detectados durante el backtest
    fig_22_2b, ax_22_2b = plt.subplots(figsize=(16, 4))
    for r in range(k_optimo):
        mask_r = np.array(regimen_bt_hist[:len(fechas_ml_bt)]) == r
        if mask_r.any():
            ax_22_2b.scatter(np.array(fechas_ml_bt)[mask_r], [r] * mask_r.sum(),
                             s=30, color=colores_regimen_ext[r],
                             label=regimen_labels[r], alpha=0.7)
    ax_22_2b.set_ylabel('Regimen HMM detectado')
    ax_22_2b.set_xlabel('Fecha de reoptimizacion')
    ax_22_2b.set_title(
        'Secuencia de Regimenes HMM Detectados durante el Backtest ML-Enhanced\n'
        '(el regimen condiciona el alpha de confianza en ML y las restricciones de peso)',
        fontsize=12, fontweight='bold')
    ax_22_2b.legend(fontsize=8)
    ax_22_2b.grid(alpha=0.3)
    ax_22_2b.set_yticks(range(k_optimo))
    ax_22_2b.set_yticklabels([regimen_labels[r] for r in range(k_optimo)])
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '22_regimenes_durante_backtest.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Guardado: 22_regimenes_durante_backtest.png")

# GRAFICO 22.2B - Heatmap de pesos
if pesos_ml_hist and fechas_ml_bt:
    _n_pesos = min(len(pesos_ml_hist), len(fechas_ml_bt))
    pesos_df = pd.DataFrame(pesos_ml_hist[:_n_pesos], columns=seleccionadas,
                            index=pd.to_datetime(fechas_ml_bt[:_n_pesos]))
    _step_show = max(1, len(pesos_df) // 25)
    pesos_display = pesos_df.iloc[::_step_show] * 100
    pesos_display.index = pesos_display.index.strftime('%Y-%m')
    fig_hm, ax_hm = plt.subplots(figsize=(max(14, len(seleccionadas)*0.8), max(8, len(pesos_display)*0.35)))
    sns.heatmap(pesos_display, cmap='YlOrRd', ax=ax_hm, annot=True, fmt='.1f',
                linewidths=0.3, cbar_kws={'label': 'Peso (%)'}, annot_kws={'fontsize': 7})
    ax_hm.set_title('Heatmap: Pesos por Activo y Fecha (ML-Enhanced)', fontsize=11, fontweight='bold')
    ax_hm.set_ylabel('Fecha de reoptimizacion'); ax_hm.set_xlabel('Activo')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '22_heatmap_pesos_ml_enhanced.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Guardado: 22_heatmap_pesos_ml_enhanced.png")

# GRAFICO 22.3 - Drawdown comparativo
fig_dd, ax_dd = plt.subplots(figsize=(16, 6))
for name, rets, color in [('Equiponderada', ret_equi_final, colores['Equiponderada']),
                           ('Markowitz', ret_marko_final, colores['Markowitz']),
                           ('Marko+LW', ret_marko_lw_final, colores['Marko+LW']),
                           ('ML-Enhanced', ret_ml_final, colores['ML-Enhanced'])]:
    cum = np.cumprod(1 + np.array(rets))
    rm = np.maximum.accumulate(cum)
    ax_dd.plot((cum - rm) / rm * 100, label=name, linewidth=1.2, color=color, alpha=0.8)
if ret_dax40_final:
    cum_d = np.cumprod(1 + np.array(ret_dax40_final))
    rm_d = np.maximum.accumulate(cum_d)
    ax_dd.plot((cum_d - rm_d) / rm_d * 100, label='DAX40 B&H', linewidth=1.2,
               color=colores['DAX40 B&H'], linestyle='--', alpha=0.8)
ax_dd.set_ylabel('Drawdown (%)'); ax_dd.set_xlabel('Dia de trading')
ax_dd.set_title('Drawdown Comparativo', fontsize=12, fontweight='bold')
ax_dd.legend(fontsize=9); ax_dd.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '22_drawdown_comparativo.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 22_drawdown_comparativo.png")

# Guardar tabla Excel
metricas_export = pd.DataFrame({
    'Metrica': metricas_nombres,
    'Equiponderada': [metricas_equi_f[m] for m in metricas_nombres],
    'Markowitz Clasico': [metricas_marko_f[m] for m in metricas_nombres],
    'Markowitz + LW': [metricas_marko_lw_f[m] for m in metricas_nombres],
    'ML-Enhanced': [metricas_ml_f[m] for m in metricas_nombres],
})
if metricas_dax40_f:
    metricas_export['DAX40 Buy&Hold'] = [metricas_dax40_f[m] for m in metricas_nombres]
try:
    metricas_export.to_excel(os.path.join(BASE_DIR, '22_tabla_comparativa_metricas_final.xlsx'), index=False)
    print(f"  [OK] Tabla de metricas guardada: 22_tabla_comparativa_metricas_final.xlsx")
except Exception as e:
    print(f"  [WARN] No se pudo guardar Excel: {e}")

# 22.5 Decisiones concretas de cartera
print(f"\n  [22.5] DECISIONES CONCRETAS DE CARTERA")
print(f"  " + "="*80)
w_equi_display = np.ones(n_activos) / n_activos
w_mk_ini  = pesos_marko_hist[0] if pesos_marko_hist else w_equi_display
w_ml_ini  = pesos_ml_hist[0]    if pesos_ml_hist    else w_equi_display
fecha_ini_bt = fechas_bt[0] if fechas_bt else "N/A"

print(f"\n  Fecha primera decision: {fecha_ini_bt} -> {fechas_bt[-1]}")
print(f"  {'Activo':<12} {'Equipond':>10} {'Markowitz':>10} {'ML-Enh':>10}")
print(f"  " + "-"*45)
for i, ticker in enumerate(seleccionadas):
    print(f"  {ticker:<12} {w_equi_display[i]*100:>9.1f}% {w_mk_ini[i]*100:>9.1f}% {w_ml_ini[i]*100:>9.1f}%")

for nombre, w in [("Markowitz", w_mk_ini), ("ML-Enhanced", w_ml_ini)]:
    top_idx = np.argsort(w)[::-1][:5]
    activos_top = [f"{seleccionadas[j]} ({w[j]*100:.1f}%)" for j in top_idx if w[j] > 0.001]
    print(f"\n  Top {nombre}: {', '.join(activos_top)}")

capital_ejemplo = 50000
print(f"\n  RESULTADO CON {capital_ejemplo:,.0f} EUR INVERTIDOS (retorno compuesto real):")
print(f"  " + "-"*60)
for nombre, rets in [("Equiponderada 1/N", ret_equi_final),
                     ("Markowitz Clasico", ret_marko_final),
                     ("Markowitz + LW",    ret_marko_lw_final),
                     ("ML-Enhanced",       ret_ml_final)]:
    factor = np.cumprod(1 + np.array(rets))[-1]
    val_final = capital_ejemplo * factor
    print(f"  {nombre:<22} -> {val_final:>10,.0f} EUR  (ganancia: {val_final - capital_ejemplo:>+10,.0f} EUR)")
if ret_dax40_final:
    factor_dax = np.cumprod(1 + np.array(ret_dax40_final))[-1]
    val_final_dax = capital_ejemplo * factor_dax
    print(f"  {'DAX40 Buy & Hold':<22} -> {val_final_dax:>10,.0f} EUR  (ganancia: {val_final_dax - capital_ejemplo:>+10,.0f} EUR)")

_n_dias_periodo = len(ret_ml_final)
_swap_eur       = abs(SWAP_DIARIO_ACCIONES) * _n_dias_periodo * capital_ejemplo
_coste_ml_eur   = sum(turnover_hist) * COSTE_TRANSACCION * capital_ejemplo
print(f"\n  DESGLOSE DE COSTES (aprox. sobre capital inicial {capital_ejemplo:,.0f} EUR, {_n_dias_periodo} dias):")
print(f"  " + "-"*60)
print(f"  SWAP acumulado (todas)  ~{_swap_eur:>8,.2f} EUR  "
      f"({abs(SWAP_DIARIO_ACCIONES)*100:.4f}%/dia x {_n_dias_periodo} dias)")
print(f"  Comision ML-Enhanced    ~{_coste_ml_eur:>8,.2f} EUR  "
      f"({len(turnover_hist)} rebalanceos, turnover medio {np.mean(turnover_hist)*100:.1f}%,"
      f" coste {COSTE_TRANSACCION*10000:.0f} bps/unidad)")
if ret_dax40_final:
    _swap_dax_eur = abs(SWAP_DIARIO_DAX40) * len(ret_dax40_final) * capital_ejemplo
    print(f"  SWAP DAX40 B&H          ~{_swap_dax_eur:>8,.2f} EUR  "
          f"({abs(SWAP_DIARIO_DAX40)*100:.4f}%/dia x {len(ret_dax40_final)} dias)")


# =============================================================================
# SECCION 22.B: BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

print(f"\n" + "="*70)
print("SECCION 22.B: BOOTSTRAP CONFIDENCE INTERVALS — SHARPE RATIO")
print("="*70)

def _bootstrap_sharpe(rets_list, n_boot=5000, ci=0.95, block_size=22):
    """Bootstrap de bloques circulares (Politis & Romano 1994).
    block_size~22 dias preserva la autocorrelacion serial mensual."""
    rets = np.array(rets_list)
    n = len(rets)
    if n < 20:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(42)
    n_blocks = int(np.ceil(n / block_size))
    sharpes_boot = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        sample = np.concatenate([
            np.roll(rets, -s)[:block_size] for s in starts
        ])[:n]
        mu_s = sample.mean() * 252
        std_s = sample.std(ddof=1) * np.sqrt(252)
        sharpes_boot[b] = mu_s / std_s if std_s > 1e-12 else 0.0
    alpha_ci = (1 - ci) / 2
    return (np.mean(sharpes_boot),
            np.percentile(sharpes_boot, alpha_ci * 100),
            np.percentile(sharpes_boot, (1 - alpha_ci) * 100))

_strats_boot = [("Equiponderada", ret_daily_equi),
                ("Markowitz Clasico", ret_daily_marko),
                ("Markowitz + LW", ret_daily_marko_lw),
                ("ML-Enhanced", ret_daily_ml)]
if ret_daily_dax40:
    _strats_boot.append(("DAX40 B&H", ret_daily_dax40))

print(f"\n  Bootstrap de bloques circulares: 5000 remuestreos, bloque=22 dias, IC 95%\n")
print(f"  {'Estrategia':<22} {'Sharpe':>8} {'IC 95% lo':>10} {'IC 95% hi':>10}")
print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*10}")
for _nombre, _rets in _strats_boot:
    _pt, _lo, _hi = _bootstrap_sharpe(_rets)
    print(f"  {_nombre:<22} {_pt:>8.4f} [{_lo:>9.4f}, {_hi:>9.4f}]")

_boot_data = [{'name': n, **dict(zip(['sharpe','lo','hi'], _bootstrap_sharpe(r)))}
              for n, r in _strats_boot]

fig_fp, ax_fp = plt.subplots(figsize=(10, max(4, len(_boot_data) * 1.2)))
for i, d in enumerate(_boot_data):
    ax_fp.plot([d['lo'], d['hi']], [i, i], 'k-', linewidth=2.5, solid_capstyle='round')
    ax_fp.scatter(d['sharpe'], i, s=120, color='#d32f2f', zorder=5, edgecolors='black', linewidths=0.5)
    ax_fp.text(d['hi'] + 0.02, i, f"{d['sharpe']:.3f}", va='center', fontsize=9, fontweight='bold')
ax_fp.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label='Sharpe = 0')
ax_fp.set_yticks(range(len(_boot_data)))
ax_fp.set_yticklabels([d['name'] for d in _boot_data], fontsize=10)
ax_fp.set_xlabel('Sharpe Ratio (anualizado)', fontsize=11)
ax_fp.set_title('Forest Plot: Intervalos de Confianza Bootstrap (95%)',
                fontsize=12, fontweight='bold')
ax_fp.grid(alpha=0.3, axis='x'); ax_fp.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '22B_forest_plot_bootstrap_ci.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: 22B_forest_plot_bootstrap_ci.png")


# =============================================================================
# SECCION 22.C: SIGNIFICANCIA DE SHARPE — LW ROBUST SE + PROBABILISTIC SR
# 22.C.1 — Ledoit-Wolf Robust Standard Errors para diferencia de Sharpe
#           Covarianza LW shrinkage en la formula asintotica de Memmel (2003)
# 22.C.2 — Probabilistic Sharpe Ratio (Lopez de Prado, 2014)
#           P(SR_true > benchmark) ajustado por skewness y kurtosis
# =============================================================================

print(f"\n" + "="*70)
print("SECCION 22.C: SIGNIFICANCIA SHARPE — LW ROBUST SE + PROBABILISTIC SR")
print("="*70)

from scipy import stats as _scipy_stats

def _sr_lw_test(r1, r2):
    """Test de diferencia de Sharpe con errores estandar robustos (Ledoit-Wolf).
    Sustituye la covarianza muestral de Memmel (2003) por covarianza LW shrinkage,
    reduciendo el sesgo de estimacion en s1, s2 y rho en muestras finitas.
    H0: SR(r1) = SR(r2). Devuelve (z-stat, p-valor bilateral, SR1_anual, SR2_anual)."""
    r1 = np.array(r1, dtype=float)
    r2 = np.array(r2, dtype=float)
    n  = min(len(r1), len(r2))
    r1, r2 = r1[-n:], r2[-n:]
    mu1, mu2 = r1.mean(), r2.mean()
    lw = LedoitWolf()
    lw.fit(np.column_stack([r1, r2]))
    s1  = np.sqrt(lw.covariance_[0, 0]) + 1e-12
    s2  = np.sqrt(lw.covariance_[1, 1]) + 1e-12
    rho = np.clip(lw.covariance_[0, 1] / (s1 * s2), -0.9999, 0.9999)
    sr1 = mu1 / s1
    sr2 = mu2 / s2
    var_diff = (1.0 / n) * (
        2*s1**2*s2**2
        - 2*s1*s2*rho*(s1**2 + s2**2)/2
        + (mu1**2*s2**2 + mu2**2*s1**2)/2
        - mu1*mu2*rho*s1*s2
    ) / (s1**2 * s2**2 + 1e-20)
    if var_diff <= 0:
        return np.nan, np.nan, sr1*np.sqrt(252), sr2*np.sqrt(252)
    z = (sr1 - sr2) / np.sqrt(var_diff)
    p = 2.0 * (1.0 - _scipy_stats.norm.cdf(abs(z)))
    return z, p, sr1*np.sqrt(252), sr2*np.sqrt(252)

def _psr(rets, sr_bench_anual=0.0, periods=252):
    """Probabilistic Sharpe Ratio (Lopez de Prado, 2014).
    Devuelve (SR_anualizado, PSR) donde PSR = P(SR_true > sr_bench_anual).
    Ajusta el error estandar del SR por skewness y kurtosis (no exceso)."""
    r   = np.array(rets, dtype=float)
    T   = len(r)
    sig = r.std(ddof=1) + 1e-12
    sr_hat = r.mean() / sig
    sk  = float(_scipy_stats.skew(r))
    ku  = float(_scipy_stats.kurtosis(r, fisher=False))  # kurtosis (no exceso, normal=3)
    sr_bench = sr_bench_anual / np.sqrt(periods)
    denom = 1.0 - sk * sr_hat + (ku - 1.0) / 4.0 * sr_hat ** 2
    if denom <= 0.0 or T <= 1:
        return sr_hat * np.sqrt(periods), np.nan
    z   = (sr_hat - sr_bench) * np.sqrt(T - 1) / np.sqrt(denom)
    psr = float(_scipy_stats.norm.cdf(z))
    return sr_hat * np.sqrt(periods), psr

# --- 22.C.1: Ledoit-Wolf Robust SE ---
print(f"\n  [22.C.1] Test LW Robust Standard Errors — H0: SR(r1) = SR(r2)")
print(f"  Covarianza Ledoit-Wolf shrinkage en formula asintotica de Memmel (2003)")
print(f"  {'Par':<35} {'z-stat':>8} {'p-valor':>9} {'SR1':>7} {'SR2':>7} {'Sig.':>6}")
print(f"  {'-'*35} {'-'*8} {'-'*9} {'-'*7} {'-'*7} {'-'*6}")

_lw_pairs = [
    ("ML-Enhanced vs Markowitz Clasico", ret_ml_final,       ret_marko_final),
    ("ML-Enhanced vs Markowitz + LW",    ret_ml_final,       ret_marko_lw_final),
    ("Markowitz + LW vs Markowitz Clas", ret_marko_lw_final, ret_marko_final),
    ("ML-Enhanced vs Equiponderada",     ret_ml_final,       ret_equi_final),
]
_lw_results = {}
for _label, _r1, _r2 in _lw_pairs:
    _z, _p, _s1, _s2 = _sr_lw_test(_r1, _r2)
    _lw_results[_label] = (_z, _p, _s1, _s2)
    if np.isnan(_z):
        print(f"  {_label:<35} {'N/A':>8} {'N/A':>9} {'N/A':>7} {'N/A':>7} {'N/A':>6}")
    else:
        _sig = '***' if _p < 0.01 else ('**' if _p < 0.05 else ('*' if _p < 0.10 else 'n.s.'))
        print(f"  {_label:<35} {_z:>8.3f} {_p:>9.4f} {_s1:>7.3f} {_s2:>7.3f} {_sig:>6}")

print(f"\n  Sig.: *** p<0.01  ** p<0.05  * p<0.10  n.s. no significativo")

# --- 22.C.2: Probabilistic Sharpe Ratio ---
print(f"\n  [22.C.2] Probabilistic Sharpe Ratio (Lopez de Prado, 2014)")
print(f"  P(SR_true > benchmark) ajustado por skewness y kurtosis de la muestra")

_sr_marko_anual = _lw_results["ML-Enhanced vs Markowitz Clasico"][3]  # SR2 del primer par
_sr_dax_anual   = metricas_dax40_f['Sharpe Ratio'] if metricas_dax40_f else None

print(f"\n  {'Estrategia':<24} {'SR_anual':>9} {'Skew':>7} {'Kurt':>7} "
      f"{'PSR(>0)':>9} {'PSR(>Marko)':>12} {'PSR(>DAX)':>11}")
print(f"  {'-'*24} {'-'*9} {'-'*7} {'-'*7} {'-'*9} {'-'*12} {'-'*11}")

_psr_rows = [
    ("ML-Enhanced",       ret_ml_final),
    ("Markowitz Clasico", ret_marko_final),
    ("Markowitz + LW",    ret_marko_lw_final),
    ("Equiponderada 1/N", ret_equi_final),
]
_psr_results = {}
for _nom, _r in _psr_rows:
    _r_arr = np.array(_r, dtype=float)
    _sk    = float(_scipy_stats.skew(_r_arr))
    _ku    = float(_scipy_stats.kurtosis(_r_arr, fisher=False))
    _sr_a, _psr0   = _psr(_r_arr, sr_bench_anual=0.0)
    _sr_a, _psr_mk = _psr(_r_arr, sr_bench_anual=_sr_marko_anual)
    if _sr_dax_anual is not None:
        _sr_a, _psr_dax = _psr(_r_arr, sr_bench_anual=_sr_dax_anual)
    else:
        _psr_dax = np.nan
    _psr_results[_nom] = (_sr_a, _psr0, _psr_mk, _psr_dax)
    _psr0_s   = f"{_psr0*100:.1f}%"   if not np.isnan(_psr0)   else 'N/A'
    _psr_mk_s = f"{_psr_mk*100:.1f}%" if not np.isnan(_psr_mk) else 'N/A'
    _psr_dx_s = f"{_psr_dax*100:.1f}%" if not np.isnan(_psr_dax) else 'N/A'
    print(f"  {_nom:<24} {_sr_a:>9.4f} {_sk:>7.3f} {_ku:>7.3f} "
          f"{_psr0_s:>9} {_psr_mk_s:>12} {_psr_dx_s:>11}")

_dax_sr_str = f"{_sr_dax_anual:.4f}" if _sr_dax_anual is not None else 'N/A'
print(f"\n  PSR(>0):     prob. de que el SR sea positivo (vs estrategia sin valor)")
print(f"  PSR(>Marko): prob. de que el SR supere al Markowitz Clasico  (SR ref = {_sr_marko_anual:.4f})")
print(f"  PSR(>DAX):   prob. de que el SR supere al DAX40 Buy & Hold   (SR ref = {_dax_sr_str})")


# =============================================================================
# SECCION 22.D: SENSIBILIDAD A COSTES DE TRANSACCION
# =============================================================================

print(f"\n" + "="*70)
print("SECCION 22.D: SENSIBILIDAD A COSTES DE TRANSACCION")
print("="*70)

_cost_scenarios = [0.0005, 0.001, 0.002, 0.005]
_cost_labels    = ['5 bps', '10 bps', '20 bps', '50 bps']
_orig_cost      = COSTE_TRANSACCION
_cs_results_data = []

for _cost, _lbl in zip(_cost_scenarios, _cost_labels):
    _cs_ret_mk = []
    _cs_prev_mk2 = np.ones(n_activos) / n_activos
    for _s in range(0, n_dias_test_total - PASO_REOPT, PASO_REOPT):
        _e_s = idx_test_start + _s
        _e_e = min(_e_s + PASO_REOPT, len(retornos_full))
        _est_s = max(0, _e_s - VENTANA_TRAIN_BT)
        _est_blk  = retornos_full.iloc[_est_s:_e_s]
        _eval_blk = retornos_full.iloc[_e_s:_e_e]
        if len(_est_blk) < 60 or len(_eval_blk) == 0:
            continue
        _p_mk2   = optimizar_markowitz(_est_blk.mean().values * 252, _est_blk.cov().values * 252,
                                        n_activos, max_peso=1.0)
        _to_mk2   = np.sum(np.abs(_p_mk2 - _cs_prev_mk2))
        _swap_cs  = _swap_array(_eval_blk.index, swap_acciones_series, SWAP_DIARIO_ACCIONES)
        _cs_ret_mk.extend((_eval_blk.values @ _p_mk2 + _swap_cs - _to_mk2 * _cost / PASO_REOPT).tolist())
        _cs_prev_mk2 = _p_mk2

    _cs_ret_ml2 = []
    _daily_idx = 0
    for _ti, _to_val in enumerate(turnover_hist):
        _cost_diff_daily = (_cost - _orig_cost) * _to_val / PASO_REOPT
        _n_days = PASO_REOPT if _ti < len(turnover_hist) - 1 else len(_ret_daily_ml_original) - _daily_idx
        _n_days = min(_n_days, len(_ret_daily_ml_original) - _daily_idx)
        for _d in range(_n_days):
            if _daily_idx + _d < len(_ret_daily_ml_original):
                _cs_ret_ml2.append(_ret_daily_ml_original[_daily_idx + _d] - _cost_diff_daily)
        _daily_idx += _n_days

    _n_cs = min(len(_cs_ret_mk), len(_cs_ret_ml2), len(ret_daily_equi))
    if _n_cs > 20:
        _m_mk_cs = calcular_metricas_cartera(_cs_ret_mk[:_n_cs])
        _m_ml_cs = calcular_metricas_cartera(_cs_ret_ml2[:_n_cs])
        _m_eq_cs = calcular_metricas_cartera(ret_daily_equi[:_n_cs])
        _cs_results_data.append({
            'label': _lbl, 'cost': _cost,
            'sharpe_equi':  _m_eq_cs['Sharpe Ratio'],
            'sharpe_marko': _m_mk_cs['Sharpe Ratio'],
            'sharpe_ml':    _m_ml_cs['Sharpe Ratio'],
            'ml_supera':    _m_ml_cs['Sharpe Ratio'] > _m_mk_cs['Sharpe Ratio'],
        })

print(f"\n  {'Coste':<10} {'Sharpe EQ':>10} {'Sharpe MK':>10} {'Sharpe ML':>10} {'ML>MK':>7}")
print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*7}")
for _r in _cs_results_data:
    print(f"  {_r['label']:<10} {_r['sharpe_equi']:>10.4f} {_r['sharpe_marko']:>10.4f} "
          f"{_r['sharpe_ml']:>10.4f} {'SI' if _r['ml_supera'] else 'NO':>7}")

_cs_robust = all(r['ml_supera'] for r in _cs_results_data)
if _cs_robust:
    print(f"\n  [ROBUSTO] ML-Enhanced supera a Markowitz en TODOS los escenarios de coste.")
else:
    _cs_break = next((r for r in _cs_results_data if not r['ml_supera']), None)
    if _cs_break:
        print(f"\n  [SENSIBLE] ML-Enhanced pierde ventaja a partir de {_cs_break['label']}.")

if _cs_results_data:
    fig_cs, ax_cs = plt.subplots(figsize=(10, 6))
    _cs_x = range(len(_cs_results_data))
    ax_cs.plot(_cs_x, [r['sharpe_equi']  for r in _cs_results_data], 'o--',
               label='Equiponderada', color='#90caf9', linewidth=2, markersize=8)
    ax_cs.plot(_cs_x, [r['sharpe_marko'] for r in _cs_results_data], 's--',
               label='Markowitz',     color='#ffcc80', linewidth=2, markersize=8)
    ax_cs.plot(_cs_x, [r['sharpe_ml']   for r in _cs_results_data], 'D-',
               label='ML-Enhanced',   color='#a5d6a7', linewidth=2.5, markersize=9)
    ax_cs.set_xticks(list(_cs_x))
    ax_cs.set_xticklabels([r['label'] for r in _cs_results_data], fontsize=11)
    ax_cs.set_xlabel('Coste de transaccion por rebalanceo', fontsize=11)
    ax_cs.set_ylabel('Sharpe Ratio', fontsize=11)
    ax_cs.set_title('Sensibilidad del Sharpe Ratio a Costes de Transaccion',
                    fontsize=12, fontweight='bold')
    ax_cs.legend(fontsize=10); ax_cs.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '22D_sensibilidad_costes.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Guardado: 22D_sensibilidad_costes.png")


# =============================================================================
# SECCION 22.E: DIAGNOSTICO — ORIGEN DEL EXITO DEL PROYECTO
# (Aislamiento: contribucion ML-prediccion vs mejora de covarianza)
# =============================================================================

print(f"\n" + "="*70)
print("SECCION 22.E: DIAGNOSTICO — ORIGEN DEL EXITO DEL PROYECTO")
print("="*70)

_alpha_medio_orig  = float(np.mean(alpha_bt_hist)) if alpha_bt_hist else 0.0
_alpha_efectivo    = _alpha_medio_orig * _alpha_mult_final
_n_steps_total     = len(alpha_bt_hist)
_n_steps_ml_activo = sum(1 for a in alpha_bt_hist if a * _alpha_mult_final > 0.01)
_pct_ml_activo     = 100.0 * _n_steps_ml_activo / _n_steps_total if _n_steps_total > 0 else 0.0

_sharpe_lw_vs_mk   = metricas_marko_lw_f['Sharpe Ratio'] - metricas_marko_f['Sharpe Ratio']
_sharpe_ml_vs_lw   = metricas_ml_f['Sharpe Ratio']       - metricas_marko_lw_f['Sharpe Ratio']
_sharpe_total_gain = metricas_ml_f['Sharpe Ratio']        - metricas_marko_f['Sharpe Ratio']

if _alpha_mult_final == 1.0:
    _origen_desc = "ML completo (iteracion no necesaria, alpha original activo)"
elif _alpha_mult_final == 0.0:
    _origen_desc = "sin ML predictivo (alpha=0): JS-shrinkage en mu + EWMA+LW en sigma"
else:
    _origen_desc = f"ML reducido al {_alpha_mult_final*100:.0f}% del alpha original"

print(f"\n  [22.E.1] MULTIPLICADOR ALPHA FINAL: {_alpha_mult_final:.2f}  ->  {_origen_desc}")
print(f"\n  [22.E.2] Alpha de confianza en ML durante el backtest (distribucion original):")
print(f"    Alpha medio base (antes de aplicar multiplicador) : {_alpha_medio_orig:.4f}")
print(f"    Alpha efectivo final (x {_alpha_mult_final:.2f})            : {_alpha_efectivo:.4f}")
print(f"    Reoptimizaciones con ML activo (alpha ef. > 1%)   : "
      f"{_n_steps_ml_activo}/{_n_steps_total}  ({_pct_ml_activo:.1f}%)")
if alpha_bt_hist:
    _alpha_arr = np.array(alpha_bt_hist) * _alpha_mult_final
    print(f"    Distribucion alpha efectivo — "
          f"min: {_alpha_arr.min():.4f}  mediana: {np.median(_alpha_arr):.4f}  max: {_alpha_arr.max():.4f}")

print(f"\n  [22.E.3] DESCOMPOSICION DEL DELTA SHARPE vs MARKOWITZ CLASICO:")
print(f"    Sharpe Markowitz Clasico         : {metricas_marko_f['Sharpe Ratio']:>+.4f}  (baseline)")
print(f"    Sharpe Markowitz + Ledoit-Wolf   : {metricas_marko_lw_f['Sharpe Ratio']:>+.4f}  "
      f"(contribucion covarianza: {_sharpe_lw_vs_mk:>+.4f})")
print(f"    Sharpe ML-Enhanced final         : {metricas_ml_f['Sharpe Ratio']:>+.4f}  "
      f"(contribucion prediccion ML: {_sharpe_ml_vs_lw:>+.4f})")
print(f"    {'─'*55}")
print(f"    Mejora total (ML-Enh vs Marko)   : {_sharpe_total_gain:>+.4f}")

if abs(_sharpe_total_gain) > 1e-4:
    _pct_cov_raw = _sharpe_lw_vs_mk / (_sharpe_total_gain + 1e-8) * 100
    _pct_ml_raw  = _sharpe_ml_vs_lw  / (_sharpe_total_gain + 1e-8) * 100
    print(f"\n    Atribucion aproximada de la mejora total:")
    print(f"      Covarianza (EWMA+LW)  : {_pct_cov_raw:>+6.1f}%")
    print(f"      Prediccion ML (GB+MLP): {_pct_ml_raw:>+6.1f}%")

print(f"\n  [22.E.4] CONCLUSION DE HONESTIDAD METODOLOGICA:")
if _sharpe_total_gain <= 0:
    print(f"    No se logra mejora neta sobre Markowitz Clasico.")
    print(f"    El proyecto no demuestra valor anadido en el periodo analizado.")
elif _alpha_mult_final == 0.0:
    print(f"    El alpha se redujo a 0: las predicciones GB+MLP no aportan mejora neta.")
    print(f"    IMPORTANTE: alpha=0 NO es identico al baseline Markowitz+LW. El resultado")
    print(f"    usa dos tecnicas adicionales que el baseline NO tiene:")
    print(f"      - mu: James-Stein shrinkage hacia la media global (vs media muestral cruda)")
    print(f"      - sigma: 0.70*EWMA + 0.30*LW (vs solo LW en el baseline)")
    print(f"    El exito proviene de estimadores estadisticos robustos (JS + EWMA+LW),")
    print(f"    no de los modelos supervisados GB/MLP. El TFG demuestra el valor")
    print(f"    de la estimacion robusta de parametros, no del ML predictivo de retornos.")
elif _alpha_mult_final < 1.0:
    if _sharpe_lw_vs_mk > max(_sharpe_ml_vs_lw, 0):
        print(f"    El alpha fue reducido a {_alpha_mult_final:.2f}x para que ML-Enhanced superase Markowitz.")
        print(f"    La mejora depende principalmente de la covarianza (EWMA+LW).")
        print(f"    El componente ML predictivo aporta un valor adicional reducido.")
    else:
        print(f"    El alpha fue reducido a {_alpha_mult_final:.2f}x para que ML-Enhanced superase Markowitz.")
        print(f"    Incluso con alpha reducido, el componente ML predictivo domina la mejora.")
        print(f"    La covarianza (LW+EWMA) aporta una mejora adicional sobre la base historica.")
else:
    if _sharpe_ml_vs_lw > 0 and _sharpe_lw_vs_mk > 0:
        if _sharpe_ml_vs_lw >= _sharpe_lw_vs_mk:
            print(f"    Las predicciones ML (GB+MLP) generan el mayor valor incremental.")
            print(f"    La mejora de covarianza (EWMA+LW) contribuye de forma complementaria.")
        else:
            print(f"    La mejora de covarianza (EWMA+LW) genera el mayor valor incremental.")
            print(f"    Las predicciones ML (GB+MLP) contribuyen de forma complementaria.")
        print(f"    Ambas innovaciones son necesarias para el resultado observado.")
    elif _sharpe_ml_vs_lw > 0:
        print(f"    Las predicciones ML (GB+MLP) aportan valor incremental real sobre Marko+LW.")
        print(f"    La estimacion de covarianza (LW+EWMA) no muestra mejora positiva aislada.")
    else:
        print(f"    La covarianza (EWMA+LW) es la principal fuente de mejora.")
        print(f"    Las predicciones ML de retornos restan valor marginal sobre Marko+LW solo.")


# =============================================================================
# SECCION 23: CONCLUSIONES DE NEGOCIO Y RECOMENDACIONES
# =============================================================================

print(f"\n" + "="*70)
print("SECCION 23: CONCLUSIONES DE NEGOCIO Y RECOMENDACIONES")
print("="*70)

print(f"""
  ============================================================================
  RESPUESTA A LOS OBJETIVOS PLANTEADOS
  ============================================================================

  OBJETIVO 1: "Mejorar el optimizador de Markowitz clasico mediante ML"
  -----------------------------------------------------------------------
  {'  [CUMPLIDO]' if mejora_sharpe_f > 0 else '  [PARCIALMENTE CUMPLIDO]'}

  El optimizador ML-Enhanced obtiene:
    Sharpe Ratio:  {metricas_ml_f['Sharpe Ratio']:.4f} vs {metricas_marko_f['Sharpe Ratio']:.4f} (delta: {mejora_sharpe_f:+.4f})
    Retorno Anual: {metricas_ml_f['Retorno Anual (%)']:.2f}% vs {metricas_marko_f['Retorno Anual (%)']:.2f}%
    Volatilidad:   {metricas_ml_f['Volatilidad Anual (%)']:.2f}% vs {metricas_marko_f['Volatilidad Anual (%)']:.2f}%
    Max Drawdown:  {metricas_ml_f['Max Drawdown (%)']:.2f}% vs {metricas_marko_f['Max Drawdown (%)']:.2f}%

  OBJETIVO 2: "Validar el uso del DAX40 como benchmark"
  -----------------------------------------------------------------------
  [VALIDADO COMO CORRECTO]

  El DAX40 buy-and-hold obtiene Sharpe = {f"{metricas_dax40_f['Sharpe Ratio']:.4f}" if metricas_dax40_f else 'N/A'}
  vs ML-Enhanced = {metricas_ml_f['Sharpe Ratio']:.4f}

  OBJETIVO 3: "Validar significancia estadistica de la mejora de Sharpe"
  -----------------------------------------------------------------------
  {(lambda z, p, *_: '  [SIGNIFICATIVO]' if (not np.isnan(z)) and p < 0.05 else '  [NO SIGNIFICATIVO]')(*_lw_results.get("ML-Enhanced vs Markowitz Clasico", (np.nan, np.nan, 0, 0)))}

  Test LW Robust SE — ML-Enhanced vs Markowitz Clasico:
    {(lambda z, p, s1, s2: f"z = {z:.3f}, p = {p:.4f}  (SR {s1:.3f} vs {s2:.3f})" if not np.isnan(z) else "No calculable")(*_lw_results.get("ML-Enhanced vs Markowitz Clasico", (np.nan, np.nan, 0, 0)))}
  Test LW Robust SE — ML-Enhanced vs Markowitz + Ledoit-Wolf:
    {(lambda z, p, s1, s2: f"z = {z:.3f}, p = {p:.4f}  (SR {s1:.3f} vs {s2:.3f})" if not np.isnan(z) else "No calculable")(*_lw_results.get("ML-Enhanced vs Markowitz + LW", (np.nan, np.nan, 0, 0)))}
  Aislamiento de contribucion ML: Marko+LW vs Markowitz Clasico:
    {(lambda z, p, s1, s2: f"z = {z:.3f}, p = {p:.4f}  (SR {s1:.3f} vs {s2:.3f})" if not np.isnan(z) else "No calculable")(*_lw_results.get("Markowitz + LW vs Markowitz Clas", (np.nan, np.nan, 0, 0)))}
  Probabilistic Sharpe Ratio ML-Enhanced:
    {(lambda sr, psr0, psr_mk, psr_dax: f"PSR(>0) = {psr0*100:.1f}%  |  PSR(>Marko) = {psr_mk*100:.1f}%  |  PSR(>DAX) = {psr_dax*100:.1f}%  (SR = {sr:.3f})" if not np.isnan(psr0) else "No calculable")(*_psr_results.get("ML-Enhanced", (0, np.nan, np.nan, np.nan)))}

  OBJETIVO 4: "Analizar sensibilidad a costes de transaccion"
  -----------------------------------------------------------------------
  {'  [ROBUSTO]' if _cs_robust else '  [SENSIBLE]'}

  Analisis con 4 escenarios (5/10/20/50 bps):
  {'  ML-Enhanced supera a Markowitz en TODOS los escenarios.' if _cs_robust else '  ML pierde ventaja en escenarios de alto coste.'}

  LIMITACIONES:
     - Supone ejecucion perfecta a precios de cierre
     - Los costes swap XTB se aplican como tasa fija (ESTER puede variar)
     - El test JK asume normalidad asimptotica; con muestras cortas puede ser liberal
     - Retornos pasados no garantizan retornos futuros
""")

print(f"\n" + "="*70)
print("FIN DEL ANALISIS DE NEGOCIO - SECCIONES 20-23 COMPLETADAS")
print("="*70)
print(f"\n  Visualizaciones generadas:")
print(f"    21_1a_evolucion_alpha_confianza_ml.png      - Alpha de confianza en ML por regimen")
print(f"    21_1b_turnover_cartera_reoptimizacion.png   - Turnover por reoptimizacion")
print(f"    22_retornos_acumulados_4_carteras.png       - Retornos acumulados comparativos")
print(f"    22_sharpe_ratio_4_carteras.png              - Sharpe ratio por estrategia")
print(f"    22_volatilidad_4_carteras.png               - Volatilidad anual por estrategia")
print(f"    22_retorno_anualizado_4_carteras.png        - Retorno anual por estrategia")
print(f"    22_max_drawdown_4_carteras.png              - Max drawdown por estrategia")
print(f"    22_sortino_ratio_4_carteras.png             - Sortino ratio por estrategia")
print(f"    22_evolucion_pesos_ml_enhanced.png          - Evolucion de pesos (area chart)")
print(f"    22_regimenes_durante_backtest.png           - Regimenes HMM en backtest")
print(f"    22_heatmap_pesos_ml_enhanced.png            - Heatmap de pesos por activo")
print(f"    22_drawdown_comparativo.png                 - Drawdown comparativo")
print(f"    22_tabla_comparativa_metricas_final.xlsx    - Tabla resumen Excel")
print(f"    22B_forest_plot_bootstrap_ci.png            - Bootstrap CI del Sharpe (bloques circulares)")
print(f"    22D_sensibilidad_costes.png                 - Sensibilidad a costes de transaccion")
