
# Para activar el entorno virtual antes de ejecutar este script, usar:
#PS C:\Users\pbarg> 
# cd "C:\Users\pbarg\Documents\UNIVERSIDAD\4\TFG\VisualStudioCode"
#PS C:\Users\pbarg\Documents\UNIVERSIDAD\4\TFG\VisualStudioCode> 
#.venv\Scripts\Activate.ps1

"""
ruta_excel = os.path.join(os.path.dirname(os.path.abspath(__file__)), '1_dataset_completo_comprobacion_estructural.xlsx')
dataset_completo.to_excel(ruta_excel, index=True)
print(f"\nDataset guardado en: {ruta_excel}")
"""

# =============================================================================
# TFG: ANÃLISIS PREDICTIVO E INGENIERÃA CUANTITATIVA PARA OPTIMIZACIÃ“N 
#      DE CARTERAS - DAX40
# =============================================================================
# MetodologÃ­a:
#   - Modelos ML: MLPRegressor supervisado (predicciÃ³n retornos)
#              + PCA + K-Means no supervisado (detecciÃ³n rÃ©gimenes)
#   - Activos: 10-15 empresas del DAX40 (selecciÃ³n por sector + clustering)
#   - Features exÃ³genas: EUR/USD, EuroStoxx 50, Oro
#   - Benchmarks: DAX40, Cartera Markowitz clÃ¡sica, Equiponderada
#   - OptimizaciÃ³n: Markowitz con rolling windows + backtest temporal
#   - El DAX40 (^GDAXI) se usa SOLO como benchmark, no como feature del modelo,
#     para evitar multicolinealidad (el Ã­ndice estÃ¡ compuesto por los mismos activos).
# =============================================================================

# 0. IMPORTACIN DE LIBRERAS
import os
import sys
import io
import numpy as np
import yfinance as yf
import pandas as pd
import subprocess

# Configurar stdout para UTF-8 en Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

try:
    import openpyxl
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'openpyxl'])
    import openpyxl
try:
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    from scipy.spatial.distance import squareform
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scipy'])
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    from scipy.spatial.distance import squareform
try:
    from scipy import stats
except ImportError:
    pass
try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.seasonal import seasonal_decompose, STL
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'statsmodels'])
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.seasonal import seasonal_decompose, STL
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'seaborn'])
    import seaborn as sns

# =============================================================================
# 1. ACTIVOS: DAX40
# =============================================================================
#
# JUSTIFICACION Y CRITERIOS DE SELECCION DE ACTIVOS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# Â¿QUE SON LOS ACTIVOS?
#   Son las 40 empresas mas grandes cotizadas en la bolsa de Frankfurt (DAX40).
#   Incluyen blue chips de todos los sectores de la economia alemana:
#   automocion (BMW, Mercedes), tecnologia (SAP, Siemens), quimico (BASF),
#   finanzas (Deutsche Bank, Allianz), etc.
#
# Â¿POR QUE DAX40 Y NO OTROS INDICES?
#   1. UNICIDAD SECTORIAL: Alemania es una potencia manufacturera e industrial.
#      A diferencia del NASDAQ (tecnologia concentrada) o S&P500 (sobrerepresentado
#      en GAFAM), el DAX40 ofrece exposicion diversificada a la economia real:
#      industriales pesados, constructores, quimicos, energia, finanzas tradicionales.
#   
#   2. MONEDA UNICA (EUR): Todos los activos cotizan en EUR -> no hay ruido por
#      tipos de cambio entre empresas. La exposicion a EUR/USD se captura en
#      features macroeconomicos separados (ver Seccion 2).
#
#   3. DATOS HISTORICOS LIMPIOS Y ACCESIBLES:
#      - Disponibles en yfinance con alta calidad desde 2011
#      - Sin eventos corporativos complejos (spin-offs, splits, quiebras)
#      que requieran ajustes especiales
#      - Estandares de reporting IFRS (transparencia financiera)
#
#   4. RELEVANCIA ACADEMICA Y EMPRESARIAL:
#      - El DAX es el benchmark de referencia para inversion en Alemania
#      - Ampliamente utilizado en investigacion sobre modelos de cartera
#      - Datos de calidad suficiente para entrenar redes neuronales
#
# Â¿POR QUE 40 EMPRESAS INICIALMENTE Y NO SELECCIONAR DIRECTAMENTE 10-15?
#   La seleccion inicial de 40 permite:
#   - Estudiar la diversificacion natural del indice
#   - Aplicar clustering jerarquico sobre 40 candidatos (los patrones de
#     comportamiento se distinguen mejor en muestras mas grandes)
#   - Eliminar empresas con datos defectuosos o gaps largos (Seccion 7)
#   - Usar scoring estadistico robusto (kurtosis, estacionariedad) sobre
#     series limpias
#   - Finalmente seleccionar 10-15 activos (Seccion 10) que son los que 
#     entrenaran el modelo DL: mejor compromiso entre diversificacion y
#     complejidad computacional
#
# TEMPORALIDAD:
#   Periodo de analisis: 2011-01-01 a 2026-01-01
#   - 15 anos de datos: suficiente para multiples ciclos economicos
#   - Incluye crisis financiera (2012), crisis de deuda soberana, pandemia COVID,
#     guerra Ucrania, crisis energetica -> modelo ve comportamiento en diversos
#     regimenes de mercado
#   - Punto de corte final (2026) es una fecha futura para validacion posterior

dax_symbols = [
    "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE",
    "BEI.DE", "BMW.DE", "BNR.DE", "CBK.DE", "CON.DE",
    "DTG.DE", "DB1.DE", "DBK.DE", "DHL.DE",
    "DTE.DE", "EOAN.DE", "FRE.DE", "FME.DE", "HNR1.DE",
    "HEI.DE", "HEN3.DE", "IFX.DE", "MRK.DE", "MBG.DE",
    "MTX.DE", "MUV2.DE", "PAH3.DE", "P911.DE", "QIA.DE",
    "RHM.DE", "RWE.DE", "SAP.DE", "SIE.DE", "ENR.DE",
    "SHL.DE", "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE"
]

# Nombres de empresas correspondientes a tickers del DAX40
dax_company_names = {
    "ADS.DE": "Adidas",
    "AIR.DE": "Airbus",
    "ALV.DE": "Allianz",
    "BAS.DE": "BASF",
    "BAYN.DE": "Bayer",
    "BEI.DE": "Beiersdorf",
    "BMW.DE": "BMW",
    "BNR.DE": "Brenntag",
    "CBK.DE": "Commerzbank",
    "CON.DE": "Continental",
    "DTG.DE": "Daimler Truck",
    "DB1.DE": "Deutsche Boerse",
    "DBK.DE": "Deutsche Bank",
    "DHL.DE": "Deutsche Post DHL",
    "DTE.DE": "Deutsche Telekom",
    "EOAN.DE": "EON",
    "FRE.DE": "Fresenius",
    "FME.DE": "Fresenius Medical Care",
    "HNR1.DE": "Hannover Rueck",
    "HEI.DE": "Heidelberg Materials",
    "HEN3.DE": "Henkel",
    "IFX.DE": "Infineon",
    "MRK.DE": "Merck KGaA",
    "MBG.DE": "Mercedes-Benz Group",
    "MTX.DE": "Merck Finck",
    "MUV2.DE": "Munich Re",
    "PAH3.DE": "Porsche AG",
    "P911.DE": "Porsche SE",
    "QIA.DE": "Qiagen",
    "RHM.DE": "Rheinmetall",
    "RWE.DE": "RWE",
    "SAP.DE": "SAP",
    "SIE.DE": "Siemens",
    "ENR.DE": "Siemens Energy",
    "SHL.DE": "Siemens Healthineers",
    "SY1.DE": "Symrise",
    "VOW3.DE": "Volkswagen",
    "VNA.DE": "Vonovia",
    "ZAL.DE": "Zalando"
}

# Mapeo UNICO ticker -> sector (fuente de verdad para todo el pipeline).
# Si un ticker entra o sale del DAX40, se modifica SOLO aqui.
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

def construir_sectores(tickers_disponibles):
    """Construye dict {sector: [tickers]} filtrando solo tickers presentes."""
    sectores = {}
    for ticker in tickers_disponibles:
        sector = TICKER_SECTOR.get(ticker)
        if sector:
            sectores.setdefault(sector, []).append(ticker)
    return {s: sorted(ts) for s, ts in sorted(sectores.items())}

# =============================================================================
# 2. FEATURES EXGENAS (VARIABLES MACRO DEL MODELO)
# =============================================================================
#
# CONCEPTO GENERAL:
#   Las features exogenas son variables macroeconomicas EXTERNAS a los activos
#   pero que IMPACTAN en su comportamiento. 
#
# SELECCION DE 5 FEATURES MACRO (EURUSD, STOXX50, ORO, VIX, TNX):
#   Se eligen estas 5 porque:
#   1. UNIVERSALIDAD: Afectan a TODOS los activos del DAX porque:
#      - EuroStoxx50 refleja sentimiento europeo general 
#      - El Oro es cobertura de riesgo (inversion defensiva en crisis)
#      - VIX mide la volatilidad implicita (proxy de miedo en mercados globales)
#      - TNX (rendimiento bono 10Y US) captura politica monetaria y tasa de descuento
#   
#   2. INDEPENDENCIA: No hay multicolinealidad severa entre ellas
#      - Oro vs VIX: ambos suben en crisis pero por mecanismos distintos
#      - EuroStoxx vs TNX: correlacion debil (renta variable vs renta fija)
#   
#   3. DISPONIBILIDAD HISTORICA: 15 anos de datos verificados en yfinance
#   
#   4. INTERPRETABILIDAD: El modelo puede aprender relaciones causales:
#      - EuroStoxx sube -> DAX sube (sentimiento macro alcista)
#      - Oro sube -> DAX baja (flight to safety en crisis)
#      - VIX alto -> regimen BEAR mas probable (panic selling)
#      - TNX sube -> DAX baja (descuento de flujos futuros mas agresivo)
#
# NOTA SOBRE EUR/USD:
#   Se incluye el tipo de cambio EUR/USD por su impacto en exportadores
#   alemanes. Disponible en yfinance para el rango temporal requerido.
#   Complementa al EuroStoxx50 aportando informacion de politica monetaria.
#
# ALTERNATIVAS DESCARTADAS:
#    Tipos de interes (BCE): datos de baja frecuencia (cambios mensuales),
#     dificil interpolar a diario. Conceptualmente relevante pero introduce
#     complejidad sin ganancia predictiva clara. Se captura indirectamente
#     via TNX (tipos US como proxy global).
#   
#    Spreads de credito (high yield): Datos muy ruidosos y de baja
#     frecuencia en series historicas largas.
#   
#    Commodity prices (cobre, petroleo): Relevantes para sector especificos
#     (cobre -> electronicos, petroleo -> energia) pero NO universales.
#     Requeriria mucho feature engineering.
#
# ESTRUCTURA DE DATOS:
#   Cada feature macro se SUMA como COLUMNA adicional en dataset_long:
#   Esto permite que se vea en cada timestep [t]:
#   - Precios de las 10-15 empresas [t] (inputs principales)
#   - Contexto macro STOXX50[t], Oro[t], VIX[t], TNX[t] (contexto adicional)
#   El modelo aprendera a ponderar el impacto de cada feature en la prediccion.
#
# Â¿POR QUE COMO COLUMNAS Y NO COMO FILAS?
#   Si metemos las 5 features como "activos adicionales" en dataset_long
#   (es decir, como filas con Ticker='EURUSD', etc.), el modelo veria
#   18 series de activos + 5 series de features = 23 inputs simultaneos.
#   Como COLUMNAS es mas limpio: cada observacion tiene su contexto macro.
#   Es la estructura estandar en econometria y aprendizaje automatico.
#
# Â¿QUE VARIABLES ESPECIFICAS DENTRO DE CADA FEATURE?
#   Se usa SOLO Adj Close (precio ajustado) para cada feature.
#   Razones:
#   - Es la serie mas limpia (yfinance la auto-ajusta)
#   - Otros OHLCV serian redundantes para variables macro
#   - Los retornos (Log_Return) se calculan DESPUES de limpieza (Seccion 9)
#
# Proxy: variable sustituta que se usa cuando quieres medir algo que no puedes
# medir directamente.

macro_symbols = {
    'EURUSD=X':  'Tipo de cambio EUR/USD',
    '^STOXX50E': 'Indice EuroStoxx 50 (sentimiento macroeconomico europeo)',
    'GC=F':      'Oro (proxy de aversion al riesgo e inflacion)',
    # MODIFICACIÓN NUEVA: Añadir VIX y tipos de interes como features macro.
    # El VIX (indice de volatilidad implicita) es el mejor predictor de regimenes
    # de mercado y complementa la deteccion por GMM. Los tipos a 10 anos (TNX)
    # capturan el efecto de politica monetaria sobre valoraciones (DCF).
    '^VIX':      'Indice de Volatilidad VIX (proxy VSTOXX - indicador de miedo)',
    '^TNX':      'Rendimiento Bono 10Y US Treasury (proxy tipos interes globales)',
}

# Nombres cortos para las columnas del dataset final
macro_col_names = {
    'EURUSD=X':  'EURUSD',
    '^STOXX50E': 'STOXX50',
    'GC=F':      'ORO',
    '^VIX':      'VIX',      # MODIFICACIÓN NUEVA: indicador de miedo del mercado
    '^TNX':      'TNX',      # MODIFICACIÓN NUEVA: proxy tipos interes globales
}

# =============================================================================
# 3. BENCHMARKS EXTERNOS (NO ENTRAN AL MODELO COMO FEATURE)
# =============================================================================
#
# PROPOSITO DEL BENCHMARK:
#   El benchmark es la **comparacion de rendimiento** del modelo DL contra
#   un indice "pasivo" (buy-and-hold). Permite responder la pregunta critica:
#   Â¿Proporciona el modelo DL *rentabilidad superior* o simplemente reproduce
#   el indice con mas complejidad?
#
# Â¿POR QUE EL DAX40 COMO BENCHMARK Y NO COMO FEATURE DEL MODELO?
#   1. RIESGO DE MULTICOLINEALIDAD EXTREMO:
#      El DAX40 es el indice de ponderacion de las 40 empresas.
#      Si incluimos DAX40 como input al MLPRegressor que predice 10-15 empresas
#      del DAX, el modelo veria basicamente CONSIGO MISMO (es recursion).
#      Correlacion DAX40 vs empresas individuales: 0.70-0.95 (altisima).
#      
#   2. SESGO DE ANTICIPACION (LOOK-AHEAD BIAS):
#      El DAX40 se calcula con precios simultaneos de las 40 empresas.
#      Si el MLPRegressor conoce el DAX40 en t, esta conociendo indirectamente
#      el precio de otros activos en t, violando la causalidad temporal.
#      
#   3. PERDIDA DE DIVERSIFICACION PREDICTIVA:
#      El objetivo es que el MLPRegressor aprenda patrones INDEPENDIENTES
#      de las empresas (comportamiento idiosincrasico + factores macro).
#      Meter el indice que es la combinacion lineal de las empresas
#      destruye esa independencia: el modelo simplemente replicaria
#      combinaciones lineales conocidas en lugar de descubrir patrones nuevos.
#
# Â¿QUE SON ENTONCES LOS BENCHMARKS?
#   Son metricas de comparacion POST-ENTRENAMIENTO:
#   - Retorno acumulativo del DAX40 durante t_train, t_valid, t_test
#   - Sharpe ratio del DAX40 vs cartera DL
#   - Maxima perdida (max drawdown) comparada
#   - Correlacion con el DAX: Â¿el modelo esta correlacionado o diversificado?
#      (si correlacion=1 el modelo es redundante; si ~0.3-0.6 agrega valor)
#
# ENTRADA AL DATASET:
#   El DAX40 se incluye como **columna adicional** en dataset_long
#   unicamente para facilitar las metricas de comparacion. El modelo DL
#   **NUNCA** ve esta columna durante el entrenamiento (se elimina antes).
#   La presencia en dataset_long es solo para conveniencia de calculos.

benchmark_symbols = {
    '^GDAXI': 'Indice DAX40 (benchmark externo)',
}

benchmark_col_names = {
    '^GDAXI': 'DAX40',
}

# =============================================================================
# 4. DESCARGA DE DATOS
# =============================================================================
#
# METODOLOGIA DE DESCARGA:
#   Se realiza DESCARGA UNICA de todos los datos mediante yfinance.
#   Esto asegura:
#   1. SINCRONIZACION TEMPORAL: Todas las empresas usan el mismo calendario
#      (mismas fechas de trading, mismos precios de cierre cada dia)
#   2. CONSISTENCIA: Mismo ajuste ante dividendos/splits para todos
#   3. EFICIENCIA: Una sola llamada a yfinance, no multiples llamadas
#      que podrian devolver datos ligeramente diferentes por cache.
#      Las empresas que se analizan son las que sobrevivieron hasta
#      2026
#
# RANGO TEMPORAL: 2011-01-01 a 2026-01-01
#   START_DATE = '2011-01-01':
#     - 15 anos de datos historicos
#     - Incluye crisis 2012 (deuda europea), COVID-19 (2020),
#       invasion Ucrania (feb 2022), crisis energetica (2022-2023)
#     - Multiples ciclos economicos: 4 ciclos completos aproximadamente
#     - Baseline de 15 anos es estandar en investigacion financiera
#   
#   END_DATE = '2026-01-01':
#     - Fecha futura para validacion posterior y simulaciones en vivo
#     - No ha ocurrido todavia: el codigo descargara hasta "hoy"
#     - Flexible para re-ejecutar el codigo en anos posteriores
#
# FRECUENCIA: DIARIO ('1d')
#   - Decision correcta para un modelo de optimizacion de cartera
#   - Frecuencia intradiaria (1h, 15min) anadiria ruido sin informacion macro
#   - Frecuencia semanal perderia senal de trading tactica
#
# DATOS DESCARGADOS:
#   - dax_data: OHLCV de 40 empresas DAX (Open, High, Low, Close, Adj Close, Volume)
#   - macro_data: OHLCV de 5 features macro (EUR/USD, EuroStoxx, Oro, VIX, TNX)
#   - benchmark_data: OHLCV del DAX40 (para comparacion final)
#     OJO: se descarga Adj Close de TODOS pero se preservan las columnas
#     OHLC originales durante el joint, para maxima flexibilidad.
#
# Â¿POR QUE auto_adjust=False?
#   auto_adjust=False preserva todos los precios sin ajustes automaticos
#   de yfinance. Esto permite:
#   - Control manual de que columnas ajustar de forma explicita
#   - Transparencia en la transformacion de datos
#   - Reproducibilidad (auto_adjust esta sujeto a cambios en yfinance)

START_DATE = '2011-01-01'
END_DATE   = '2026-01-01'

print("Descargando datos historicos de activos del DAX40...")
dax_data = yf.download(dax_symbols, start=START_DATE, end=END_DATE,
                       interval='1d', auto_adjust=False)

print("\nDescargando features exogenas (variables macro del modelo)...")
macro_data = yf.download(list(macro_symbols.keys()), start=START_DATE, end=END_DATE,
                         interval='1d', auto_adjust=False)

print("\nDescargando benchmark externo (DAX40)...")
benchmark_data = yf.download(list(benchmark_symbols.keys()), start=START_DATE, end=END_DATE,
                              interval='1d', auto_adjust=False)

# =============================================================================
# 5. VALIDACION Y CONTROL DE CALIDAD DE DATOS DESCARGADOS
# =============================================================================
#
# Â¿QUE HACEMOS?
# Despues de descargar datos de yfinance, verificamos:
#   1. Completitud: Â¿Todas las empresas tienen datos? Â¿Cuantos dias de datos?
#   2. Coherencia: Â¿Hay precios negativos o ceros? Â¿Hay saltos descomunales?
#   3. Cobertura temporal: Â¿El periodo [START_DATE, END_DATE] esta completo?
#   4. Consistencia estructural: Â¿Todas las empresas tienen las mismas columnas?
#
# Â¿POR QUE?
# Los datos de yfinance pueden tener:
#   - Descargas incompletas (timeout, servidor)
#   - Datos faltantes sin aviso
#   - Valores extremos por errores de cotizacion (fat-finger errors)
#   - Dividendos que no se ajustaron correctamente (si auto_adjust=False)
# Un control de calidad aqui evita descubrimientos sorpresivos en Secciones 7+.
#
# IMPLICACIONES:
#   - Si una empresa tiene <80% completitud, la eliminamos (no hay suficiente
#     historia para entrenar el MLPRegressor supervisado.
#   - Si hay saltos anormales, los documentamos (pueden ser eventos reales:
#     splits, mergers) o errores de datos (los corregimos).
#
# CONCLUSION:
#   Este paso garantiza que entramos a limpieza (Secciones 6-7) con datos
#   estructuralmente sanos. Reduce sorpresas y aumenta confianza en resultados.

print("\n" + "="*70)
print("SECCION 5: VALIDACION DE CALIDAD DE DATOS DESCARGADOS")
print("="*70)

dax_df       = pd.DataFrame(dax_data)
macro_df     = pd.DataFrame(macro_data)
benchmark_df = pd.DataFrame(benchmark_data)

print(f"\n  [5.1] Control de completitud de datos descargados...")
print(f"  {'Simbolo':<12} {'Filas':>10} {'% Completitud':>15} {' Estado':>15}")
print(f"  {'-'*55}")

# Verificar columnas (Adj Close) por cada ticker
completitud_activos = {}
for ticker in dax_symbols:
    if ('Adj Close', ticker) in dax_df.columns:
        col = ('Adj Close', ticker)
        n_filas = len(dax_df)
        n_nulos = dax_df[col].isna().sum()
        completitud = (n_filas - n_nulos) / n_filas * 100
        completitud_activos[ticker] = completitud
        estado = 'OK [*]' if completitud >= 80 else 'ALERTA [!]'
        print(f"  {ticker:<12} {n_filas:>10} {completitud:>14.1f}% {estado:>15}")

# Identificar activos problematicos
activos_defectuosos = [t for t, c in completitud_activos.items() if c < 80]
if activos_defectuosos:
    print(f"\n  [WARN] {len(activos_defectuosos)} activos con completitud < 80%:")
    for t in activos_defectuosos:
        print(f"    - {t}: {completitud_activos[t]:.1f}%")
    print(f"  Accion: Se excluiran de analisis posterior")
else:
    print(f"\n  [OK] Todos los activos tienen completitud â‰¥ 80%")

# Verificar rango de precios (candidatos a errores)
print(f"\n  [5.2] Diagnostico de precios anormales (fat-finger errors)...")
print(f"  Buscando: precios = 0 o saltos > 50% en un dia")
precios_anomalos = {}
for ticker in dax_symbols:
    if ('Adj Close', ticker) in dax_df.columns:
        col = ('Adj Close', ticker)
        precios = dax_df[col].dropna()
        
        # Precios negativos o cero
        invalidos = (precios <= 0).sum()
        # Saltos diarios > 50%
        cambios = precios.pct_change().abs()
        saltos = (cambios > 0.50).sum()
        
        if invalidos > 0 or saltos > 0:
            precios_anomalos[ticker] = {'invalidos': invalidos, 'saltos': saltos}

if precios_anomalos:
    print(f"  [WARN] {len(precios_anomalos)} activos con anomalias:")
    for t, anom in precios_anomalos.items():
        print(f"    - {t}: {anom['invalidos']} precios invalidos, {anom['saltos']} saltos >50%")
    print(f"  Accion: Inspeccionar manualmente o descartar")
else:
    print(f"  [OK] Sin anomalias detectadas en precios")

print(f"\n  [5.3] Verificacion de variables macro y benchmark...")

# Definimos dónde debería estar cada cosa para buscar con precisión
mapeo_verificacion = [
    ('EURUSD', 'EURUSD=X', macro_df),
    ('STOXX50', '^STOXX50E', macro_df),
    ('ORO', 'GC=F', macro_df),
    ('VIX', '^VIX', macro_df),
    ('TNX', '^TNX', macro_df),
    ('DAX40', '^GDAXI', benchmark_df) # <--- Aquí estaba el detalle, debe mirar en benchmark_df
]

for col_name, col_symbol, df_origen in mapeo_verificacion:
    # Verificamos si la tupla ('Adj Close', col_symbol) existe en las columnas del dataframe origen
    if ('Adj Close', col_symbol) in df_origen.columns:
        datos = df_origen[('Adj Close', col_symbol)].dropna()
        
        if len(datos) > 0:
            completitud_pct = len(datos) / len(df_origen) * 100
            print(f"    {col_name:<10} : {len(datos):>6} obs ({completitud_pct:>4.1f}%)")
        else:
            print(f"    {col_name:<10} : ENCONTRADO PERO VACÍO [!]")
    else:
        print(f"    {col_name:<10} : NO ENCONTRADO EN {col_symbol} [!]")
# GRAFICO 5 -- Completitud de datos por activo
print(f"\n  [5.4] Visualizacion de completitud...")
fig_compl, ax_compl = plt.subplots(figsize=(12, 8))
compl_series = pd.Series(completitud_activos).sort_values(ascending=True)
colores_compl = ['darkgreen' if c >= 95 else 'steelblue' for c in compl_series.values]
ax_compl.barh([a for a in compl_series.index], compl_series.values, 
               color=colores_compl, edgecolor='black', linewidth=0.5)
ax_compl.axvline(80, color='red', linestyle='--', linewidth=1.5, label='Umbral minimo (80%)')
ax_compl.axvline(95, color='darkgreen', linestyle='--', linewidth=1.5, label='Optimo (95%)')
ax_compl.set_xlabel('Completitud (%)', fontsize=10)
ax_compl.set_title('Completitud de datos por activo DAX40\n(verde â‰¥95%, azul 80-95%)', 
                   fontsize=11, fontweight='bold')
ax_compl.set_xlim([0, 105])
ax_compl.legend(fontsize=9, loc='lower right')
ax_compl.grid(axis='x', alpha=0.3)
for i, v in enumerate(compl_series.values):
    ax_compl.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=8)
plt.tight_layout()
ruta_compl = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          '5_completitud_activos.png')
plt.savefig(ruta_compl, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Grafica guardada: {ruta_compl}")

print(f"\n  CONCLUSION SECCION 5:")
print(f"  [OK] Datos descargados: {len(dax_df)} observaciones")
print(f"  [OK] Completitud media: {np.mean(list(completitud_activos.values())):.1f}%")
print(f"  [OK] Periodo: {dax_df.index.min().date()} a {dax_df.index.max().date()}")
print(f"  [OK] Dataset apto para procesamiento posterior")

# =============================================================================
# 5.1 CONVERSION A DATAFRAMES
# =============================================================================
print("\n" + "="*70)
print("SECCION 5.1: CONVIRTIENDO DATOS A DATAFRAMES DE PANDAS...")
print("="*70)

dax_df       = pd.DataFrame(dax_data)
macro_df     = pd.DataFrame(macro_data)
benchmark_df = pd.DataFrame(benchmark_data)

"""
# =============================================================================
# EXPORTAR DATAFRAMES PUROS A EXCEL
# =============================================================================
print("\nExportando dataframes puros a Excel...")
ruta_excel_puros = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 '0_dataframes_puros_yfinance.xlsx')

with pd.ExcelWriter(ruta_excel_puros, engine='openpyxl') as writer:
    dax_df.to_excel(writer, sheet_name='DAX40_Activos')
    macro_df.to_excel(writer, sheet_name='Macro_Features')
    benchmark_df.to_excel(writer, sheet_name='Benchmark_DAX40')

print(f"[OK] Archivos guardados en: {ruta_excel_puros}")
"""

# =============================================================================
# 6. JOIN + RESHAPE + INTEGRACIN DE MACRO Y BENCHMARK COMO COLUMNAS
# =============================================================================
# En esta seccin hacemos TRES pasos simples:
#
#   PASO 1 - Recortar al horizonte de anlisis (2015-)
#
#   PASO 2 - Reshape: pasar activos DAX a formato largo (una fila por fecha-ticker)
#            Macro y benchmark se extraen como columnas adicionales.
#
#   PASO 3 - Crear Primary Key (PK) que identifique unvocamente
#            cada observacin: PK = "YYYY-MM-DD_TICKER"
#
# Por qu macro y benchmark como columnas?
#   Son variables independientes (features exgenas) que queremos usar
#   como inputs del modelo. Cada fila ya tiene el contexto macro del mismo da.
#
#   Estructura final:
#   PK | Date | Ticker | Open | High | Low | Close | Adj Close | Volume 

print("\n" + "="*70)
print("SECCIN 6: JOIN + RESHAPE + MACRO/BENCHMARK COMO COLUMNAS")
print("="*70)

# PASO 1: Recortar al horizonte de anlisis
dax_join       = dax_df.loc['2015-01-01':].copy()
macro_join     = macro_df.loc['2015-01-01':].copy()
benchmark_join = benchmark_df.loc['2015-01-01':].copy()

print(f"\n  [6.1] Shapes antes del join:")
print(f"    dax_join       : {dax_join.shape}")
print(f"    macro_join     : {macro_join.shape}")
print(f"    benchmark_join : {benchmark_join.shape}")

# PASO 2a: Reshape de activos DAX a formato largo
# stack(level=1) dobla el nivel de ticker de las columnas a filas
dataset_long = dax_join.stack(level=1, future_stack=True)
dataset_long.index.names = ['Date', 'Ticker']
dataset_long = dataset_long.reset_index()



print(f"\n  [6.2] Shape dataset DAX formato largo: {dataset_long.shape}")

# PASO 2b: Preparar macro y benchmark como columnas por fecha
# Extraemos SOLO Adj Close de macro y benchmark (sin LogReturn)
# Los rendimientos se calcularn en Seccin 9 despus de la limpieza de datos
def extraer_adj_close(df_yf, ticker, nombre_col):
    """
    Extrae Adj Close de un ticker en DataFrame MultiIndex de yfinance,
    aplica ffill+bfill para nulos de calendario.
    """
    if ('Adj Close', ticker) in df_yf.columns:
        serie = df_yf[('Adj Close', ticker)].copy()
    elif 'Adj Close' in df_yf.columns:
        serie = df_yf['Adj Close'].copy()
    else:
        raise KeyError(f"No se encontr Adj Close para {ticker}")

    #serie = serie.ffill().bfill()

    resultado = pd.DataFrame({
        f'{nombre_col}_Close': serie,
    })
    resultado.index.name = 'Date'
    return resultado

# Extraer macro
dfs_macro = []
for ticker, nombre in macro_col_names.items():
    try:
        df_m = extraer_adj_close(macro_join, ticker, nombre)
        dfs_macro.append(df_m)
    except KeyError as e:
        print(f"    - {ticker:<12} -> error: {e}")

# Extraer benchmark
dfs_bench = []
for ticker, nombre in benchmark_col_names.items():
    try:
        df_b = extraer_adj_close(benchmark_join, ticker, nombre)
        dfs_bench.append(df_b)
    except KeyError as e:
        print(f"    - {ticker:<12} -> error: {e}")

# Combinar columnas externas
df_externas = pd.concat(dfs_macro + dfs_bench, axis=1)
# IMPORTANTE: Usar reindex() con el indice de dax_join para mantener TODAS las fechas
# de las empresas, aunque macro/DAX40 no tengan datos para algunas fechas.
# Esto genera NaNs para las fechas que no existen en df_externas,
# que se tratarn apropiadamente en Seccin 7.6
df_externas = df_externas.reindex(dax_join.index)
df_externas.index = pd.to_datetime(df_externas.index)
df_externas.index.name = 'Date'


# Join del dataset largo con columnas externas
# Ambos ya tienen Date como tipo date

#COMENTADO PRUEBA
df_externas_reset = df_externas.reset_index()

dataset_long = dataset_long.merge(
    df_externas_reset,
    on='Date',
    how='left'
)

# Reordenar columnas
cols_id        = ['Date', 'Ticker']
cols_ohlcv     = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
cols_macro_ext = [c for c in dataset_long.columns
                  if any(c.startswith(n) for n in macro_col_names.values())]
cols_bench_ext = [c for c in dataset_long.columns
                  if any(c.startswith(n) for n in benchmark_col_names.values())]
cols_resto     = [c for c in dataset_long.columns
                  if c not in cols_id + cols_ohlcv + cols_macro_ext + cols_bench_ext]

dataset_long = dataset_long[
    cols_id + cols_ohlcv + cols_resto + cols_macro_ext + cols_bench_ext
]

# PASO 3: Crear Primary Key
# Se crea DESPUS de que Date sea tipo date
dataset_long['PK'] = (
    dataset_long['Date'].astype(str) + '_' + dataset_long['Ticker']
)

# Reposicionar PK al inicio
cols_final = ['PK'] + [c for c in dataset_long.columns if c != 'PK']
dataset_long = dataset_long[cols_final]

print(f"\n  [6.3] Despus de join + reshape + PK:")
print(f"    Shape dataset_long: {dataset_long.shape}")
print(f"    Columnas: {len(dataset_long.columns)}")
print(f"    Rango fechas: {dataset_long['Date'].min()} -> {dataset_long['Date'].max()}")

# CONVERTIR Date INMEDIATAMENTE despues del reset_index
dataset_long['Date'] = pd.to_datetime(dataset_long['Date']).dt.date

# Convertirmos los campos a su formato optimo
dataset_long['Ticker'] = dataset_long['Ticker'].astype(str)
dataset_long['Adj Close'] = pd.to_numeric(dataset_long['Adj Close'], errors='coerce')
dataset_long['PK'] = dataset_long['PK'].astype(str)

# Guardado para inspeccin estructural
ruta_excel = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '1_dataset_long_con_macro_columnas.xlsx'
)
try:
    dataset_long.to_csv(ruta_excel.replace('.xlsx', '.csv'), index=False)
    dataset_long.to_excel(ruta_excel, index=False)
    print(f"\n  [OK] Dataset guardado: {ruta_excel}")
except PermissionError:
    print(f"\n  [ADVERTENCIA] No se pudo guardar Excel (archivo bloqueado). Continuando...")
except Exception as e:
    print(f"\n  [ADVERTENCIA] Error al guardar Excel: {e}. Continuando...")
print(f"\n  Seccin 6 completada: JOIN + RESHAPE + MACRO/BENCHMARK + PK")


# =============================================================================
# 6B. ANALISIS SECTORIAL PREVIO A LA LIMPIEZA
# =============================================================================
#
# Â¿QUE HACEMOS?
# Identificamos y cuantificamos la composicion sectorial del dataset.
# Los mercados funcionan por sectores: cuando sube la tecnologia, bajan los
# bancos (normalmente). El MLPRegressor debe capturar esta estructura.
#
# Â¿POR QUE?
# Un dataset sesgado sectorialmente (p.ej. 60% tecnologia) produce un modelo
# que aprende a predecir "ese" sector bien pero falla en otros.
# La diversificacion sectorial garantiza que el modelo aprende patrones
# robustos y generalizables a multiples regimenes economicos.
#
# IMPLICACIONES:
# Este analisis justifica nuestras selecciones posteriors (Seccion 10):
# elegimos 1 empresa por cluster (diversificacion estadistica) pero TAMBIEN
# verificamos cobertura sectorial (diversificacion economica).
#
# CONCLUSION:
# Un dataset sectorialmente diverso evita overfitting a un unico sector.

print("\n" + "="*70)
print("ANALISIS SECTORIAL - COMPOSICION DEL DAX40")
print("="*70)

# Extraer tickers unicos del dataset_long
tickers_empresas = sorted(dataset_long['Ticker'].unique())
tickers_candidatos = sorted(tickers_empresas)

# SECTORES se construye dinamicamente a partir de TICKER_SECTOR (seccion 1)
# filtrando solo los tickers que existen en el dataset.
SECTORES = construir_sectores(tickers_candidatos)

print(f"\n  Distribucion sectorial de activos candidatos:")
print(f"  {'Sector':<18} {'Empresas':>10} {'% del DAX':>12}")
print(f"  {'-'*42}")

for nombre_sector, empresas_sector in sorted(SECTORES.items()):
    en_dataset = [e for e in empresas_sector if e in tickers_candidatos]
    pct = len(en_dataset) / len(tickers_candidatos) * 100
    print(f"  {nombre_sector:<18} {len(en_dataset):>10} {pct:>11.1f}%")

# Analisis de concentracion sectorial
sector_counts = pd.Series({
    s: len([e for e in es if e in tickers_candidatos])
    for s, es in SECTORES.items()
})
sector_counts = sector_counts[sector_counts > 0].sort_values(ascending=False)

print(f"\n  Observaciones:")
print(f"  - Presentes {len(sector_counts)} de {len(SECTORES)} sectores")
print(f"  - Sector mas representado: {sector_counts.index[0]} ({sector_counts.values[0]} empresas)")
print(f"  - Concentracion Herfindahl: {(sector_counts / sector_counts.sum()).pow(2).sum():.3f}")
if (sector_counts / sector_counts.sum()).pow(2).sum() < 0.2:
    print(f"    -> Diversificacion BUENA (<0.2)")
elif (sector_counts / sector_counts.sum()).pow(2).sum() < 0.3:
    print(f"    -> Diversificacion MODERADA (0.2-0.3)")
else:
    print(f"    -> Concentracion ALTA (>0.3) -- seleccion posterior debe mejorar esto")

# GRAFICO 6B -- Distribucion sectorial del DAX40
print(f"\n  [6B.2] Visualizacion de distribucion sectorial...")
herfindahl = (sector_counts / sector_counts.sum()).pow(2).sum()
colores_pie = plt.cm.Blues(np.linspace(0.35, 0.85, len(sector_counts)))

# Figura 6B-a: Pie chart de distribucion por sector
fig_6ba, ax_6ba = plt.subplots(figsize=(9, 7))
wedges, texts, autotexts = ax_6ba.pie(
    sector_counts.values,
    labels=sector_counts.index,
    autopct='%1.1f%%',
    colors=colores_pie,
    startangle=45,
    textprops={'fontsize': 9}
)
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(8)
ax_6ba.set_title(
    f'Distribucion de Empresas DAX40 por Sector\n'
    f'(Indice Herfindahl: {herfindahl:.3f} — {len(sector_counts)} sectores)',
    fontsize=12, fontweight='bold')
plt.tight_layout()
ruta_6ba = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '6b_distribucion_sectorial_pie.png')
plt.savefig(ruta_6ba, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Grafica guardada: {ruta_6ba}")

# Figura 6B-b: Barplot de conteos por sector
fig_6bb, ax_6bb = plt.subplots(figsize=(9, 6))
ax_6bb.barh(sector_counts.index, sector_counts.values,
            color=colores_pie, edgecolor='black', linewidth=0.5)
ax_6bb.set_xlabel('Numero de empresas', fontsize=10)
ax_6bb.set_title(
    f'Conteo de Empresas DAX40 por Sector\n'
    f'(Sector dominante: {sector_counts.index[0]} con {sector_counts.values[0]} empresas)',
    fontsize=12, fontweight='bold')
ax_6bb.grid(axis='x', alpha=0.3)
for i, (sector, count) in enumerate(sector_counts.items()):
    ax_6bb.text(count + 0.1, i, str(int(count)), va='center', fontsize=9)
plt.tight_layout()
ruta_6bb = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '6b_distribucion_sectorial_barras.png')
plt.savefig(ruta_6bb, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Grafica guardada: {ruta_6bb}")

# GRAFICO 6 -- Serie temporal de precios de candidatos
print(f"\n  [6.3] Visualizacion de serie temporal de precios candidatos...")
fig_templ, ax_templ = plt.subplots(figsize=(16, 8))
tickers_para_plot = dax_symbols[:15] if len(dax_symbols) > 15 else dax_symbols
for ticker in tickers_para_plot:
    ticker_data = dataset_long[dataset_long['Ticker'] == ticker].sort_values('Date')
    ax_templ.plot(ticker_data['Date'], ticker_data['Adj Close'], label=ticker, alpha=0.7, linewidth=1)
ax_templ.set_xlabel('Fecha', fontsize=10)
ax_templ.set_ylabel('Precio de cierre ajustado (EUR)', fontsize=10)
ax_templ.set_title(f'Serie Temporal de Precios - {len(tickers_para_plot)} Activos DAX\n(Periodo: {dataset_long["Date"].min()} a {dataset_long["Date"].max()})',
                   fontsize=12, fontweight='bold')
ax_templ.legend(ncol=3, fontsize=7, loc='upper left')
ax_templ.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
ruta_templ = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          '6_serie_temporal_precios.png')
plt.savefig(ruta_templ, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Grafica guardada: {ruta_templ}")

# =============================================================================
# 7. LIMPIEZA DE DATOS: TRATAMIENTO DE VALORES NULOS (NaN)
# =============================================================================
# Qu tipos de nulos buscamos?
#   1. Nulos por falta de datos (activos que no cotizaban en ciertas fechas)
#   2. Nulos por das no laborables (festivos, fines de semana)
#   3. Nulos por errores de descarga o gaps puntuales (causas tcnicas)
#   4. Nulos por ventanas rolling (primeros 20/60 das calculando volatilidad)
#
# Estrategia de eliminacin/interpolacin:
#   1. Eliminar activos CON HISTORIAL INSUFICIENTE (primer nulo <= 2015-02-01)
#   2. Eliminar activos CON GAPS LARGOS (> 2 das consecutivos)
#   3. Aplicar INTERPOLACIN AVANZADA a nulos residuales:
#      - Forward Fill (limit=2): para gaps de 1-2 das
#      - Spline Cubica: para gaps de 3-10 das
#      COMPARATIVA: evaluar cual preserva mejor la integridad de datos

# Snapshot pre-limpieza para visualizaciones comparativas (Seccion 12)
df_snap_antes_limpieza = dataset_long.copy()

print("\n" + "="*70)
print("SECCIN 7: LIMPIEZA DE DATOS - TRATAMIENTO DE NULOS")
print("="*70)

# Ahora tickers_externos ya no son filas del dataset - son columnas.
# Todos los tickers en dataset_long son empresas del DAX.
tickers_empresas  = sorted(dataset_long['Ticker'].unique())
cols_precio       = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
cols_precio_exist = [c for c in cols_precio if c in dataset_long.columns]

# 
# 7.1 DIAGNSTICO INICIAL DE NULOS
# 
print("\n  [7.1] Diagnstico inicial de nulos por ticker:")
print(f"  {'Ticker':<15} {'Nulos totales':>15} {'% nulos':>10} {'1 fecha nulo':>15} {'Gap mx':>10}")
print("  " + "-"*67)

diagnostico_nulos = []
for ticker in tickers_empresas:
    ticker_data   = dataset_long[dataset_long['Ticker'] == ticker].sort_values('Date')
    mask_nulos    = ticker_data[cols_precio_exist].isna().any(axis=1)
    nulos_totales = mask_nulos.sum()
    pct_nulos     = nulos_totales / len(ticker_data) * 100

    primera_fecha_nan = ticker_data[mask_nulos]['Date'].min() if mask_nulos.any() else None

    grupos  = mask_nulos.astype(int).groupby((~mask_nulos).cumsum())
    gap_max = int(grupos.sum().max()) if mask_nulos.any() else 0

    diagnostico_nulos.append({
        'ticker':        ticker,
        'nulos_totales': nulos_totales,
        'pct_nulos':     round(pct_nulos, 2),
        'primera_fecha': primera_fecha_nan,
        'gap_max':       gap_max,
    })
    print(f"  {ticker:<15} {nulos_totales:>15} {pct_nulos:>9.2f}% "
          f"{str(primera_fecha_nan):>15} {gap_max:>10}")

diagnostico_df = pd.DataFrame(diagnostico_nulos).set_index('ticker')

# 
# 7.2 CRITERIO 1 - ELIMINAR ACTIVOS CON HISTORIAL INSUFICIENTE
# 
FECHA_CORTE_HISTORIAL = pd.Timestamp('2015-02-01').date()

tickers_historial_insuf = [
    t for t in tickers_empresas
    if (
        pd.notna(diagnostico_df.loc[t, 'primera_fecha']) and
        pd.Timestamp(diagnostico_df.loc[t, 'primera_fecha']).date() <= FECHA_CORTE_HISTORIAL
    )
]

print(f"\n  [7.2] Activos eliminados por historial insuficiente "
      f"(primer nulo <= {FECHA_CORTE_HISTORIAL}):")
if tickers_historial_insuf:
    for t in tickers_historial_insuf:
        print(f"    - {t:<12} - primer nulo: {diagnostico_df.loc[t, 'primera_fecha']}")
else:
    print("    [OK] Ninguno eliminado por este criterio")

# 
# 7.3 TRATAMIENTO DIFERENCIADO DE GAPS
# - Gaps 1-2 nulos consecutivos: Rellenar con ffill/bfill (rapido y efectivo)
# - Gaps > 2 nulos consecutivos: Aplicar comparativa de tecnicas (KNN vs Spline Cubica)
# 
MAX_NULOS_PEQUENO = 2  # Threshold para separar pequenos gaps de grandes gaps

tickers_a_eliminar = tickers_historial_insuf  # Solo eliminamos por historial insuficiente
dataset_long       = dataset_long[
    ~dataset_long['Ticker'].isin(tickers_a_eliminar)
].reset_index(drop=True)

tickers_candidatos = [t for t in tickers_empresas if t not in tickers_a_eliminar]
print(f"\n  [7.3] Clasificacion de gaps para tratamiento diferenciado:")
print(f"    Empresas candidatas: {len(tickers_candidatos)}")

# Separar tickers segun tamano de gaps
tickers_gap_pequeno = []  # gaps 1-2 nulos
tickers_gap_grande = []   # gaps > 2 nulos

for ticker in tickers_candidatos:
    gap_max = diagnostico_df.loc[ticker, 'gap_max']
    if gap_max <= MAX_NULOS_PEQUENO and gap_max > 0:
        tickers_gap_pequeno.append(ticker)
    elif gap_max > MAX_NULOS_PEQUENO:
        tickers_gap_grande.append(ticker)

print(f"    - Gaps pequenos (1-{MAX_NULOS_PEQUENO}): {len(tickers_gap_pequeno)} empresas")
if tickers_gap_pequeno:
    for t in tickers_gap_pequeno:
        print(f"      {t:<12} - gap maximo: {diagnostico_df.loc[t, 'gap_max']} dias")

print(f"    - Gaps grandes (>{MAX_NULOS_PEQUENO}): {len(tickers_gap_grande)} empresas")
if tickers_gap_grande:
    for t in tickers_gap_grande:
        print(f"      {t:<12} - gap maximo: {diagnostico_df.loc[t, 'gap_max']} dias")

# 
# 7.3.1 TRATAR GAPS PEQUENOS CON FFILL/BFILL
# 
print(f"\n  [7.3.1] Rellenando gaps pequenos (<={MAX_NULOS_PEQUENO}) con ffill/bfill...")
for ticker in tickers_gap_pequeno:
    mask = dataset_long['Ticker'] == ticker
    # Aplicar ffill (forward fill) y luego bfill (backward fill) para los extremos
    dataset_long.loc[mask, cols_precio_exist] = (
        dataset_long.loc[mask, cols_precio_exist].ffill().bfill()
    )
print(f"    [OK] {len(tickers_gap_pequeno)} empresas procesadas")

# 
# 7.4 COMPARATIVA DE TECNICAS PARA GAPS GRANDES (>2 NULOS CONSECUTIVOS)
# Comparamos Spline Cubica vs KNN Imputation
# Forward Fill ya fue aplicado a gaps pequenos en 7.3.1
# 

try:
    from scipy.interpolate import CubicSpline
    from sklearn.impute import KNNImputer
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-learn'])
    from scipy.interpolate import CubicSpline
    from sklearn.impute import KNNImputer

if len(tickers_gap_grande) > 0:

    print(f"\n  [7.4] COMPARATIVA DE TECNICAS (gaps > {MAX_NULOS_PEQUENO} nulos)...")
    print(f"    Evaluando 2 tecnicas: Spline Cubica, KNN Imputation")
    print(f"    Empresas afectadas: {len(tickers_gap_grande)}")

    # Crear 2 copias del dataset para probar cada tecnica
    # Comenzamos con el dataset donde ya hemos rellenado gaps pequenos
    dataset_spline = dataset_long.copy()  # Spline Cubica
    dataset_knn = dataset_long.copy()     # KNN Imputation

    # ===== TECNICA 1: SPLINE CUBICA =====
    print(f"\n    [TECNICA 1] Spline Cubica (por ticker)...")
    for ticker in tickers_gap_grande:
        mask_ticker = dataset_spline['Ticker'] == ticker
        idx_ticker = dataset_spline[mask_ticker].index
        
        for col in cols_precio_exist:
            datos = dataset_spline.loc[idx_ticker, col].values
            idx_vals = np.arange(len(datos))
            
            mask_nulos = np.isnan(datos)
            if mask_nulos.sum() > 0:
                idx_validos = idx_vals[~mask_nulos]
                datos_validos = datos[~mask_nulos]
                
                if len(idx_validos) > 2:  # Minimo 3 puntos para spline
                    try:
                        cs = CubicSpline(idx_validos, datos_validos, bc_type="natural")
                        datos_interp = cs(idx_vals)
                        dataset_spline.loc[idx_ticker, col] = datos_interp
                    except:
                        # Si falla, aplicar ffill como fallback
                        dataset_spline.loc[idx_ticker, col] = dataset_spline.loc[idx_ticker, col].ffill().bfill()

    # Aplicar ffill final para nulos residuales de spline
    # Usamos transform en lugar de apply para mantener la estructura del DataFrame
    for ticker in tickers_gap_grande:
        mask = dataset_spline['Ticker'] == ticker
        dataset_spline.loc[mask, cols_precio_exist] = (
            dataset_spline.loc[mask, cols_precio_exist].ffill().bfill()
        )

    nulos_spline = dataset_spline[cols_precio_exist].isna().sum().sum()

    # ===== TECNICA 2: KNN IMPUTATION =====
    print(f"    [TECNICA 2] KNN Imputation (k=5)...")
    try:
        knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
        dataset_knn[cols_precio_exist] = knn_imputer.fit_transform(
            dataset_knn[cols_precio_exist]
        )
        nulos_knn = dataset_knn[cols_precio_exist].isna().sum().sum()
    except Exception as e:
        print(f"      [ERROR] KNN fallo: {e}. Usando Spline como fallback.")
        dataset_knn = dataset_spline.copy()
        nulos_knn = nulos_spline
else:
    print(f"\n  [7.4] No hay gaps > {MAX_NULOS_PEQUENO}. Saltando comparativa de tecnicas.")
    dataset_spline = dataset_long.copy()
    dataset_knn = dataset_long.copy()
    nulos_spline = dataset_long[cols_precio_exist].isna().sum().sum()
    nulos_knn = nulos_spline

# 
# 7.4.1 COMPARATIVA ESTADISTICA (si hay gaps grandes)
# 
if len(tickers_gap_grande) > 0:
    print(f"\n  [7.4.1] Comparativa estadistica de las 2 tecnicas...")

    # Seleccionar un ticker con gap grande para analisis
    ticker_ejemplo = tickers_gap_grande[0]
    mask_ej = dataset_long['Ticker'] == ticker_ejemplo

    # Extraer retornos
    col_retorno = 'Adj Close'
    ret_original = np.log(dataset_long.loc[mask_ej, col_retorno] / dataset_long.loc[mask_ej, col_retorno].shift(1))
    ret_spline = np.log(dataset_spline.loc[mask_ej, col_retorno] / dataset_spline.loc[mask_ej, col_retorno].shift(1))
    ret_knn = np.log(dataset_knn.loc[mask_ej, col_retorno] / dataset_knn.loc[mask_ej, col_retorno].shift(1))

    print(f"\n    Comparativa de estadisticas (Retornos Log - Ticker: {ticker_ejemplo}):")
    print(f"    {'Metrica':<20} {'Original':>12} {'Spline':>12} {'KNN':>12}")
    print(f"    " + "-"*50)

    metricas = {
        'Media': lambda x: np.nanmean(x),
        'Std': lambda x: np.nanstd(x),
        'Skewness': lambda x: stats.skew(x[~np.isnan(x)]) if len(x[~np.isnan(x)]) > 2 else np.nan,
        'Kurtosis': lambda x: stats.kurtosis(x[~np.isnan(x)]) if len(x[~np.isnan(x)]) > 2 else np.nan,
    }

    comparativa_stats = {}
    for nombre, func in metricas.items():
        orig = func(ret_original.values)
        spl = func(ret_spline.values)
        knn_val = func(ret_knn.values)
        
        print(f"    {nombre:<20} {orig:>12.4f} {spl:>12.4f} {knn_val:>12.4f}")
        comparativa_stats[nombre] = {'Original': orig, 'Spline': spl, 'KNN': knn_val}
else:
    comparativa_stats = {}

# 
# 7.4.2 STATIONARITY TEST (ADF) PARA CADA TECNICA (si hay gaps grandes)
# 
if len(tickers_gap_grande) > 0:
    print(f"\n  [7.4.2] Tests de estacionariedad (ADF) por tecnica...")
    print(f"    ADF test sobre retornos log del ticker: {ticker_ejemplo}")

    def adf_test_simple(serie):
        """Ejecutar ADF test y retornar p-value"""
        try:
            result = adfuller(serie[~np.isnan(serie)], autolag='AIC')
            return result[1]  # p-value
        except:
            return np.nan

    pvalue_original = adf_test_simple(ret_original.values)
    pvalue_spline = adf_test_simple(ret_spline.values)
    pvalue_knn = adf_test_simple(ret_knn.values)

    print(f"    {'Tecnica':<20} {'p-value':>12} {'Estacionario (p<0.05)':>25}")
    print(f"    " + "-"*60)
    print(f"    {'Original':<20} {pvalue_original:>12.4f} {'SI' if pvalue_original < 0.05 else 'NO':>25}")
    print(f"    {'Spline Cubica':<20} {pvalue_spline:>12.4f} {'SI' if pvalue_spline < 0.05 else 'NO':>25}")
    print(f"    {'KNN Imputation':<20} {pvalue_knn:>12.4f} {'SI' if pvalue_knn < 0.05 else 'NO':>25}")

    # 
    # 7.4.3 CRITERIO DE SELECCION
    # Elegir la tecnica que: (1) tenga mas estacionariedad, (2) preserve mejor estadisticas
    # 
    print(f"\n  [7.4.3] Seleccion de tecnica optima...")

    # Puntuacion: favorece tecnicas que mantienen estacionariedad y estadisticas cercanas al original
    if comparativa_stats and 'Std' in comparativa_stats:
        scores = {
            'Spline': abs(pvalue_spline - pvalue_original) + abs(comparativa_stats['Std']['Spline'] - comparativa_stats['Std']['Original']) / comparativa_stats['Std']['Original'],
            'KNN': abs(pvalue_knn - pvalue_original) + abs(comparativa_stats['Std']['KNN'] - comparativa_stats['Std']['Original']) / comparativa_stats['Std']['Original'],
        }

        tecnica_elegida = min(scores, key=scores.get)
        print(f"    Scores de desviacion (menor = mejor):")
        for tecnica, score in sorted(scores.items(), key=lambda x: x[1]):
            print(f"      {tecnica:<20} : {score:.4f}")

        print(f"\n    [ELEGIDA] {tecnica_elegida}")
        print(f"    Justificacion: Preserva mejor estacionariedad y estadisticas de retornos")
    else:
        tecnica_elegida = 'Spline'
        print(f"    [ELEGIDA] {tecnica_elegida} (por defecto)")

    # Asignar dataset segun tecnica elegida
    if tecnica_elegida == 'Spline':
        dataset_long = dataset_spline.copy()
        nulos_final = nulos_spline
    else:
        dataset_long = dataset_knn.copy()
        nulos_final = nulos_knn
else:
    # Si no hay gaps grandes, el dataset_long ya esta completamente rellenado
    nulos_final = dataset_long[cols_precio_exist].isna().sum().sum()

# 
# 7.5 VERIFICACIN FINAL DE NULOS
# 
print(f"\n  [7.5] Verificacion final de nulos (post-interpolacion):")
nulos_finales = dataset_long.groupby('Ticker').apply(
    lambda x: x[cols_precio_exist].isna().sum().sum()
)
nulos_con_problemas = nulos_finales[nulos_finales > 0]

if len(nulos_con_problemas) == 0:
    print("    [OK] Dataset limpio - sin nulos en columnas de precio")
else:
    print("    - Tickers con nulos residuales:")
    print(nulos_con_problemas.to_string())

print(f"\n  Resumen limpieza nulos:")
if len(tickers_gap_grande) > 0:
    print(f"    Estrategia              : ffill/bfill (gaps 1-{MAX_NULOS_PEQUENO}) + {tecnica_elegida} (gaps >{MAX_NULOS_PEQUENO})")
else:
    print(f"    Estrategia              : ffill/bfill (gaps 1-{MAX_NULOS_PEQUENO})")
print(f"    Gaps pequenos rellenados: {len(tickers_gap_pequeno)}")
print(f"    Gaps grandes comparados : {len(tickers_gap_grande)}")
print(f"    Nulos finales           : {dataset_long[cols_precio_exist].isna().sum().sum()}")
print(f"    Activos eliminados      : {len(tickers_a_eliminar)}")
print(f"    Activos candidatos final: {len(tickers_candidatos)}")
print(f"    Shape dataset           : {dataset_long.shape}")

ruta_excel = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '2.1_dataset_long_sin_nulos.xlsx'
)
try:
    dataset_long.to_excel(ruta_excel, index=False)
    print(f"\n  Dataset guardado: {ruta_excel}")
except PermissionError:
    print(f"\n  [ADVERTENCIA] No se pudo guardar Excel (archivo bloqueado). Continuando...")
except Exception as e:
    print(f"\n  [ADVERTENCIA] Error al guardar Excel: {e}. Continuando...")

# 
# 7.6 TRATAMIENTO DE NULOS EN VARIABLES MACRO
# Estrategia identica a la de tickers: 
#   - Gaps 1-2 nulos: ffill/bfill
#   - Gaps > 2 nulos: Comparativa KNN vs Spline Cubica
# IMPORTANTE: NO se elimina ninguna variable macro - todas se rellenan sin excepcion
# 
print(f"\n" + "="*70)
print(f"SECCION 7.6: TRATAMIENTO DE NULOS - VARIABLES MACRO")
print(f"="*70)

# Identificar columnas de variables macro
# NOTA: Los LogReturn se calcularn en Seccin 9, aqu solo existen _Close
cols_macro = ['EURUSD_Close', 'STOXX50_Close', 'ORO_Close', 'VIX_Close', 'TNX_Close', 'DAX40_Close']
cols_macro_exist = [c for c in cols_macro if c in dataset_long.columns]

# 
# 7.6.1 DIAGNOSTICO INICIAL DE NULOS EN MACRO
# 
print(f"\n  [7.6.1] Diagnostico de nulos en variables macro:")
print(f"  {'Variable':<20} {'Nulos totales':>15} {'% nulos':>10} {'Gap maximo':>10}")
print(f"  " + "-"*60)

diagnostico_macro = []
for col in cols_macro_exist:
    nulos = dataset_long[col].isna().sum()
    pct = nulos / len(dataset_long) * 100
    
    # Calcular gap maximo en esta columna
    mask = dataset_long[col].isna()
    if mask.any():
        grupos = mask.astype(int).groupby((~mask).cumsum())
        gap_max = int(grupos.sum().max()) if mask.any() else 0
    else:
        gap_max = 0
    
    diagnostico_macro.append({
        'variable': col,
        'nulos': nulos,
        'pct': pct,
        'gap_max': gap_max,
    })
    print(f"  {col:<20} {nulos:>15} {pct:>9.2f}% {gap_max:>10}")

df_diagnostico_macro = pd.DataFrame(diagnostico_macro).set_index('variable')

# 
# 7.6.1b FILTRO: ELIMINAR VARIABLES MACRO CON DEMASIADOS NULOS
# 
MAX_PCT_NULOS_MACRO = 5  # Umbral maximo de % nulos permitido
cols_macro_eliminar = df_diagnostico_macro[df_diagnostico_macro['pct'] > MAX_PCT_NULOS_MACRO].index.tolist()

if cols_macro_eliminar:
    print(f"\n  [7.6.1b] FILTRO DE CALIDAD: eliminando variables con >{MAX_PCT_NULOS_MACRO}% nulos")
    for col in cols_macro_eliminar:
        pct_col = df_diagnostico_macro.loc[col, 'pct']
        print(f"    [ELIMINADA] {col:<20} ({pct_col:.2f}% nulos - supera umbral {MAX_PCT_NULOS_MACRO}%)")
        dataset_long.drop(columns=[col], inplace=True)
    cols_macro_exist = [c for c in cols_macro_exist if c not in cols_macro_eliminar]
    df_diagnostico_macro = df_diagnostico_macro.drop(cols_macro_eliminar)
    print(f"    [OK] {len(cols_macro_eliminar)} variable(s) eliminada(s). Quedan {len(cols_macro_exist)} variables macro.")
else:
    print(f"\n  [7.6.1b] FILTRO DE CALIDAD: todas las variables tienen <={MAX_PCT_NULOS_MACRO}% nulos. No se elimina ninguna.")

# 
# 7.6.2 CLASIFICACION SEGUN GAP
# 
print(f"\n  [7.6.2] Clasificacion de variables segun tamano de gap:")

cols_macro_gap_pequeno = []
cols_macro_gap_grande = []

for col in cols_macro_exist:
    gap = df_diagnostico_macro.loc[col, 'gap_max']
    if gap > 0 and gap <= MAX_NULOS_PEQUENO:
        cols_macro_gap_pequeno.append(col)
    elif gap > MAX_NULOS_PEQUENO:
        cols_macro_gap_grande.append(col)

print(f"    - Gaps pequenos (1-{MAX_NULOS_PEQUENO}): {len(cols_macro_gap_pequeno)} variables")
if cols_macro_gap_pequeno:
    for col in cols_macro_gap_pequeno:
        print(f"      {col:<20} - gap maximo: {df_diagnostico_macro.loc[col, 'gap_max']} dias")

print(f"    - Gaps grandes (>{MAX_NULOS_PEQUENO}): {len(cols_macro_gap_grande)} variables")
if cols_macro_gap_grande:
    for col in cols_macro_gap_grande:
        print(f"      {col:<20} - gap maximo: {df_diagnostico_macro.loc[col, 'gap_max']} dias")

# 
# 7.6.3 RELLENAR GAPS PEQUENOS CON FFILL/BFILL
# 
print(f"\n  [7.6.3] Rellenando gaps pequenos (<={MAX_NULOS_PEQUENO}) con ffill/bfill...")
for col in cols_macro_gap_pequeno:
    dataset_long[col] = dataset_long[col].ffill().bfill()
print(f"    [OK] {len(cols_macro_gap_pequeno)} variables procesadas")

# 
# 7.6.4 COMPARATIVA DE TECNICAS PARA GAPS GRANDES (>2 NULOS)
# 
if len(cols_macro_gap_grande) > 0:
    print(f"\n  [7.6.4] COMPARATIVA DE TECNICAS (gaps > {MAX_NULOS_PEQUENO} nulos)...")
    print(f"    Evaluando 2 tecnicas: Spline Cubica, KNN Imputation")
    print(f"    Variables afectadas: {len(cols_macro_gap_grande)}")
    
    # Crear 2 copias para cada tecnica
    dataset_macro_spline = dataset_long.copy()
    dataset_macro_knn = dataset_long.copy()
    
    # ===== TECNICA 1: SPLINE CUBICA =====
    print(f"\n    [TECNICA 1] Spline Cubica...")
    for col in cols_macro_gap_grande:
        datos = dataset_macro_spline[col].values
        idx_vals = np.arange(len(datos))
        
        mask_nulos = np.isnan(datos)
        if mask_nulos.sum() > 0:
            idx_validos = idx_vals[~mask_nulos]
            datos_validos = datos[~mask_nulos]
            
            if len(idx_validos) > 2:
                try:
                    cs = CubicSpline(idx_validos, datos_validos, bc_type="natural")
                    datos_interp = cs(idx_vals)
                    dataset_macro_spline[col] = datos_interp
                except:
                    dataset_macro_spline[col] = dataset_macro_spline[col].ffill().bfill()
    
    # Aplicar ffill final para residuales
    for col in cols_macro_gap_grande:
        dataset_macro_spline[col] = dataset_macro_spline[col].ffill().bfill()
    
    # ===== TECNICA 2: KNN IMPUTATION =====
    print(f"    [TECNICA 2] KNN Imputation (k=5)...")
    try:
        knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
        dataset_macro_knn[cols_macro_gap_grande] = knn_imputer.fit_transform(
            dataset_macro_knn[cols_macro_gap_grande]
        )
    except Exception as e:
        print(f"      [ERROR] KNN fallo: {e}. Usando Spline como fallback.")
        dataset_macro_knn = dataset_macro_spline.copy()
    
    # 
    # 7.6.5 COMPARATIVA ESTADISTICA
    # 
    print(f"\n  [7.6.5] Comparativa estadistica de las 2 tecnicas...")
    
    col_ejemplo = cols_macro_gap_grande[0]
    
    print(f"\n    Variable de ejemplo: {col_ejemplo}")
    print(f"    {'Estadistica':<20} {'Original':>12} {'Spline':>12} {'KNN':>12}")
    print(f"    " + "-"*50)
    
    metricas_macro = {
        'Media': lambda x: np.nanmean(x),
        'Std': lambda x: np.nanstd(x),
    }
    
    stats_comparativa = {}
    for nombre, func in metricas_macro.items():
        orig = func(dataset_long[col_ejemplo].values)
        spl = func(dataset_macro_spline[col_ejemplo].values)
        knn_val = func(dataset_macro_knn[col_ejemplo].values)
        
        print(f"    {nombre:<20} {orig:>12.6f} {spl:>12.6f} {knn_val:>12.6f}")
        stats_comparativa[nombre] = {'Original': orig, 'Spline': spl, 'KNN': knn_val}
    
    # 
    # 7.6.6 SELECCION DE TECNICA
    # 
    print(f"\n  [7.6.6] Seleccion de tecnica optima...")
    
    if stats_comparativa and 'Std' in stats_comparativa:
        # Favorecer tecnica que preserva mejor la media y desviacion estandar
        score_spline = abs(stats_comparativa['Media']['Spline'] - stats_comparativa['Media']['Original']) + \
                       abs(stats_comparativa['Std']['Spline'] - stats_comparativa['Std']['Original'])
        score_knn = abs(stats_comparativa['Media']['KNN'] - stats_comparativa['Media']['Original']) + \
                    abs(stats_comparativa['Std']['KNN'] - stats_comparativa['Std']['Original'])
        
        scores_macro = {'Spline': score_spline, 'KNN': score_knn}
        tecnica_macro_elegida = min(scores_macro, key=scores_macro.get)
        
        print(f"    Scores de desviacion (menor = mejor):")
        for tec, score in sorted(scores_macro.items(), key=lambda x: x[1]):
            print(f"      {tec:<20} : {score:.6f}")
        
        print(f"\n    [ELEGIDA] {tecnica_macro_elegida}")
        print(f"    Justificacion: Preserva mejor las estadisticas de las variables macro")
    else:
        tecnica_macro_elegida = 'Spline'
        print(f"    [ELEGIDA] {tecnica_macro_elegida} (por defecto)")
    
    # Asignar dataset segun tecnica elegida
    if tecnica_macro_elegida == 'Spline':
        dataset_long = dataset_macro_spline.copy()
    else:
        dataset_long = dataset_macro_knn.copy()
else:
    print(f"\n  [7.6.4] No hay gaps > {MAX_NULOS_PEQUENO}. Saltando comparativa.")

# 
# 7.6.7 VERIFICACION FINAL DE MACRO
# 
print(f"\n  [7.6.7] Verificacion final de nulos en macro:")
nulos_macro_finales = dataset_long[cols_macro_exist].isna().sum()
nulos_macro_problemas = nulos_macro_finales[nulos_macro_finales > 0]

if len(nulos_macro_problemas) == 0:
    print(f"    [OK] Todas las variables macro sin nulos")
else:
    print(f"    [WARN] Variables con nulos residuales:")
    print(nulos_macro_problemas.to_string())

print(f"\n  Resumen tratamiento macro:")
print(f"    Gaps pequenos rellenados: {len(cols_macro_gap_pequeno)}")
print(f"    Gaps grandes comparados : {len(cols_macro_gap_grande)}")
print(f"    Nulos finales           : {nulos_macro_finales.sum()}")
print(f"    Variables macro         : {len(cols_macro_exist)}")
print(f"    [OK] Ninguna variable macro eliminada - todas preservadas")

ruta_excel = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '2.2_dataset_long_sin_nulos.xlsx'
)
try:
    dataset_long.to_excel(ruta_excel, index=False)
    print(f"\n  Dataset guardado: {ruta_excel}")
except PermissionError:
    print(f"\n  [ADVERTENCIA] No se pudo guardar Excel (archivo bloqueado). Continuando...")
except Exception as e:
    print(f"\n  [ADVERTENCIA] Error al guardar Excel: {e}. Continuando...")

# =============================================================================
# 7B. VISUALIZACION: SERIES TEMPORALES -- GAPS RELLENADOS
# =============================================================================
print("\n" + "="*70)
print("VISUALIZACION 7B: SERIES TEMPORALES DE GAPS RELLENADOS")
print("="*70)

tickers_con_gaps = tickers_gap_pequeno + tickers_gap_grande
n_graficos_gaps  = min(len(tickers_con_gaps), 4)

if n_graficos_gaps > 0:
    fig_g, axes_g = plt.subplots(n_graficos_gaps, 1,
                                  figsize=(16, 4 * n_graficos_gaps))
    if n_graficos_gaps == 1:
        axes_g = [axes_g]
    fig_g.suptitle('Series Temporales: Gaps Rellenados -- Adj Close',
                   fontsize=13, fontweight='bold')

    def _serie_ticker(df, t):
        return (
            df[df['Ticker'] == t]
            .assign(DateDT=lambda x: pd.to_datetime(x['Date']))
            .set_index('DateDT')['Adj Close']
            .sort_index()
        )

    for idx_g, ticker_g in enumerate(tickers_con_gaps[:n_graficos_gaps]):
        ax_g = axes_g[idx_g]

        s_ant = _serie_ticker(df_snap_antes_limpieza,   ticker_g)
        s_des = _serie_ticker(dataset_long,             ticker_g)
        mask_nan = s_ant.isna()

        if mask_nan.any():
            primer = mask_nan[mask_nan].index.min()
            win_ini = primer - pd.Timedelta(days=40)
            win_fin = primer + pd.Timedelta(days=60)
            s_ant    = s_ant[win_ini:win_fin]
            s_des    = s_des[win_ini:win_fin]
            mask_nan = mask_nan[win_ini:win_fin]

        ax_g.plot(s_des.index, s_des.values,
                  color='steelblue', linewidth=1.5,
                  label='Despues (rellenado)', zorder=3)
        ax_g.scatter(s_ant.dropna().index, s_ant.dropna().values,
                     color='black', s=12, zorder=5,
                     label='Dato original', alpha=0.8)

        fechas_gap = mask_nan[mask_nan].index
        if len(fechas_gap):
            vals_gap = s_des.reindex(fechas_gap)
            ax_g.scatter(vals_gap.index, vals_gap.values,
                         color='red', s=55, marker='x', zorder=6,
                         linewidths=2,
                         label=f'Imputado ({len(fechas_gap)} dias)')

        tipo_gap = ('gap grande (Spline/KNN)'
                    if ticker_g in tickers_gap_grande
                    else 'gap pequeno (ffill)')
        gap_max_g = diagnostico_df.loc[ticker_g, 'gap_max']
        ax_g.set_title(
            f'{ticker_g}  --  {tipo_gap}  (gap max. {gap_max_g} dias)',
            fontsize=10)
        ax_g.set_ylabel('Adj Close (EUR)')
        ax_g.legend(fontsize=8)
        ax_g.grid(alpha=0.3)

    plt.tight_layout()
    ruta_7b = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '7b_series_gaps_rellenados.png')
    plt.savefig(ruta_7b, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Guardado: {ruta_7b}")
else:
    print("  [INFO] No se detectaron gaps -- serie temporal omitida")

# =============================================================================
# 8. LIMPIEZA DE DATOS. DUPLICADOS
# =============================================================================
#
# Â¿QUE HACEMOS?
# Comprobar que cada observacion (fecha, ticker) aparece exactamente UNA vez.
# Un duplicado significa que la misma empresa cotiza dos veces el mismo dia 
# en nuestro dataset, violando la logica de la series de precios diarios.
#
# Â¿POR QUE?
# Los duplicados pueden ocurrir por:
#   - Errores humanos en data pipelines anteriores
#   - Bugs en descarga de yfinance (cache corrupto)
#   - Procesamiento manual que repite datos
# El MLPRegressor asume que cada fila es una observacion UNICA en el espacio temporal.
# Duplicados violarian esta suposicion e introduciriansesgo.
#
# IMPLICACIONES:
#   - Si no hay duplicados: procedemos sin cambios
#   - Si los hay: eliminamos, conservando first (mas conservador,
#     menos corrupcion probable en datos mas antiguos)
#
# CONCLUSION:
#   Verificacion rapida pero critica. Si hay duplicados, los eliminamos.
#   Si no los hay (lo probable), documento que PK es unica global.

print("\n" + "="*70)
print("SECCIN 8: LIMPIEZA DE DATOS - DUPLICADOS")
print("="*70)

# 
# 8.1 DETECCIN
# 
print("\n  [8.1] Deteccin de duplicados...")

duplicados_pk           = dataset_long.duplicated(subset=['PK'], keep=False)
duplicados_fecha_ticker = dataset_long.duplicated(subset=['Date', 'Ticker'], keep=False)

print(f"    Duplicados por PK            : {duplicados_pk.sum()}")
print(f"    Duplicados por Date + Ticker : {duplicados_fecha_ticker.sum()}")

if duplicados_pk.sum() > 0:
    print("\n    - Filas duplicadas encontradas:")
    print(
        dataset_long[duplicados_pk][['PK', 'Date', 'Ticker', 'Adj Close']]
        .sort_values(['Ticker', 'Date'])
        .to_string(index=False)
    )
else:
    print("    [OK] Sin duplicados - cada PK es nica")

# 
# 8.2 ELIMINACIN (si existen)
# 
if duplicados_pk.sum() > 0:
    shape_antes   = dataset_long.shape
    dataset_long  = dataset_long.drop_duplicates(subset=['PK'], keep='first').reset_index(drop=True)
    shape_despues = dataset_long.shape
    print(f"\n  [8.2] Duplicados eliminados:")
    print(f"    Shape antes   : {shape_antes}")
    print(f"    Shape despus : {shape_despues}")
    print(f"    Filas eliminadas: {shape_antes[0] - shape_despues[0]}")
else:
    print("\n  [8.2] No se requiere eliminacin de duplicados")

# 
# 8.3 VERIFICACIN FINAL
# 
assert dataset_long.duplicated(subset=['PK']).sum() == 0, \
    "- ERROR: siguen existiendo duplicados en PK tras la limpieza"
print(f"\n  [8.3] [OK] PK nica garantizada - {len(dataset_long):,} observaciones")
print(f"    Shape final: {dataset_long.shape}")

print(f"\n  CONCLUSION SECCION 8:")
print(f"  [OK] Integridad referencial garantizada (PK unica)")
print(f"  [OK] Dataset listo para transformaciones de Seccion 9")
# 8B. VISUALIZACION: HEATMAP DE NULOS ANTES/DESPUES DE LA LIMPIEZA
# =============================================================================
print("\n" + "="*70)
print("VISUALIZACION 8B: HEATMAP DE NULOS (antes vs despues)")
print("="*70)

# Snapshot post-limpieza (nulos + duplicados ya tratados)
df_snap_despues_limpieza = dataset_long.copy()

def _nulos_mensual(df, tickers, col='Adj Close'):
    """% de nulos por (Ticker, Mes) -> matriz (Mes x Ticker)."""
    df2 = df[df['Ticker'].isin(tickers)].copy()
    df2['YearMonth'] = pd.to_datetime(df2['Date']).dt.to_period('M')
    return (
        df2.groupby(['YearMonth', 'Ticker'])[col]
        .apply(lambda x: x.isna().mean() * 100)
        .unstack('Ticker')
        .fillna(0)
        .reindex(columns=sorted(tickers))
    )

tickers_hm  = sorted(tickers_candidatos)
mat_antes   = _nulos_mensual(df_snap_antes_limpieza,   tickers_hm)
mat_despues = _nulos_mensual(df_snap_despues_limpieza, tickers_hm)

periodos    = mat_antes.index.union(mat_despues.index)
mat_antes   = mat_antes.reindex(periodos).fillna(0)
mat_despues = mat_despues.reindex(periodos).fillna(0)

def _dibujar_heatmap_nulos(mat, titulo, tickers_lista, periodos_lista):
    fig_hm_single, ax_hm_single = plt.subplots(figsize=(13, 7))
    n   = len(periodos_lista)
    stp = max(1, n // 12)
    xt  = list(range(0, n, stp))
    im = ax_hm_single.imshow(mat.T.values, aspect='auto', cmap='Reds',
                              vmin=0, vmax=100, interpolation='nearest')
    ax_hm_single.set_title(titulo, fontsize=11, fontweight='bold')
    ax_hm_single.set_xlabel('Periodo (Mes-Anio)', fontsize=9)
    ax_hm_single.set_xticks(xt)
    ax_hm_single.set_xticklabels([str(periodos_lista[j]) for j in xt],
                                   rotation=45, ha='right', fontsize=7)
    ax_hm_single.set_yticks(range(len(tickers_lista)))
    ax_hm_single.set_yticklabels(tickers_lista, fontsize=7)
    plt.colorbar(im, ax=ax_hm_single, label='% nulos', fraction=0.025, pad=0.04)
    return fig_hm_single

# Heatmap ANTES de la limpieza
fig_8b_a = _dibujar_heatmap_nulos(
    mat_antes,
    'Heatmap de Valores Nulos (Adj Close) ANTES de la Limpieza\n(% nulos por ticker y mes)',
    tickers_hm, periodos
)
plt.tight_layout()
ruta_8b_a = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '8b_heatmap_nulos_antes.png')
plt.savefig(ruta_8b_a, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: {ruta_8b_a}")

# Heatmap DESPUES de la limpieza
fig_8b_d = _dibujar_heatmap_nulos(
    mat_despues,
    'Heatmap de Valores Nulos (Adj Close) DESPUES de la Limpieza\n(% nulos por ticker y mes — verificacion post-imputacion)',
    tickers_hm, periodos
)
plt.tight_layout()
ruta_8b_d = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '8b_heatmap_nulos_despues.png')
plt.savefig(ruta_8b_d, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: {ruta_8b_d}")

# =============================================================================
# 8C. VISUALIZACION: HEATMAP DE NULOS EN VARIABLES MACRO (ANTES/DESPUES)
# =============================================================================
print("\n" + "="*70)
print("VISUALIZACION 8C: HEATMAP DE NULOS - VARIABLES MACRO (antes vs despues)")
print("="*70)

# Columnas macro que existian antes de la limpieza
cols_macro_todas = ['EURUSD_Close', 'STOXX50_Close', 'ORO_Close',
                    'VIX_Close', 'TNX_Close', 'DAX40_Close']
cols_macro_antes = [c for c in cols_macro_todas if c in df_snap_antes_limpieza.columns]
cols_macro_despues = [c for c in cols_macro_todas if c in df_snap_despues_limpieza.columns]

if cols_macro_antes:
    def _nulos_mensual_macro(df, cols):
        """% de nulos por (Mes, Variable macro) -> matriz (Mes x Variable)."""
        df2 = df.drop_duplicates(subset=['Date']).copy()
        df2['YearMonth'] = pd.to_datetime(df2['Date']).dt.to_period('M')
        resultado = pd.DataFrame()
        for col in cols:
            if col in df2.columns:
                serie = df2.groupby('YearMonth')[col].apply(
                    lambda x: x.isna().mean() * 100
                )
                resultado[col] = serie
        return resultado.fillna(0)

    mat_macro_antes = _nulos_mensual_macro(df_snap_antes_limpieza, cols_macro_antes)
    mat_macro_despues = _nulos_mensual_macro(df_snap_despues_limpieza, cols_macro_despues)

    # Alinear periodos
    periodos_macro = mat_macro_antes.index.union(mat_macro_despues.index)
    mat_macro_antes = mat_macro_antes.reindex(periodos_macro).fillna(0)
    mat_macro_despues = mat_macro_despues.reindex(periodos_macro).fillna(0)

    # Variables eliminadas por filtro de calidad (si las hay)
    cols_eliminadas_macro = set(cols_macro_antes) - set(cols_macro_despues)

    def _dibujar_heatmap_macro(mat, titulo, cols_plot, periodos_lista):
        cols_en_mat = [c for c in cols_plot if c in mat.columns]
        datos = mat[cols_en_mat].T.values if cols_en_mat else np.zeros((1, len(periodos_lista)))
        labels_y = [c.replace('_Close', '') for c in cols_en_mat] if cols_en_mat else ['N/A']
        fig_hm_m, ax_hm_m = plt.subplots(figsize=(14, 5))
        im = ax_hm_m.imshow(datos, aspect='auto', cmap='Reds',
                             vmin=0, vmax=100, interpolation='nearest')
        ax_hm_m.set_title(titulo, fontsize=11, fontweight='bold')
        ax_hm_m.set_xlabel('Periodo (Mes-Anio)', fontsize=9)
        n = len(periodos_lista)
        stp = max(1, n // 12)
        xt = list(range(0, n, stp))
        ax_hm_m.set_xticks(xt)
        ax_hm_m.set_xticklabels([str(periodos_lista[j]) for j in xt],
                                  rotation=45, ha='right', fontsize=7)
        ax_hm_m.set_yticks(range(len(labels_y)))
        ax_hm_m.set_yticklabels(labels_y, fontsize=9)
        plt.colorbar(im, ax=ax_hm_m, label='% nulos', fraction=0.025, pad=0.04)
        return fig_hm_m

    # Heatmap macro ANTES de la limpieza
    fig_8ca = _dibujar_heatmap_macro(
        mat_macro_antes,
        'Heatmap de Valores Nulos — Variables Macro ANTES de la Limpieza\n(% nulos por variable y mes)',
        cols_macro_antes, periodos_macro
    )
    plt.tight_layout()
    ruta_8ca = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '8c_heatmap_nulos_macro_antes.png')
    plt.savefig(ruta_8ca, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Guardado: {ruta_8ca}")

    # Heatmap macro DESPUES de la limpieza
    _nota_elim = (f' (Eliminadas: {", ".join(cols_eliminadas_macro)})'
                  if cols_eliminadas_macro else '')
    fig_8cd = _dibujar_heatmap_macro(
        mat_macro_despues,
        f'Heatmap de Valores Nulos — Variables Macro DESPUES de la Limpieza{_nota_elim}\n(% nulos por variable y mes)',
        cols_macro_despues, periodos_macro
    )
    plt.tight_layout()
    ruta_8cd = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '8c_heatmap_nulos_macro_despues.png')
    plt.savefig(ruta_8cd, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Guardado: {ruta_8cd}")
else:
    print(f"  [SKIP] No hay variables macro para visualizar")


# =============================================================================
# 9. TRANSFORMACIN. COLUMNAS ADICIONALES
# =============================================================================
# Aadimos rendimientos logartmicos y desviaciones para 3 horizontes temporales:
#
#   HORIZONTES TEMPORALES:
#     - Diario (1 da)
#     - Mensual (~22 das hbiles)
#     - Trimestral (~66 das hbiles)
#
#   PARA CADA VARIABLE/ACTIVO:
#     - Retorno logartmico: ln(Precio_hoy / Precio_ayer)
#     - Desviacion tipica movil: std de retornos en ventana temporal
#     - Retorno acumulado: suma de retornos diarios en ventana
#
#   VARIABLES A TRANSFORMAR:
#     1. Empresas seleccionadas (por Ticker) - de Adj Close
#     2. Variables macro (EURUSD, STOXX50, ORO) - de _Close
#     3. Benchmark (DAX40) - de _Close
#
# Por qu aqu y no antes?
#   Los datos an no estaban limpios (nulos, duplicados). 
#   Ahora en Seccin 9 tenemos el dataset final y limpio tras Secciones 6-8.

print("\n" + "="*70)
print("SECCIN 9: TRANSFORMACIN - COLUMNAS ADICIONALES")
print("="*70)

dataset_long = dataset_long.sort_values(['Ticker', 'Date']).reset_index(drop=True)

# Configuracin de horizontes temporales
horizonte_diario = 1        # 1 da
horizonte_mensual = 22      # ~1 mes de trading
horizonte_trimestral = 66   # ~3 meses de trading

# 
#  9.1 RENDIMIENTOS Y DESVIACIONES PARA EMPRESAS (por Ticker, de Adj Close)
# 
print(f"\n  [9.1] Calculando rendimientos y desviaciones para empresas...")

# Retorno logartmico diario
dataset_long['Log_Return_D'] = dataset_long.groupby('Ticker')['Adj Close'].transform(
    lambda x: np.log(x / x.shift(1))
)

# Retorno acumulado mensual (suma de retornos diarios ltimos 22 das)
dataset_long['Log_Return_M'] = dataset_long.groupby('Ticker')['Log_Return_D'].transform(
    lambda x: x.rolling(window=horizonte_mensual).sum()
)

# Retorno acumulado trimestral (suma de retornos diarios ltimos 66 das)
dataset_long['Log_Return_Q'] = dataset_long.groupby('Ticker')['Log_Return_D'].transform(
    lambda x: x.rolling(window=horizonte_trimestral).sum()
)

# Desviacion tipica diaria (volatilidad ultimos 5 dias = 1 semana bursatil)
dataset_long['Std_D'] = dataset_long.groupby('Ticker')['Log_Return_D'].transform(
    lambda x: x.rolling(window=5).std()
)

# Desviacion tipica mensual (volatilidad ltimos 22 das)
dataset_long['Std_M'] = dataset_long.groupby('Ticker')['Log_Return_D'].transform(
    lambda x: x.rolling(window=horizonte_mensual).std()
)

# Desviacion tipica trimestral (volatilidad ltimos 66 das)
dataset_long['Std_Q'] = dataset_long.groupby('Ticker')['Log_Return_D'].transform(
    lambda x: x.rolling(window=horizonte_trimestral).std()
)

# MODIFICACIÓN NUEVA: Volume_Ratio_22 se precalcula aquí para EDA y visualizaciones.
# Las features de volumen para el modelo supervisado se construyen en
# DAX_Analisis_del_Dato.py (construir_features()) con proteccion contra div/0.
if 'Volume' in dataset_long.columns:
    dataset_long['Volume_Ratio_22'] = dataset_long.groupby('Ticker')['Volume'].transform(
        lambda x: x / (x.rolling(window=horizonte_mensual).mean() + 1e-8)
    )
    print(f"    [OK] Empresas: Log_Return_D/M/Q, Std_D/M/Q, Volume_Ratio_22")
else:
    print(f"    [OK] Empresas: Log_Return_D/M/Q, Std_D/M/Q (Volume no disponible)")

# 
#  9.2 RENDIMIENTOS Y DESVIACIONES PARA VARIABLES MACRO (EURUSD, STOXX50, ORO)
# 
print(f"\n  [9.2] Calculando rendimientos y desviaciones para variables macro...")

# MODIFICACIÓN NUEVA: Añadir VIX y TNX a macro_vars para que se generen
# las columnas VIX_Log_Return_D, TNX_Log_Return_D, etc. en el CSV final.
# Sin esto, DAX_Analisis_del_Dato.py no encontraría esas columnas.
macro_vars = ['EURUSD', 'STOXX50', 'ORO', 'VIX', 'TNX', 'DAX40']
for macro_var in macro_vars:
    col_close = f'{macro_var}_Close'
    
    # Verificar que la columna existe
    if col_close not in dataset_long.columns:
        print(f"    [ADVERTENCIA] {col_close} no encontrada, saltando...")
        continue
    
    # Retorno logartmico diario (CORREGIDO: groupby para no cruzar entre tickers)
    col_ret_d = f'{macro_var}_Log_Return_D'
    dataset_long[col_ret_d] = dataset_long.groupby('Ticker')[col_close].transform(
        lambda x: np.log(x / x.shift(1))
    )
    
    # Retorno acumulado mensual
    col_ret_m = f'{macro_var}_Log_Return_M'
    dataset_long[col_ret_m] = dataset_long.groupby('Ticker')[col_ret_d].transform(
        lambda x: x.rolling(window=horizonte_mensual).sum()
    )
    
    # Retorno acumulado trimestral
    col_ret_q = f'{macro_var}_Log_Return_Q'
    dataset_long[col_ret_q] = dataset_long.groupby('Ticker')[col_ret_d].transform(
        lambda x: x.rolling(window=horizonte_trimestral).sum()
    )
    
    # Desviacion tipica diaria (ventana 5: semanal, ya que rolling window=1 es indefinido)
    col_std_d = f'{macro_var}_Std_D'
    dataset_long[col_std_d] = dataset_long.groupby('Ticker')[col_ret_d].transform(
        lambda x: x.rolling(window=5).std()
    )
    
    # Desviacion tipica mensual
    col_std_m = f'{macro_var}_Std_M'
    dataset_long[col_std_m] = dataset_long.groupby('Ticker')[col_ret_d].transform(
        lambda x: x.rolling(window=horizonte_mensual).std()
    )
    
    # Desviacion tipica trimestral
    col_std_q = f'{macro_var}_Std_Q'
    dataset_long[col_std_q] = dataset_long.groupby('Ticker')[col_ret_d].transform(
        lambda x: x.rolling(window=horizonte_trimestral).std()
    )
    
    print(f"    [OK] {macro_var}: Log_Return_D/M/Q, Std_D/M/Q")

# Verificacin
print(f"\n  [9.3] Verificacin de columnas creadas:")
print(f"    Ticker en columnas: {'Ticker' in dataset_long.columns}")
print(f"    Total columnas: {len(dataset_long.columns)}")

# Columnas nuevas para el resumen
cols_nuevas_empresas = ['Log_Return_D', 'Log_Return_M', 'Log_Return_Q', 
                        'Std_D', 'Std_M', 'Std_Q']
cols_nuevas_macro = []
for macro_var in macro_vars:
    if f'{macro_var}_Close' in dataset_long.columns:
        cols_nuevas_macro.extend([
            f'{macro_var}_Log_Return_D', f'{macro_var}_Log_Return_M', f'{macro_var}_Log_Return_Q',
            f'{macro_var}_Std_D', f'{macro_var}_Std_M', f'{macro_var}_Std_Q'
        ])

print(f"\n  Nulos por columna nueva (muestra):")
for col in cols_nuevas_empresas[:3]:
    nulos = dataset_long[col].isna().sum()
    pct = nulos / len(dataset_long) * 100
    print(f"    {col:<20} : {nulos:>6} nulos ({pct:>5.2f}%)")

print(f"\n  Ejemplo para {tickers_candidatos[0]} (primeras 5 filas con datos):")
ejemplo = (
    dataset_long[dataset_long['Ticker'] == tickers_candidatos[0]]
    [['Date', 'Adj Close', 'Log_Return_D', 'Log_Return_M', 'Std_M']]
    .dropna()
    .head(5)
)
print(ejemplo.to_string(index=False))

print(f"\n  Shape dataset tras columnas adicionales: {dataset_long.shape}")

ruta_excel = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '3_dataset_long_con_columnas_estadisticas.xlsx'
)
try:
    dataset_long.to_excel(ruta_excel, index=False)
    print(f"\n  Dataset guardado: {ruta_excel}")
except PermissionError:
    print(f"\n  [ADVERTENCIA] No se pudo guardar Excel (archivo bloqueado). Continuando...")
except Exception as e:
    print(f"\n  [ADVERTENCIA] Error al guardar Excel: {e}. Continuando...")

print(f"\n  Seccin 9 completada: Transformacin + columnas para 3 horizontes temporales")


# GRAFICO 9 -- Histograma de retornos diarios con curva normal superpuesta
print(f"\n  [9.3] Visualizacion de distribucion de retornos...")
fig_hist, axes_hist = plt.subplots(2, 3, figsize=(16, 10))
fig_hist.suptitle('Distribucion de Retornos Diarios (Log_Return_D) - Muestra de 6 empresas',
                  fontsize=12, fontweight='bold')
tickers_sample = sorted(tickers_candidatos)[:6]
for idx, ticker in enumerate(tickers_sample):
    ax = axes_hist[idx // 3, idx % 3]
    rets = dataset_long[dataset_long['Ticker'] == ticker]['Log_Return_D'].dropna()
    
    # Histograma
    ax.hist(rets, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black')
    
    # Curva normal superpuesta
    mu, sigma = rets.mean(), rets.std()
    x = np.linspace(rets.min(), rets.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal(Î¼,Ïƒ)')
    
    # Estadisticas
    kurt = stats.kurtosis(rets)
    skew = stats.skew(rets)
    ax.set_title(f'{ticker}\nKurtosis={kurt:.2f}, Skew={skew:.2f}',
                fontsize=9, fontweight='bold')
    ax.set_xlabel('Log_Return_D', fontsize=8)
    ax.set_ylabel('Densidad', fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

plt.tight_layout()
ruta_hist = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         '9_histograma_retornos.png')
plt.savefig(ruta_hist, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Grafica guardada: {ruta_hist}")

# =============================================================================
# 10. SELECCIN DE 10-15 ACTIVOS PARA EL MODELO
# =============================================================================
# Por qu seleccionamos aqu y no antes?
#   Ya tenemos los retornos calculados (Log_Return) y los datos limpios.
#   El scoring estadstico y el clustering se hacen sobre retornos limpios
#   -> tiene sentido seleccionar ahora que los datos estn en su estado definitivo.
#
# Criterios (en orden):
#   1. Scoring estadstico : kurtosis + ADF + volatilidad + cobertura
#   2. Clustering jerárquico : diversificacin cuantitativa por comportamiento
#   3. Ajuste sectorial : diversificacin econmica de la economa alemana

print("\n" + "="*70)
print("SECCIN 10: SELECCIN DE 10-15 ACTIVOS PARA EL MODELO")
print("="*70)

#  10.1 Pivot a formato ancho para operaciones matriciales 
df_precios = (
    dataset_long[dataset_long['Ticker'].isin(tickers_candidatos)]
    .pivot(index='Date', columns='Ticker', values='Adj Close')
)

cobertura = df_precios.notna().sum() / len(df_precios)

print(f"\n  Precios en formato ancho: {df_precios.shape}")

#  10.2 Retornos logartmicos temporales para scoring y clustering 
# TEMPORALES - no se anaden al dataset. Se usan solo para scoring/clustering.
# Usamos los precios (no Log_Return de dataset_long) para tener la serie
# completa sin NaNs iniciales que truncaran el anlisis.

retornos_temp = np.log(
    df_precios[tickers_candidatos] / df_precios[tickers_candidatos].shift(1)
).dropna()

print(f"  Retornos temporales: {retornos_temp.shape}")

#  10.3 Winsorizacin temporal para scoring y clustering 
# Se aplica ANTES del scoring para que kurtosis y correlaciones reflejen
# el comportamiento habitual, no eventos extremos puntuales.
# TEMPORAL - no modifica dataset_long.

def winsorizacion_temp(df, lower=0.01, upper=0.99):
    df_w = df.copy()
    for col in df.columns:
        p_lo = df[col].quantile(lower)
        p_hi = df[col].quantile(upper)
        df_w[col] = df[col].clip(lower=p_lo, upper=p_hi)
    return df_w

retornos_limpios = winsorizacion_temp(retornos_temp)

#  10.4 Scoring estadstico 
# QUE: Calcular score compuesto que combine varios indices estadsticos
# POR QUE: Seleccionar activos que balanceen 2 criterios crticos:
#   a) Estabilidad de colas (bajo kurtosis)
#   b) Riesgo moderado (volatilidad no extrema)
# IMPLICA: Ponderacion 0.50 (kurtosis) + 0.50 (volatilidad)
# JUSTIFICACION DE PESOS:
#   - 0.50 kurtosis (MAXIMA PRIORIDAD): Evitar fat tails extremas que invaldan
#     distribuciones normales asumidas por el MLPRegressor. Kurtosis elevado = outliers
#     severos = predicciones perturbadas.
#   - 0.50 volatilidad (MAXIMA PRIORIDAD): Balance de riesgo. Queremos
#     activos "normales" sin volatilidad extrema (que seran idiosincrasia pura),
#     pero tampoco cero volatilidad (que seria irrelevante). Penalizamos desvio
#     del mediano (activos muy volatiles O muy estables = no informativos).
# CONCLUSION: Scoring favore estabilidad (kurtosis) + riesgo balanceado (volatilidad)

print(f"\n  [10.4] Scoring estadstico...")

scoring = pd.DataFrame(index=tickers_candidatos)
scoring['kurtosis']    = retornos_limpios.apply(stats.kurtosis)
scoring['adf_pval']    = retornos_limpios.apply(
    lambda x: adfuller(x.dropna(), autolag='AIC')[1]
)
scoring['volatilidad'] = retornos_limpios.std()
scoring['skewness']    = retornos_limpios.apply(stats.skew)

def norm_inv(s):
    """Normalizar INVERSO: valores mayores en s -> valores menores en resultado.
    Usado para kurtosis (queremos BAJO, no alto)."""
    rng = s.max() - s.min()
    return 1 - (s - s.min()) / rng if rng > 0 else pd.Series(1.0, index=s.index)

def norm(s):
    """Normalizar DIRECTO: valores mayores en s -> valores mayores en resultado."""
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else pd.Series(1.0, index=s.index)

# MODIFICACIÓN NUEVA: Añadir Sharpe historico al scoring (40/40/20).
# El scoring original (50/50) solo penalizaba kurtosis extrema + vol no equilibrada,
# pero no consideraba si el activo era RENTABLE. Un activo estable pero con
# retorno negativo no aporta valor a la cartera. Pesos: 40% kurtosis (estabilidad),
# 40% vol equilibrada (riesgo moderado), 20% Sharpe historico (rentabilidad).
scoring['sharpe_hist'] = (retornos_limpios.mean() / retornos_limpios.std()) * np.sqrt(252)
scoring['score'] = (
    0.40 * norm_inv(scoring['kurtosis']) + 
    0.40 * norm_inv((scoring['volatilidad'] - scoring['volatilidad'].median()).abs()) +
    0.20 * norm(scoring['sharpe_hist'])
)


print(f"\n  Scoring compuesto (40% kurtosis inv + 40% desvio vol + 20% Sharpe hist):")
print(f"  Interpretacion: score alto = activos ESTABLES + RIESGO BALANCEADO + RENTABLES")
cols_ver = ['kurtosis', 'skewness', 'volatilidad', 'adf_pval', 'sharpe_hist', 'score']
print(scoring[cols_ver].sort_values('score', ascending=False).round(4).to_string())

# GRAFICO 10.4 -- Heatmap de scoring (matriz de metricas)
print(f"\n  [10.4.2] Visualizacion de heatmap de scoring...")
scoring_viz = scoring[['kurtosis', 'volatilidad', 'score']].copy()
# Normalizar para visualizacion (0-1)
scoring_norm = (scoring_viz - scoring_viz.min()) / (scoring_viz.max() - scoring_viz.min())
# Invertir kurtosis (queremos valores bajos = buenos = colores claros)
scoring_norm['kurtosis'] = 1 - scoring_norm['kurtosis']

fig_heat, ax_heat = plt.subplots(figsize=(10, 12))
im = ax_heat.imshow(scoring_norm.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax_heat.set_xticks(range(len(scoring_norm.columns)))
ax_heat.set_yticks(range(len(scoring_norm.index)))
ax_heat.set_xticklabels(scoring_norm.columns, fontsize=9)
ax_heat.set_yticklabels(scoring_norm.index, fontsize=8)
ax_heat.set_title('Heatmap de Scoring Estadistico\n(verde=optimo, rojo=malo; kurtosis invertido)',
                 fontsize=11, fontweight='bold', pad=15)

# Marcar seleccionadas con borde (se define despues en linea ~1900)
# Por ahora solo mostramos la matriz

plt.colorbar(im, ax=ax_heat, label='Score normalizado (0-1)', fraction=0.046, pad=0.04)
plt.tight_layout()
ruta_heat = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         '10_4_heatmap_scoring.png')
plt.savefig(ruta_heat, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Heatmap de scoring guardado: {ruta_heat}")

#  10.5 Clustering jerarquico 
# QUE: Agrupar activos en N clusters (seleccionado por Gap Statistic)
#      y seleccionar 1 representante por cluster.
# POR QUE:
#   a) DIVERSIFICACION: 18 activos de 18 clusters independientes. Garantiza
#      estructuras de correlacion distintas (ej: SAP.DE no correlaciona igual
#      a BMW.DE, garantiza portafolio con riesgo idiosincratico no redundante,).
#      que es el tipo de riesgo individual por ejemplo dentro de un sector. 
#      ej) automovilistico con china como nuevo gran productor, no es un riesgo
#      del sector farmacÃ©utico
#   b) EVITAR MULTICOLINEALIDAD: Retornos del cluster SAP + SIE casi identicos
#      (Tech sector). Seleccionar solo 1 evita ruido y redundancia en features.
#   c) TAMANO MANEJABLE: 18 activos es equilibrio ML: suficientes para
#      diversificar, pocos para entrenamiento eficiente.
# IMPLICA: Usar Spearman + Ward con N_CLUSTERS seleccionado por Gap Statistic (Tibshirani, 2001)
# JUSTIFICACIONES TECNICAS:
#   1. SPEARMAN vs PEARSON:
#      - Pearson asume relaciones lineales. Retornos financieros = no lineales
#        (periodos calma vs crisis tienen estructuras correlacion distintas).
#      - Spearman usa ranks, invariante a transformaciones monotonas,
#        captura dependencias curvilineales (volatility clustering).
#        Por decirlo de alguna forma introduce la normalidad de la que carecen
#        los datos. One Note ClusterizaciÃ³n
#      - Distancia: sqrt(2*(1-Ï_spearman)) = metrica valida que cumple
#        desigualdad triangular, apropiada para linkage.
#   2. WARD vs LINKAGE ALTERNATIVES:
#      - Complete linkage (maxima distancia): produce clusters muy dispersos.
#      - Single linkage (minima distancia): produce "cadenas" (bad for ML).
#      - Ward (minimiza varianza intra-cluster): clusters compactos, balanceados.
#      - Implicacion: Ward garantiza que miembros de cada cluster son SIMILARES
#        en comportamiento retorno, reduciendo multicolinealidad.
#   3. N_CLUSTERS (seleccionado por Gap Statistic, Tibshirani et al., 2001):
#      - Se evalua k=2..25 comparando inercia real vs inercia bajo distribucion
#        uniforme de referencia (B=100 bootstrap).
#      - Regla 1-SE: menor k cuyo Gap(k) >= Gap(k+1) - s_{k+1}
#        (modelo mas parsimonioso estadisticamente comparable al optimo).
#      - Restriccion: k>=10 (minimo para cartera diversificada).
# CONCLUSION: Spearman + Ward asegura clusters densos + representacionalmente
#   diversos, con seleccion posterior por score (KURTOSIS + COBERTURA).

print(f"\n  [10.5] Clustering jerarquico (Ward + Spearman correlation)...")

# Calcular matriz de correlacion y distancia (necesarias para el analisis)
corr_spearman  = retornos_limpios.corr(method='spearman')
distancia      = np.sqrt(2 * (1 - corr_spearman))
distancia_array = np.array(distancia.values, copy=True, dtype=np.float64)
np.fill_diagonal(distancia_array, 0)
linkage_matrix = linkage(squareform(distancia_array), method='ward')

# -------------------------------------------------------------------------
# 10.5.0 SELECCION OPTIMA DE N_CLUSTERS (Gap Statistic - Tibshirani et al., 2001)
# -------------------------------------------------------------------------
# El Gap Statistic compara la dispersion intra-cluster del clustering real
# contra la esperada bajo una distribucion de referencia uniforme (sin estructura).
#   Gap(k) = E*[log(W_k)] - log(W_k)
# donde W_k = inercia intra-cluster, E* = esperanza sobre B muestras bootstrap.
#
# Regla de seleccion (1-SE rule, Tibshirani 2001):
#   Elegir el menor k tal que Gap(k) >= Gap(k+1) - s_{k+1}
#   donde s_k = desviacion estandar de los log(W*_k) bootstrap * sqrt(1 + 1/B).
#   Esto selecciona el modelo mas parsimonioso que sea estadisticamente
#   comparable al optimo, evitando sobreajuste.
#
# Restriccion adicional: k >= 10 (minimo para cartera diversificada).
# Referencia: Tibshirani, Walther & Hastie (2001), JRSS-B, 63(2), 411-423.

K_MIN, K_MAX = 2, min(25, len(tickers_candidatos) - 1)
B_BOOTSTRAP  = 100  # Muestras de referencia (50-100 recomendado por Tibshirani)
K_MINIMO_CARTERA = 10

print(f"\n  [10.5.0] Gap Statistic: evaluando k={K_MIN}..{K_MAX} con B={B_BOOTSTRAP} muestras bootstrap...")

def calcular_inercia_ward(dist_matrix, labels):
    """Inercia intra-cluster: suma de distancias al cuadrado dentro de cada cluster."""
    W = 0.0
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) > 1:
            sub = dist_matrix[np.ix_(idx, idx)]
            W += (sub ** 2).sum() / (2 * len(idx))
    return W

# Inercia real para cada k
inercia_real = {}
for k in range(K_MIN, K_MAX + 1):
    labels = fcluster(linkage_matrix, t=k, criterion='maxclust')
    inercia_real[k] = calcular_inercia_ward(distancia_array, labels)

# Bootstrap: generar B muestras uniformes en el rango de los datos y calcular inercia
n_activos = distancia_array.shape[0]
rng = np.random.default_rng(seed=42)

log_W_star = {k: [] for k in range(K_MIN, K_MAX + 1)}

for b in range(B_BOOTSTRAP):
    # Generar datos uniformes en el espacio de distancias
    # Permutamos las filas/columnas de la matriz de distancia para romper estructura
    idx_perm = rng.permutation(n_activos)
    dist_perm = distancia_array[np.ix_(idx_perm, idx_perm)]
    # Anadir ruido uniforme para destruir la estructura de clusters
    ruido = rng.uniform(0, distancia_array.max() * 0.1, size=dist_perm.shape)
    ruido = (ruido + ruido.T) / 2  # Simetrizar
    np.fill_diagonal(ruido, 0)
    dist_ref = np.clip(dist_perm + ruido, 0, None)
    
    linkage_ref = linkage(squareform(dist_ref), method='ward')
    for k in range(K_MIN, K_MAX + 1):
        labels_ref = fcluster(linkage_ref, t=k, criterion='maxclust')
        W_ref = calcular_inercia_ward(dist_ref, labels_ref)
        log_W_star[k].append(np.log(W_ref + 1e-10))

# Calcular Gap, desviacion estandar y regla 1-SE
resultados_gap = []
print(f"\n  {'k':>4} {'log(W_k)':>10} {'E*[log(W*)]':>13} {'Gap(k)':>10} {'s_k':>10}")
print(f"  {'-'*50}")

for k in range(K_MIN, K_MAX + 1):
    log_Wk = np.log(inercia_real[k] + 1e-10)
    E_log_Wstar = np.mean(log_W_star[k])
    sd_log_Wstar = np.std(log_W_star[k], ddof=1)
    s_k = sd_log_Wstar * np.sqrt(1 + 1 / B_BOOTSTRAP)
    gap = E_log_Wstar - log_Wk
    
    resultados_gap.append({
        'k': k, 'log_Wk': log_Wk, 'E_log_Wstar': E_log_Wstar,
        'gap': gap, 's_k': s_k
    })
    print(f"  {k:>4} {log_Wk:>10.4f} {E_log_Wstar:>13.4f} {gap:>10.4f} {s_k:>10.4f}")

df_gap = pd.DataFrame(resultados_gap)

# Regla 1-SE de Tibshirani: menor k tal que Gap(k) >= Gap(k+1) - s_{k+1}
k_optimo_gap = None
for i in range(len(df_gap) - 1):
    k_actual = df_gap.iloc[i]['k']
    gap_actual = df_gap.iloc[i]['gap']
    gap_siguiente = df_gap.iloc[i + 1]['gap']
    s_siguiente = df_gap.iloc[i + 1]['s_k']
    
    if k_actual >= K_MINIMO_CARTERA and gap_actual >= gap_siguiente - s_siguiente:
        k_optimo_gap = int(k_actual)
        break

# Fallback: si la regla 1-SE no encuentra k >= K_MINIMO_CARTERA, usar maximo gap con k >= 10
if k_optimo_gap is None:
    df_gap_validos = df_gap[df_gap['k'] >= K_MINIMO_CARTERA]
    k_optimo_gap = int(df_gap_validos.loc[df_gap_validos['gap'].idxmax(), 'k'])

# Gap maximo global (sin restriccion) para referencia
k_gap_global = int(df_gap.loc[df_gap['gap'].idxmax(), 'k'])

N_CLUSTERS = k_optimo_gap

print(f"\n  Resultado Gap Statistic (Tibshirani et al., 2001):")
print(f"    - Gap maximo global: k={k_gap_global} (gap={df_gap.loc[df_gap['gap'].idxmax(), 'gap']:.4f})")
print(f"    - Regla 1-SE con k>={K_MINIMO_CARTERA}: k={k_optimo_gap}")
print(f"    -> N_CLUSTERS seleccionado: {N_CLUSTERS}")

# GRAFICO 10.5.0 -- Gap Statistic vs k
# Muestra cinco elementos:
#   (1) Curva azul con barras de error: Gap(k) +/- s_k para cada k evaluado.
#   (2) Linea verde discontinua: umbral 1-SE = Gap(k+1) - s_{k+1}. Cuando la
#       curva azul supera esta linea, ese k satisface la condicion de Tibshirani.
#   (3) Zona sombreada verde: region donde Gap(k) >= umbral (k candidatos validos
#       con k >= K_MINIMO_CARTERA).
#   (4) Diamantes verdes: todos los k candidatos validos (alternativas al elegido).
#   (5) Linea roja: k finalmente seleccionado (el menor candidato valido segun
#       la regla 1-SE con restriccion k >= K_MINIMO_CARTERA).
#   Opcionalmente, linea naranja si el k de gap maximo global difiere del elegido.
fig_k, ax_gap = plt.subplots(figsize=(11, 7))
fig_k.suptitle('Seleccion optima de N_CLUSTERS - Gap Statistic\n(Tibshirani, Walther & Hastie, 2001)',
               fontsize=12, fontweight='bold')

# Umbral 1-SE para k = K_MIN..K_MAX-1: Gap(k+1) - s_{k+1}
# Indica el minimo valor que debe tener Gap(k) para que k sea candidato valido.
ks_umbral  = df_gap['k'].values[:-1]                                 # k de K_MIN a K_MAX-1
umbral_1se = df_gap['gap'].values[1:] - df_gap['s_k'].values[1:]    # Gap(k+1) - s_{k+1}

# Mascara de k candidatos: cumplen regla 1-SE y restriccion de cartera minima
mascara_candidatos = (
    (df_gap['gap'].values[:-1] >= umbral_1se) &
    (ks_umbral >= K_MINIMO_CARTERA)
)
ks_candidatos = [int(x) for x in ks_umbral[mascara_candidatos]]     # convertir a int puro

# (1) Curva Gap(k) con barras de error +/- s_k
ax_gap.errorbar(df_gap['k'], df_gap['gap'], yerr=df_gap['s_k'],
                fmt='b-o', markersize=5, capsize=3, label='Gap(k) +/- s_k', zorder=3)
# (2) Linea de umbral 1-SE
ax_gap.plot(ks_umbral, umbral_1se, 'g--', linewidth=1.5, alpha=0.8,
            label='Umbral 1-SE: Gap(k+1) - s(k+1)')
# (3) Zona sombreada donde Gap(k) supera el umbral (candidatos validos)
ax_gap.fill_between(ks_umbral, umbral_1se, df_gap['gap'].values[:-1],
                    where=mascara_candidatos, color='green', alpha=0.12,
                    label='Zona 1-SE satisfecha')
# (4) Diamantes sobre cada k candidato valido
if len(ks_candidatos) > 0:
    gaps_candidatos = df_gap.set_index('k').loc[ks_candidatos, 'gap'].values
    ax_gap.scatter(ks_candidatos, gaps_candidatos, color='green', s=70,
                   zorder=5, marker='D',
                   label='k candidatos: ' + str(ks_candidatos))
# (5) Linea roja: k seleccionado (menor candidato valido)
ax_gap.axvline(N_CLUSTERS, color='red', linestyle='--', linewidth=2,
               label=f'k seleccionado = {N_CLUSTERS} (regla 1-SE, min k)')
# Linea naranja opcional: k con gap maximo global si difiere del seleccionado
if k_gap_global != N_CLUSTERS:
    ax_gap.axvline(k_gap_global, color='orange', linestyle=':',
                   label=f'k gap maximo global = {k_gap_global}')
ax_gap.axvline(K_MINIMO_CARTERA, color='gray', linestyle=':', alpha=0.5,
               label=f'k minimo cartera = {K_MINIMO_CARTERA}')
ax_gap.set_xlabel('Numero de clusters (k)', fontsize=10)
ax_gap.set_ylabel('Gap(k)', fontsize=10)
# Leyenda debajo del grafico en 3 columnas para no tapar la curva
ax_gap.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=True)
ax_gap.grid(alpha=0.3)

# rect=[0, 0.1, 1, 1] reserva el 10% inferior para la leyenda exterior
plt.tight_layout(rect=[0, 0.1, 1, 1])
ruta_k = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '10_5_0_gap_statistic.png')
plt.savefig(ruta_k, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Grafica guardada: {ruta_k}")

# TABLA 10.5.1 -- Matriz de Correlacion Spearman
print(f"\n  [10.5.1] Matriz de Correlacion de Spearman (retornos limpios):")
print(f"\n  {corr_spearman.round(3).to_string()}")

# TABLA 10.5.2 -- Matriz de Distancias Ward
print(f"\n  [10.5.2] Matriz de Distancias Ward (basada en Spearman):")
print(f"  Distancia = sqrt(2 * (1 - Rho_Spearman))")
print(f"\n  {distancia.round(3).to_string()}")

clusters_arr = fcluster(linkage_matrix, t=N_CLUSTERS, criterion='maxclust')
cluster_df   = pd.DataFrame(
    {'Ticker': tickers_candidatos, 'Cluster': clusters_arr}
).set_index('Ticker')

print(f"\n  -> {N_CLUSTERS} clusters encontrados usando Ward linkage sobre Spearman distances")

# Representante de cada cluster: empresa con mayor score compuesto
seleccionadas = []
for c in sorted(cluster_df['Cluster'].unique()):
    miembros = cluster_df[cluster_df['Cluster'] == c].index.tolist()
    elegida  = scoring.loc[miembros, 'score'].idxmax()
    seleccionadas.append(elegida)
    
    # Documentar la decision: muestra scores de todos los miembros
    score_elegida = scoring.loc[elegida, 'score']
    print(f"\n    Cluster {c:2d}: {miembros}")
    for miembro in sorted(miembros, key=lambda x: scoring.loc[x, 'score'], reverse=True):
        marca = "[OK] ELEGIDA" if miembro == elegida else "  DESCARTADA"
        sc = scoring.loc[miembro, 'score']
        kurt = scoring.loc[miembro, 'kurtosis']
        vol = scoring.loc[miembro, 'volatilidad']
        print(f"      {marca:13s}: {miembro:8s} | score={sc:.3f} | "
              f"kurt={kurt:6.2f} | vol={vol:.4f}")
    
    # Justificacion de la seleccion
    dart = scoring.loc[elegida, 'kurtosis']
    print(f"      -> JUSTIFICACION: {elegida} elegida por score maximo ({score_elegida:.3f})")
    print(f"         - Mejor control de outliers (kurtosis={dart:.2f})")
    print(f"         - Volatilidad equilibrada ({scoring.loc[elegida, 'volatilidad']:.4f})")
    if len(miembros) > 1:
        alternatives = [(m, scoring.loc[m, 'score']) for m in miembros if m != elegida]
        alternatives.sort(key=lambda x: x[1], reverse=True)
        print(f"         - Alternativas descartadas: {alternatives[0][0]} ({alternatives[0][1]:.3f}), "
              f"{alternatives[1][0] if len(alternatives) > 1 else 'N/A'} "
              f"({alternatives[1][1]:.3f} si aplica)" if len(alternatives) > 1 else "")

#  10.6 Verificacin cobertura sectorial 
# Reconstruir SECTORES con los tickers candidatos que llegan al clustering
SECTORES = construir_sectores(tickers_candidatos)

print(f"\n  [10.6] Cobertura sectorial de la seleccin:")
sectores_sin_repr = []
for sector, empresas in SECTORES.items():
    representadas  = [e for e in empresas if e in seleccionadas]
    en_candidatos  = [e for e in empresas if e in tickers_candidatos]
    if representadas:
        print(f"    [OK] [{sector:15s}]  {representadas}")
    elif en_candidatos:
        print(f"    - [{sector:15s}]  SIN REPRESENTACIN - candidatas: {en_candidatos}")
        sectores_sin_repr.append({'sector': sector, 'candidatas': en_candidatos})
    else:
        print(f"    - [{sector:15s}]  sin candidatas disponibles")

if sectores_sin_repr:
    print(f"\n  [AJUSTE SECTORIAL] Aadiendo representante de sectores descubiertos:")
    for s in sectores_sin_repr:
        mejor = scoring.loc[s['candidatas'], 'score'].idxmax()
        seleccionadas.append(mejor)
        print(f"    + {mejor:8s} (score={scoring.loc[mejor, 'score']:.3f}) "
              f"-> cubre sector '{s['sector']}'")
    print(f"\n  Seleccin tras ajuste sectorial: {len(seleccionadas)} empresas")
else:
    print(f"\n  [OK] Todos los sectores tienen al menos un representante.")

#  10.7 Filtrar dataset_long a empresas seleccionadas
dataset_long = dataset_long[
    dataset_long['Ticker'].isin(seleccionadas)
].reset_index(drop=True)

print(f"\n  Seleccin final ({len(seleccionadas)} empresas): {sorted(seleccionadas)}")
print(f"  Shape dataset_long tras seleccin: {dataset_long.shape}")

# Dendrograma - figura clave para la memoria del TFG
fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(
    linkage_matrix,
    labels=tickers_candidatos,
    orientation='top',
    leaf_rotation=45,
    ax=ax,
)
for label in ax.get_xticklabels():
    if label.get_text() in seleccionadas:
        label.set_color('darkgreen')
        label.set_fontweight('bold')
ax.set_title('Clustering Jerrquico Ward - DAX40 (verde = seleccionadas)', fontsize=12)
ax.set_ylabel('Distancia')
plt.tight_layout()
ruta_dendrograma = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '10_dendrograma_clustering.png'
)
plt.savefig(ruta_dendrograma, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  [OK] Dendrograma guardado: {ruta_dendrograma}")

ruta_excel = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '4_dataset_long_empresas_seleccionadas.xlsx'
)
try:
    dataset_long.to_excel(ruta_excel, index=False)
    print(f"  Dataset seleccionado guardado: {ruta_excel}")
except PermissionError:
    print(f"  [ADVERTENCIA] No se pudo guardar Excel (archivo bloqueado). Continuando...")
except Exception as e:
    print(f"  [ADVERTENCIA] Error al guardar Excel: {e}. Continuando...")


# =============================================================================
# 10B. VISUALIZACION: METRICAS DE SELECCION DE ACTIVOS
# =============================================================================
print("\n" + "="*70)
print("VISUALIZACION 10B: METRICAS DE SELECCION DE ACTIVOS")
print("="*70)

from matplotlib.lines import Line2D as _Line2D

scoring_sorted = scoring['score'].sort_values(ascending=True)
colores_barras = ['darkgreen' if t in seleccionadas else 'steelblue'
                  for t in scoring_sorted.index]
umbral_score = scoring_sorted[scoring_sorted.index.isin(seleccionadas)].min()

# Figura 10B-a: Score compuesto por empresa
fig_10ba, ax_10ba = plt.subplots(figsize=(10, max(6, len(scoring_sorted) * 0.3)))
ax_10ba.barh(
    [dax_company_names.get(t, t) for t in scoring_sorted.index],
    scoring_sorted.values,
    color=colores_barras, edgecolor='white', linewidth=0.5
)
ax_10ba.axvline(umbral_score, color='darkgreen', linestyle='--',
                linewidth=1.2, label=f'Umbral min. seleccion ({umbral_score:.2f})')
ax_10ba.set_title(
    'Score Compuesto por Empresa DAX40\n(verde = seleccionada para el modelo, azul = descartada)',
    fontsize=12, fontweight='bold')
ax_10ba.set_xlabel('Score (0-1)')
ax_10ba.legend(fontsize=8)
ax_10ba.grid(axis='x', alpha=0.3)
ax_10ba.tick_params(labelsize=7)
plt.tight_layout()
ruta_10ba = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '10b_score_compuesto_empresas.png')
plt.savefig(ruta_10ba, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: {ruta_10ba}")

# Figura 10B-b: Scatter volatilidad vs kurtosis por cluster
cmap_cl = plt.cm.tab20
max_cl  = int(cluster_df['Cluster'].max())
fig_10bb, ax_10bb = plt.subplots(figsize=(10, 7))
for ticker_sc in scoring.index:
    c_id = int(cluster_df.loc[ticker_sc, 'Cluster']) if ticker_sc in cluster_df.index else 0
    sel_sc   = ticker_sc in seleccionadas
    color_sc = cmap_cl(c_id / max(max_cl, 1))
    marker_sc = '*' if sel_sc else 'o'
    size_sc   = 160  if sel_sc else 35
    ax_10bb.scatter(
        scoring.loc[ticker_sc, 'volatilidad'],
        scoring.loc[ticker_sc, 'kurtosis'],
        color=color_sc, marker=marker_sc, s=size_sc,
        zorder=5 if sel_sc else 3, alpha=0.85
    )
    if sel_sc:
        ax_10bb.annotate(
            ticker_sc.replace('.DE', ''),
            (scoring.loc[ticker_sc, 'volatilidad'],
             scoring.loc[ticker_sc, 'kurtosis']),
            textcoords='offset points', xytext=(5, 3), fontsize=7
        )
legend_10bb = [
    _Line2D([0], [0], marker='*', color='gray', linestyle='None',
            markersize=11, label='Seleccionada'),
    _Line2D([0], [0], marker='o', color='gray', linestyle='None',
            markersize=6,  label='No seleccionada'),
]
ax_10bb.legend(handles=legend_10bb, fontsize=8)
ax_10bb.set_title(
    'Volatilidad vs Kurtosis por Empresa DAX40 — Clustering Jerarquico\n'
    '(estrella = empresa seleccionada para el modelo, color = cluster asignado)',
    fontsize=12, fontweight='bold')
ax_10bb.set_xlabel('Volatilidad (std Log Return)')
ax_10bb.set_ylabel('Kurtosis')
ax_10bb.grid(alpha=0.3)
plt.tight_layout()
ruta_10bb = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '10b_volatilidad_vs_kurtosis_clusters.png')
plt.savefig(ruta_10bb, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Guardado: {ruta_10bb}")


# =============================================================================
# 11. ANLISIS Y TRATAMIENTO DE OUTLIERS
# =============================================================================
# Los outliers se analizan y tratan SOBRE Log_Return (ya calculado en seccin 9)
# y DESPUS de la seleccin de empresas - trabajamos solo sobre las 13-15
# empresas que entran al modelo, no sobre las 35 candidatas.
#
# El tratamiento produce Log_Return_Wins: versin winsorizada de Log_Return
# que se aade como columna adicional en dataset_long.
# Esto permite al modelo elegir entre retornos originales y winzorizados,
# y facilita la comparacin de ambas configuraciones en la fase de entrenamiento.

print("\n" + "="*70)
print("SECCIN 11: ANLISIS Y TRATAMIENTO DE OUTLIERS")
print("="*70)

UMBRAL_ZSCORE = 3

# Pivot de Log_Return_D para anlisis matricial
retornos_modelo = (
    dataset_long[dataset_long['Ticker'].isin(seleccionadas)]
    .pivot(index='Date', columns='Ticker', values='Log_Return_D')
    .dropna()
)

#  11.1 Deteccin por z-score 
print(f"\n  [11.1] Deteccin de outliers por z-score (|z| > {UMBRAL_ZSCORE})...")

z_scores      = retornos_modelo.apply(stats.zscore)
mask_outliers = z_scores.abs() > UMBRAL_ZSCORE

outliers_resumen = pd.DataFrame({
    'n_outliers':   mask_outliers.sum(),
    'pct_outliers': (mask_outliers.sum() / len(retornos_modelo) * 100).round(3),
    'kurtosis':     retornos_modelo.apply(stats.kurtosis).round(2),
    'skewness':     retornos_modelo.apply(stats.skew).round(2),
    'ret_max':      retornos_modelo.max().round(4),
    'ret_min':      retornos_modelo.min().round(4),
}).sort_values('pct_outliers', ascending=False)

print(f"\n  {'Ticker':<12} {'N outliers':>12} {'% das':>10} "
      f"{'Kurtosis':>10} {'Skewness':>10} {'Ret max':>10} {'Ret min':>10}")
print("  " + "-"*76)
for ticker, row in outliers_resumen.iterrows():
    print(f"  {ticker:<12} {int(row['n_outliers']):>12} {row['pct_outliers']:>9.3f}% "
          f"{row['kurtosis']:>10.2f} {row['skewness']:>10.2f} "
          f"{row['ret_max']:>10.4f} {row['ret_min']:>10.4f}")

#  11.2 Deteccin por LOF 
print(f"\n  [11.2] Deteccin por Local Outlier Factor (LOF)...")

try:
    from sklearn.neighbors import LocalOutlierFactor

    lof_resultados = {}
    for ticker in seleccionadas:
        if ticker not in retornos_modelo.columns:
            continue
        serie_lof = pd.DataFrame({
            'ret_t':  retornos_modelo[ticker],
            'ret_t1': retornos_modelo[ticker].shift(1),
        }).dropna()
        lof  = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
        pred = lof.fit_predict(serie_lof)
        n_out = (pred == -1).sum()
        lof_resultados[ticker] = {
            'n_lof': n_out,
            'pct_lof': round(n_out / len(serie_lof) * 100, 3)
        }

    lof_df = pd.DataFrame(lof_resultados).T.sort_values('pct_lof', ascending=False)
    print(f"\n  {'Ticker':<12} {'N LOF':>10} {'% LOF':>10}")
    print("  " + "-"*34)
    for ticker, row in lof_df.iterrows():
        print(f"  {ticker:<12} {int(row['n_lof']):>10} {row['pct_lof']:>9.3f}%")

except ImportError:
    print("    [WARN] scikit-learn no disponible - LOF omitido")

#  11.3 Boxplot visual 
print(f"\n  [11.3] Generando boxplot de outliers...")

ticker_top = outliers_resumen.index[0]
serie      = retornos_modelo[ticker_top]
mask_out   = mask_outliers[ticker_top]

# Figura 11-a: Boxplot de distribuciones por activo
fig_11a, ax_11a = plt.subplots(figsize=(16, 7))
retornos_modelo.boxplot(
    ax=ax_11a, rot=45,
    flierprops=dict(marker='o', markerfacecolor='red',
                    markersize=2.5, alpha=0.4, linestyle='none'),
    medianprops=dict(color='darkblue', linewidth=1.5),
    boxprops=dict(color='steelblue'),
    whiskerprops=dict(color='steelblue'),
    capprops=dict(color='steelblue'),
)
ax_11a.set_title(
    'Distribucion de Log_Return Diario por Activo — Deteccion de Outliers (Tukey IQR)\n'
    '(puntos rojos = outliers, anotaciones = % de dias clasificados como outlier)',
    fontsize=12, fontweight='bold')
ax_11a.set_ylabel('Log_Return diario')
ax_11a.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax_11a.grid(axis='y', alpha=0.3)
for i, ticker in enumerate(retornos_modelo.columns):
    pct = outliers_resumen.loc[ticker, 'pct_outliers'] if ticker in outliers_resumen.index else 0
    ax_11a.text(i + 1, retornos_modelo[ticker].max() * 1.05,
                f'{pct:.1f}%', ha='center', va='bottom',
                fontsize=7, color='darkred', rotation=45)
plt.tight_layout()
ruta_11a = os.path.join(os.path.dirname(os.path.abspath(__file__)), '11_3a_boxplot_outliers.png')
plt.savefig(ruta_11a, dpi=150, bbox_inches='tight')
plt.close()
print(f"    [OK] Boxplot guardado: {ruta_11a}")

# Figura 11-b: Serie temporal del activo con mayor % de outliers
fig_11b, ax_11b = plt.subplots(figsize=(16, 5))
ax_11b.plot(serie.index, serie.values, color='steelblue',
            linewidth=0.7, alpha=0.8, label='Log_Return diario')
ax_11b.scatter(serie[mask_out].index, serie[mask_out].values,
               color='red', s=15, zorder=5, alpha=0.7,
               label=f'Outliers (|z|>{UMBRAL_ZSCORE}): {mask_out.sum()} dias')
ax_11b.fill_between(serie.index,
                    retornos_modelo[ticker_top].quantile(0.01),
                    retornos_modelo[ticker_top].quantile(0.99),
                    alpha=0.08, color='green', label='Banda P1%-P99%')
ax_11b.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax_11b.set_title(
    f'Serie Temporal de Retornos — {ticker_top} (mayor % de outliers: {outliers_resumen.loc[ticker_top, "pct_outliers"]:.2f}%)\n'
    f'(Outliers marcados en rojo, banda verde = rango normal P1%-P99%)',
    fontsize=12, fontweight='bold')
ax_11b.set_ylabel('Log_Return diario')
ax_11b.legend(fontsize=9)
ax_11b.grid(alpha=0.3)
plt.tight_layout()
ruta_11b_ot = os.path.join(os.path.dirname(os.path.abspath(__file__)), '11_3b_serie_temporal_outliers.png')
plt.savefig(ruta_11b_ot, dpi=150, bbox_inches='tight')
plt.close()
print(f"    [OK] Serie temporal outliers guardada: {ruta_11b_ot}")

#  11.4 Winsorizacin definitiva sobre Log_Return 
# Aplicamos winsorizacin P1%-P99% sobre Log_Return y guardamos el resultado
# como Log_Return_Wins en dataset_long.
#
# Por qu no sobreescribimos Log_Return?
#   Conservar ambas columnas permite comparar en la fase de entrenamiento:
#     Config A: Log_Return original + Huber Loss
#     Config B: Log_Return_Wins     + MSE
#   La comparacin de ambas configuraciones es un resultado del TFG.

print(f"\n  [11.4] Aplicando winsorizacion definitiva (P1%-P99%) sobre Log_Return...")

# MODIFICACIÓN NUEVA: Calcular percentiles de winsorizacion SOLO sobre datos de
# entrenamiento (80% temporal) para evitar look-ahead bias. Antes se usaba el
# dataset completo (incluyendo test), lo que significaba que los umbrales P1%/P99%
# incorporaban informacion futura. Ahora los umbrales se calculan en train y se
# aplican a todo el dataset, simulando el escenario real de produccion.
_fecha_min_w = dataset_long['Date'].min()
_fecha_max_w = dataset_long['Date'].max()
_fecha_corte_w = _fecha_min_w + pd.Timedelta(days=int((_fecha_max_w - _fecha_min_w).days * 0.80))
_train_mask_w = dataset_long['Date'] <= _fecha_corte_w

# Calcular percentiles solo en train
_train_percentiles = {}
for _ticker_w in seleccionadas:
    _t_mask = (dataset_long['Ticker'] == _ticker_w) & _train_mask_w
    _train_rets = dataset_long.loc[_t_mask, 'Log_Return_D'].dropna()
    _train_percentiles[_ticker_w] = (_train_rets.quantile(0.01), _train_rets.quantile(0.99))

# Aplicar umbrales de train a todo el dataset
dataset_long['Log_Return_Wins'] = dataset_long['Log_Return_D'].copy()
for _ticker_w in seleccionadas:
    _t_mask_all = dataset_long['Ticker'] == _ticker_w
    _p_lo, _p_hi = _train_percentiles[_ticker_w]
    dataset_long.loc[_t_mask_all, 'Log_Return_Wins'] = (
        dataset_long.loc[_t_mask_all, 'Log_Return_D'].clip(lower=_p_lo, upper=_p_hi)
    )
print(f"    [INFO] Percentiles calculados sobre train (hasta {_fecha_corte_w})")

# Verificacion: kurtosis antes y despues
# IMPORTANTE: Dropear NaN antes de calcular kurtosis (primeros valores son NaN por rolling window)
kurt_antes  = dataset_long.groupby('Ticker')['Log_Return_D'].apply(lambda x: stats.kurtosis(x.dropna())).round(2)
kurt_despues = dataset_long.groupby('Ticker')['Log_Return_Wins'].apply(lambda x: stats.kurtosis(x.dropna())).round(2)

print(f"\n  Comparativa kurtosis antes/despus de winsorizacin:")
print(f"  {'Ticker':<12} {'Kurtosis pre':>14} {'Kurtosis post':>14} {'Reduccin':>12}")
print("  " + "-"*54)
for ticker in sorted(seleccionadas):
    if ticker in kurt_antes.index and ticker in kurt_despues.index:
        k_pre  = kurt_antes[ticker]
        k_post = kurt_despues[ticker]
        red    = k_pre - k_post
        print(f"  {ticker:<12} {k_pre:>14.2f} {k_post:>14.2f} {red:>11.2f}")

# GRAFICO 11.4 - Before/After Winsorizacion
print(f"\n  [11.4.2] Visualizacion before/after winsorizacion...")
rets_antes = dataset_long['Log_Return_D'].dropna()
rets_despues = dataset_long['Log_Return_Wins'].dropna()

# Calcular kurtosis (ya sin NaN tras dropna)
kurt_global_antes = stats.kurtosis(rets_antes) if len(rets_antes) > 0 else np.nan
kurt_global_despues = stats.kurtosis(rets_despues) if len(rets_despues) > 0 else np.nan

# Figura 11.4-a: Distribucion ANTES de la winsorizacion
fig_wins_a, ax_wins_a = plt.subplots(figsize=(8, 6))
ax_wins_a.hist(rets_antes, bins=80, alpha=0.7, color='coral', edgecolor='black')
ax_wins_a.set_title(
    f'Distribucion de Retornos Diarios ANTES de la Winsorizacion (Log_Return_D original)\n'
    f'Kurtosis={kurt_global_antes:.2f} | Extremos: [{rets_antes.min():.4f}, {rets_antes.max():.4f}]',
    fontsize=11, fontweight='bold')
ax_wins_a.set_xlabel('Retorno diario', fontsize=9)
ax_wins_a.set_ylabel('Frecuencia', fontsize=9)
ax_wins_a.grid(alpha=0.3)
ax_wins_a.axvline(rets_antes.mean(), color='darkred', linestyle='--', linewidth=2, label='Media')
ax_wins_a.axvline(rets_antes.quantile(0.01), color='red', linestyle=':', linewidth=1.5, label='P1%')
ax_wins_a.axvline(rets_antes.quantile(0.99), color='red', linestyle=':', linewidth=1.5, label='P99%')
ax_wins_a.legend(fontsize=8)
plt.tight_layout()
ruta_wins_a = os.path.join(os.path.dirname(os.path.abspath(__file__)), '11_4a_retornos_antes_winsorizacion.png')
plt.savefig(ruta_wins_a, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Grafica ANTES guardada: {ruta_wins_a}")

# Figura 11.4-b: Distribucion DESPUES de la winsorizacion
fig_wins_b, ax_wins_b = plt.subplots(figsize=(8, 6))
ax_wins_b.hist(rets_despues, bins=80, alpha=0.7, color='lightgreen', edgecolor='black')
ax_wins_b.set_title(
    f'Distribucion de Retornos Diarios DESPUES de la Winsorizacion P1%-P99% (Log_Return_Wins)\n'
    f'Kurtosis={kurt_global_despues:.2f} | Reduccion: {kurt_global_antes - kurt_global_despues:.2f} puntos',
    fontsize=11, fontweight='bold')
ax_wins_b.set_xlabel('Retorno diario', fontsize=9)
ax_wins_b.set_ylabel('Frecuencia', fontsize=9)
ax_wins_b.grid(alpha=0.3)
ax_wins_b.axvline(rets_despues.mean(), color='darkgreen', linestyle='--', linewidth=2, label='Media')
ax_wins_b.legend(fontsize=8)
plt.tight_layout()
ruta_wins_b = os.path.join(os.path.dirname(os.path.abspath(__file__)), '11_4b_retornos_despues_winsorizacion.png')
plt.savefig(ruta_wins_b, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Grafica DESPUES guardada: {ruta_wins_b}")
print(f"       Reduccion de kurtosis: {kurt_global_antes - kurt_global_despues:.2f}")

#  11.5 Resumen final 
print(f"\n  [11.5] Resumen final:")
print(f"    Columnas en dataset_long: {dataset_long.columns.tolist()}")
print(f"    Shape final             : {dataset_long.shape}")
print(f"    Empresas seleccionadas  : {sorted(seleccionadas)}")

nulos_log = dataset_long[['Log_Return_D', 'Log_Return_Wins', 'Std_D', 'Std_M']].isna().sum()
print(f"\n  Nulos en columnas calculadas (esperados por ventanas):")
print(nulos_log.to_string())

ruta_excel = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '7_dataset_final_completo.xlsx'
)
try:
    dataset_long.to_csv(ruta_excel.replace('.xlsx', '.csv'), index=False)
    dataset_long.to_excel(ruta_excel, index=False)
    print(f"  [OK] Dataset final completo guardado: {ruta_excel}")
except PermissionError:
    print(f"  [ADVERTENCIA] No se pudo guardar Excel (archivo bloqueado). Continuando...")
except Exception as e:
    print(f"  [ADVERTENCIA] Error al guardar Excel: {e}. Continuando...")

print(f"\n  Pipeline de limpieza, transformacin y seleccin completado.")
print(f"  Prximo paso: construccin del modelo ML (MLPRegressor / Markowitz).")


# =============================================================================
# 11B. VISUALIZACION: HISTOGRAMAS PRE/POST WINZORIZACION
# =============================================================================
print("\n" + "="*70)
print("VISUALIZACION 11B: HISTOGRAMAS PRE/POST WINZORIZACION (Log Return)")
print("="*70)

# Detectar nombre real de la columna de retorno diario
_col_ret  = next((c for c in ['Log_Return_D', 'Log_Return']
                  if c in dataset_long.columns), None)
_col_wins = 'Log_Return_Wins' if 'Log_Return_Wins' in dataset_long.columns else None

if _col_ret is None:
    print("  [WARN] No se encontro columna Log_Return -- histogramas omitidos")
else:
    _n_sel  = len(seleccionadas)
    _ncols  = 3
    _nrows  = (_n_sel + _ncols - 1) // _ncols
    fig_w, axes_w = plt.subplots(_nrows, _ncols,
                                  figsize=(5 * _ncols, 4 * _nrows))
    _axes_flat = axes_w.flatten() if _n_sel > 1 else [axes_w]

    _titulo_w = ('Distribucion Log Return Diario: Pre vs Post Winzorizacion (P1%-P99%)'
                 if _col_wins
                 else 'Distribucion Log Return Diario (pre-winzorizacion)')
    fig_w.suptitle(_titulo_w, fontsize=12, fontweight='bold')

    for _i, _ticker in enumerate(sorted(seleccionadas)):
        _ax = _axes_flat[_i]
        _sub = dataset_long[dataset_long['Ticker'] == _ticker].dropna(
            subset=[_col_ret])
        if _sub.empty:
            _ax.set_visible(False)
            continue

        _ret_pre = _sub[_col_ret].values
        _p1  = np.percentile(_ret_pre, 1)
        _p99 = np.percentile(_ret_pre, 99)

        _ax.hist(_ret_pre, bins=60, alpha=0.55, color='tomato',
                 label='Pre-wins', density=True, edgecolor='none')

        _kurt_str = f'Kurt: {stats.kurtosis(_ret_pre):.1f}'
        if _col_wins and _col_wins in _sub.columns:
            _ret_post = _sub[_col_wins].dropna().values
            _ax.hist(_ret_post, bins=60, alpha=0.55, color='steelblue',
                     label='Post-wins', density=True, edgecolor='none')
            _kurt_str = (f'Kurt: {stats.kurtosis(_ret_pre):.1f}'
                         f' -> {stats.kurtosis(_ret_post):.1f}')

        _ax.axvline(_p1,  color='darkred', linestyle='--', linewidth=1,
                    alpha=0.8, label=f'P1% = {_p1:.3f}')
        _ax.axvline(_p99, color='darkred', linestyle='--', linewidth=1,
                    alpha=0.8, label=f'P99% = {_p99:.3f}')

        _ax.set_title(f'{_ticker}\n{_kurt_str}', fontsize=9)
        _ax.set_xlabel('Log Return diario', fontsize=7)
        _ax.set_ylabel('Densidad', fontsize=7)
        _ax.tick_params(labelsize=6)
        _ax.legend(fontsize=6)
        _ax.grid(alpha=0.3)

    for _j in range(_i + 1, len(_axes_flat)):
        _axes_flat[_j].set_visible(False)

    plt.tight_layout()
    ruta_11b = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '11b_histogramas_winzorizacion.png')
    plt.savefig(ruta_11b, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Guardado: {ruta_11b}")

print("\n  Resumen de visualizaciones generadas:")
print("    7b_series_gaps_rellenados.png              -- series temporales gaps")
print("    8b_heatmap_nulos_antes.png                 -- heatmap nulos ANTES (por ticker/mes)")
print("    8b_heatmap_nulos_despues.png               -- heatmap nulos DESPUES (por ticker/mes)")
print("    8c_heatmap_nulos_macro_antes.png           -- heatmap nulos macro ANTES")
print("    8c_heatmap_nulos_macro_despues.png         -- heatmap nulos macro DESPUES")
print("    10b_score_compuesto_empresas.png           -- score compuesto por empresa")
print("    10b_volatilidad_vs_kurtosis_clusters.png   -- scatter volatilidad vs kurtosis")
print("    11_3a_boxplot_outliers.png                 -- boxplot distribuciones por activo")
print("    11_3b_serie_temporal_outliers.png          -- serie temporal activo con mas outliers")
print("    11_4a_retornos_antes_winsorizacion.png     -- histograma retornos originales")
print("    11_4b_retornos_despues_winsorizacion.png   -- histograma retornos winzorizados")
print("    11b_histogramas_winzorizacion.png          -- distribuciones pre/post wins (grid)")
print("    12_4a_matriz_correlaciones_pearson.png     -- heatmap de correlaciones")
print("    12_4b_dendrograma_correlaciones.png        -- dendrograma de activos")


# =============================================================================
# 12. ANALISIS EXPLORATORIO DE SERIES TEMPORALES (EDA TEMPORAL)
# =============================================================================
#
# MOTIVACION GENERAL
# Un modelo de machine learning (MLPRegressor) aprende patrones temporales en los datos.
# Para disenarlo bien, necesitamos responder previamente cuatro preguntas
# fundamentales sobre la estructura estadistica de las series:
#
#   1. Â¿Tienen las series de precios componente estacional / tendencial?
#      Respuesta: Descomposicion STL  -> seccion 12.1
#
#   2. Â¿Son los retornos independientes o hay memoria lineal?
#      Respuesta: Test Ljung-Box + ACF -> seccion 12.2
#
#   3. Â¿Es constante la varianza de los retornos o hay clusters de volatilidad?
#      Respuesta: Test ARCH de Engle + volatilidad rodante -> seccion 12.3
#
#   4. Â¿Que estructura de dependencia temporal tienen los retornos?
#      Â¿Siguen una distribucion Normal?
#      Respuesta: ACF/PACF + Q-Q plot + test Jarque-Bera -> seccion 12.4
#
# Todos los analisis sobre retornos usan Log_Return_Wins (version winsorizada)
# porque es exactamente la columna que entrara al modelo. Usar los datos "reales"
# del modelo garantiza coherencia entre EDA y entrenamiento.

print("\n" + "="*70)
print("SECCION 12: ANALISIS EXPLORATORIO DE SERIES TEMPORALES (EDA TEMPORAL)")
print("="*70)
print(f"\n  Analisis comprehensivo de TODAS las {len(seleccionadas)} empresas seleccionadas")

# Columna de retornos utilizada en el EDA
COL_RET  = 'Log_Return_Wins'   # retorno diario winsorizado
COL_PREC = 'Adj Close'         # precio ajustado (para STL)
NLAGS    = 40                  # lags en ACF/PACF: 40 dias â‰ˆ 2 meses de trading


# =============================================================================
# 12.1  DESCOMPOSICION ESTACIONAL -- STL
#        (Seasonal and Trend decomposition using Loess)
# =============================================================================
#
# Â¿QUE DESCOMPONEMOS?
# La serie de PRECIOS (Adj Close), NO los retornos.
# Los precios exhiben tendencia (drift alcista estructural de la renta variable)
# y potencialmente patrones estacionales anuales: efecto enero, verano bursatil,
# cierre de posiciones en diciembre, etc.
# Los retornos (diferencias logaritmicas) ya eliminan la tendencia, por lo que
# su descomposicion estacional seria trivialmente plana y poco informativa.
#
# Â¿POR QUE STL Y NO seasonal_decompose CLASICO?
# seasonal_decompose usa medias moviles simetricas simples: cualquier outlier
# extremo (crash COVID marzo-2020, crisis energetica oct-2022) sesga la
# estimacion de la tendencia durante ventanas completas.
# STL (Cleveland et al. 1990) aplica suavizados LOESS iterativos y dispone del
# parametro robust=True, que asigna menor peso a las observaciones atipicas.
# Es el estandar actual en analisis de series financieras y macroeconomicas y
# el metodo que recomiendan tanto Hyndman & Athanasopoulos (Forecasting: P&P)
# como la literatura reciente de deep learning para finanzas.
#
# PERIODO = 252 dias de trading â‰ˆ 1 ano natural.
# Con ~5 anos de datos por ticker (â‰¥1 260 observaciones) disponemos de mas de
# 5 ciclos completos, muy por encima del minimo de 2 que necesita STL para
# estimar de forma estable la componente estacional anual.
# Alternativas validas serian periodo=22 (ciclo mensual) o periodo=5 (ciclo
# semanal), pero para una estrategia de inversion a medio plazo como la del
# TFG, el ciclo anual es el mas relevante economicamente.

print(f"\n  [12.1] Descomposicion estacional STL (Adj Close, periodo=252 - TODAS las empresas)...")

# Tabla resumen de tendencia y amplitud estacional para TODAS las empresas
_stl_results = {}
print(f"\n  {'Ticker':<12} {'Obs':>8} {'Periodo':>8} {'Tend. Amp.':>12} {'Est. Amp.':>12} {'Res. Vol':>12}")
print("  " + "-"*68)

for _tick in sorted(seleccionadas):
    _df_ser = (dataset_long[dataset_long['Ticker'] == _tick]
               .set_index('Date')[COL_PREC]
               .dropna()
               .sort_index())
    
    if len(_df_ser) < 2 * 252:
        _periodo = 22
    else:
        _periodo = 252
    
    try:
        _stl_m = STL(_df_ser, period=_periodo, robust=True)
        _stl_r = _stl_m.fit()
        
        # Amplitud de componentes = rango (max - min)
        _trend_amp = _stl_r.trend.max() - _stl_r.trend.min()
        _seas_amp = _stl_r.seasonal.max() - _stl_r.seasonal.min()
        _resid_vol = _stl_r.resid.std()
        
        _stl_results[_tick] = (_stl_m, _stl_r, _periodo, _trend_amp, _seas_amp, _resid_vol)
        
        print(f"  {_tick:<12} {len(_df_ser):>8} {_periodo:>8} {_trend_amp:>12.2f} "
              f"{_seas_amp:>12.4f} {_resid_vol:>12.6f}")
    except Exception as e:
        print(f"  {_tick:<12} [ERROR]: {str(e)[:40]}")

# GRAFICO 12.1 -- Grid compacto de STL para TODAS las empresas
# 3 empresas por fila, mostrando solo Tendencia + Estacional + Residuos
# (omitimos "Observado" para ahorrar espacio; la tendencia lo representa bien)
_n_tickers = len(_stl_results)
_ncols_stl = 3
_nrows_stl = (_n_tickers + _ncols_stl - 1) // _ncols_stl

fig_stl, axes_stl = plt.subplots(_nrows_stl, _ncols_stl, figsize=(16, 4 * _nrows_stl))
_axes_stl_flat = axes_stl.flatten() if _n_tickers > 1 else [axes_stl]
fig_stl.suptitle(
    'Descomposicion STL (Tendencia + Estacional + Residuos) -- TODAS las empresas',
    fontsize=13, fontweight='bold', y=0.995
)

for _idx, (_tick, (_stl_m, _stl_r, _per, _t_amp, _s_amp, _r_vol)) in enumerate(sorted(_stl_results.items())):
    _ax = _axes_stl_flat[_idx]
    
    # Extraer indices para graficar
    _idx_dates = _stl_r.trend.index
    
    # Graficar 3 componentes en escala normalizada (para comparabilidad visual)
    _ax.plot(_idx_dates, _stl_r.trend.values, label='Tendencia', color='darkorange', linewidth=1.0, alpha=0.8)
    _ax.plot(_idx_dates, _stl_r.seasonal.values, label='Estacional', color='seagreen', linewidth=0.7, alpha=0.7)
    _ax.plot(_idx_dates, _stl_r.resid.values, label='Residuos', color='gray', linewidth=0.5, alpha=0.5)
    
    _ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    _ax.set_title(f'{_tick}  (p={_per}, T-amp={_t_amp:.1f}, S-amp={_s_amp:.4f}, R-vol={_r_vol:.5f})',
                  fontsize=8, fontweight='bold')
    _ax.set_xlabel('Fecha', fontsize=7)
    _ax.set_ylabel('Valor', fontsize=7)
    _ax.tick_params(labelsize=6)
    _ax.grid(alpha=0.2)
    _ax.legend(fontsize=6, loc='best')

# Ocultar ejes no usados
for _j in range(_idx + 1, len(_axes_stl_flat)):
    _axes_stl_flat[_j].axis('off')

plt.tight_layout()
ruta_stl = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '12_1_descomposicion_stl.png')
plt.savefig(ruta_stl, dpi=150, bbox_inches='tight')
plt.close()
print(f"    [OK] Guardado: {ruta_stl}")


# =============================================================================
# 12.2  AUTOCORRELACION -- Test de Ljung-Box + Grafico ACF de retornos
# =============================================================================
#
# Â¿QUE MEDIMOS?
# La autocorrelacion de orden k mide si el retorno del dia t esta correlacionado
# linealmente con el retorno del dia tâˆ’k.
# La Hipotesis del Mercado Eficiente (EMH) en su forma debil implica que los
# retornos no tienen autocorrelacion -> ningun patron pasado es explotable.
# En la practica, los retornos del DAX muestran autocorrelaciones pequenas pero
# estadisticamente significativas a lags cortos (1-5 dias), lo que motiva el uso
# de un MLPRegressor que puede capturar esas dependencias no lineales.
#
# Â¿POR QUE EL TEST DE LJUNG-BOX?
# El test de Ljung-Box (1978) evalua simultaneamente si los primeros K
# coeficientes de autocorrelacion son conjuntamente cero:
#   Hâ‚€: Ï(1) = Ï(2) = ... = Ï(K) = 0  (no hay autocorrelacion en los K primeros lags)
#   Hâ‚: al menos una Ï(k) â‰  0
# Se prefiere sobre Box-Pierce porque su estadistico tiene mejor aproximacion
# chiÂ² en muestras finitas (Ljung & Box, 1978, Biometrika).
# Lags habituales en series financieras diarias: 10 (â‰ˆ2 semanas) y 20 (â‰ˆ1 mes).
#
# Â¿POR QUE EL GRAFICO ACF ADEMAS DEL TEST?
# Ljung-Box da un unico p-valor agregado para K lags.
# El grafico ACF muestra el patron lag a lag: podemos identificar si la
# autocorrelacion se concentra en lag 5 (efecto dia de la semana), lag 22
# (efecto mensual) o decae exponencialmente (proceso AR).
# Esta informacion es relevante para calibrar la ventana temporal del MLPRegressor.

print(f"\n  [12.2] Test de Ljung-Box y grafico ACF de retornos...")

_resultados_lb = {}
for _tick in sorted(seleccionadas):
    _serie = (dataset_long[dataset_long['Ticker'] == _tick][COL_RET]
              .dropna().values)
    if len(_serie) < 50:
        continue
    # return_df=True devuelve un DataFrame con columnas lb_stat y lb_pvalue
    _lb = acorr_ljungbox(_serie, lags=[10, 20], return_df=True)
    _resultados_lb[_tick] = _lb

print(f"\n  {'Ticker':<12} {'LB(10) p-val':>13} {'LB(20) p-val':>13} {'Autocorrelación':>11}")
print("  " + "-"*52)
for _tick, _lb in _resultados_lb.items():
    _pv10 = _lb['lb_pvalue'].iloc[0]   # p-valor acumulado en lag 10
    _pv20 = _lb['lb_pvalue'].iloc[1]   # p-valor acumulado en lag 20
    # Significativo al 5% -> hay estructura de autocorrelacion explotable
    _sig  = 'SI (*)' if _pv10 < 0.05 else 'NO'
    print(f"  {_tick:<12} {_pv10:>13.4f} {_pv20:>13.4f} {_sig:>11}")

# GRAFICO 12.2 -- Grid de ACF para todas las empresas seleccionadas
# Cada panel muestra los coeficientes de autocorrelacion Ï(k) con bandas de
# confianza al 95% (lineas azules Â±1.96/âˆšn). Barras que sobrepasan la banda
# son estadisticamente significativas a nivel individual.
# Se muestra el grid completo (no solo el ticker de referencia) porque la
# autocorrelacion varia entre sectores: las empresas financieras (DBK, MUV2)
# suelen mostrar mas persistencia que las industriales (BMW, MBG).
_n_sel       = len(seleccionadas)
_ncols_acf   = 3
_nrows_acf   = (_n_sel + _ncols_acf - 1) // _ncols_acf
fig_acf, axes_acf = plt.subplots(_nrows_acf, _ncols_acf,
                                  figsize=(5 * _ncols_acf, 3.5 * _nrows_acf))
_axes_acf_flat = axes_acf.flatten() if _n_sel > 1 else [axes_acf]
fig_acf.suptitle(
    f'ACF de Log_Return_Wins ({NLAGS} lags) -- empresas seleccionadas',
    fontsize=12, fontweight='bold'
)

for _i, _tick in enumerate(sorted(seleccionadas)):
    _ax  = _axes_acf_flat[_i]
    _ser = (dataset_long[dataset_long['Ticker'] == _tick][COL_RET]
            .dropna().values)
    if len(_ser) < NLAGS + 5:
        _ax.set_visible(False)
        continue
    plot_acf(_ser, lags=NLAGS, ax=_ax, title='', alpha=0.05,
             zero=False,
             color='steelblue', vlines_kwargs={'colors': 'steelblue'})
    _ax.set_ylim(-0.15, 0.15)
    # Anotar el p-valor de Ljung-Box(10) en el titulo del panel
    _pv_str = (f"  LB10 p={_resultados_lb[_tick]['lb_pvalue'].iloc[0]:.3f}"
               if _tick in _resultados_lb else '')
    _ax.set_title(f'{_tick}{_pv_str}', fontsize=8)
    _ax.set_xlabel('Lag (dias)', fontsize=7)
    _ax.set_ylabel('ACF', fontsize=7)
    _ax.tick_params(labelsize=6)
    _ax.grid(alpha=0.3)

for _j in range(_i + 1, len(_axes_acf_flat)):
    _axes_acf_flat[_j].set_visible(False)

plt.tight_layout()
ruta_acf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '12_2_autocorrelacion_acf.png')
plt.savefig(ruta_acf, dpi=150, bbox_inches='tight')
plt.close()
print(f"    [OK] Guardado: {ruta_acf}")

# -------------------------------------------------------------------------
# GRAFICO 12.2B -- PACF (Partial Autocorrelation Function)
# -------------------------------------------------------------------------
# ¿QUE ES LA PACF Y EN QUE SE DIFERENCIA DE LA ACF?
#   ACF(k)  = correlacion entre r(t) y r(t-k), INCLUYENDO la influencia
#             indirecta de los lags intermedios (r(t-1), ..., r(t-k+1)).
#   PACF(k) = correlacion entre r(t) y r(t-k) UNA VEZ AISLADA la influencia
#             de los lags intermedios (correlacion parcial pura).
#
# ¿POR QUE ES RELEVANTE PARA ESTE TFG?
#   1) La PACF revela cuantos lags tienen influencia DIRECTA sobre el retorno
#      actual. Esto ayuda a dimensionar la ventana temporal de features del
#      MLPRegressor: si la PACF se corta en lag p, un input de p dias es
#      suficiente y usar mas lags solo anade ruido.
#   2) Un corte abrupto en la PACF tras lag p sugiere un proceso AR(p) puro.
#      Si la PACF decae gradualmente -> proceso ARMA mas complejo, donde
#      el MLP tiene ventaja sobre modelos lineales AR.
#   3) Lags individuales significativos en PACF (ej: lag 5 = semanal) revelan
#      microestructura del mercado explotable como feature.
#
# METODO: Yule-Walker (statsmodels default), bandas al 95%.

print(f"\n  [12.2B] Grafico PACF de Log_Return_Wins...")

fig_pacf, axes_pacf = plt.subplots(_nrows_acf, _ncols_acf,
                                    figsize=(5 * _ncols_acf, 3.5 * _nrows_acf))
_axes_pacf_flat = axes_pacf.flatten() if _n_sel > 1 else [axes_pacf]
fig_pacf.suptitle(
    f'PACF de Log_Return_Wins ({NLAGS} lags) -- empresas seleccionadas',
    fontsize=12, fontweight='bold'
)

_pacf_signif = {}  # almacenar lags significativos por ticker
for _i, _tick in enumerate(sorted(seleccionadas)):
    _ax  = _axes_pacf_flat[_i]
    _ser = (dataset_long[dataset_long['Ticker'] == _tick][COL_RET]
            .dropna().values)
    if len(_ser) < NLAGS + 5:
        _ax.set_visible(False)
        continue
    plot_pacf(_ser, lags=NLAGS, ax=_ax, title='', alpha=0.05,
              method='ywm', zero=False,
              color='steelblue', vlines_kwargs={'colors': 'steelblue'})
    _ax.set_ylim(-0.15, 0.15)

    # Contar lags significativos (fuera de banda +-1.96/sqrt(n))
    from statsmodels.tsa.stattools import pacf as _pacf_func
    _pacf_vals = _pacf_func(_ser, nlags=NLAGS, method='ywm')
    _banda = 1.96 / np.sqrt(len(_ser))
    _lags_signif = [k for k in range(1, NLAGS + 1) if abs(_pacf_vals[k]) > _banda]
    _pacf_signif[_tick] = _lags_signif

    _n_sig = len(_lags_signif)
    _ax.set_title(f'{_tick}  ({_n_sig} lags signif.)', fontsize=8)
    _ax.set_xlabel('Lag (dias)', fontsize=7)
    _ax.set_ylabel('PACF', fontsize=7)
    _ax.tick_params(labelsize=6)
    _ax.grid(alpha=0.3)

for _j in range(_i + 1, len(_axes_pacf_flat)):
    _axes_pacf_flat[_j].set_visible(False)

plt.tight_layout()
ruta_pacf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '12_2b_autocorrelacion_pacf.png')
plt.savefig(ruta_pacf, dpi=150, bbox_inches='tight')
plt.close()
print(f"    [OK] Guardado: {ruta_pacf}")

# TABLA 12.2B -- Resumen PACF: lags con influencia directa significativa
print(f"\n  [12.2B] Lags con autocorrelacion parcial significativa (alpha=0.05):")
print(f"  {'Ticker':<12} {'N lags':>8} {'Lags significativos'}")
print("  " + "-" * 60)
_total_lags_signif = []
for _tick in sorted(seleccionadas):
    _lags = _pacf_signif.get(_tick, [])
    _total_lags_signif.append(len(_lags))
    _lags_str = str(_lags) if _lags else '(ninguno)'
    print(f"  {_tick:<12} {len(_lags):>8}   {_lags_str}")

_media_lags = np.mean(_total_lags_signif)
_max_lags_tick = sorted(seleccionadas)[np.argmax(_total_lags_signif)]
print(f"\n  Resumen:")
print(f"    - Media de lags significativos: {_media_lags:.1f}")
print(f"    - Ticker con mas memoria: {_max_lags_tick} ({max(_total_lags_signif)} lags)")
if _media_lags <= 3:
    print(f"    - INTERPRETACION: La mayoria de activos muestra pocos lags directos")
    print(f"      significativos -> estructura AR de orden bajo. Una ventana de")
    print(f"      5-10 dias como input del MLPRegressor es suficiente.")
elif _media_lags <= 8:
    print(f"    - INTERPRETACION: Memoria directa moderada. Una ventana de")
    print(f"      10-20 dias para el MLPRegressor capturaria la mayor parte.")
else:
    print(f"    - INTERPRETACION: Memoria directa extensa. Considerar ventanas")
    print(f"      de 20-40 dias para el MLPRegressor.")


# =============================================================================
# 12.3  HETEROCEDASTICIDAD -- Test ARCH de Engle + Volatilidad Rodante
# =============================================================================
#
# Â¿QUE ES LA HETEROCEDASTICIDAD EN SERIES FINANCIERAS?
# Una serie es heterocedastica cuando su varianza no es constante en el tiempo.
# En finanzas esto se manifiesta como "clusters de volatilidad": periodos de
# alta volatilidad (crisis) seguidos de periodos de calma, propiedad
# documentada desde Mandelbrot (1963) y formalizada por Engle (1982) con ARCH.
# Los retornos del DAX mostraron esta propiedad especialmente en:
#   â€“ Mar 2020 (COVID): volatilidad diaria > 5 veces la normal
#   â€“ Feb-Oct 2022 (inflacion + guerra): volatilidad sostenida alta
#
# Â¿POR QUE ES RELEVANTE PARA ESTE TFG?
# 1) Markowitz clasico usa la matriz de covarianzas historica estatica.
#    Si la heterocedasticidad es significativa, esa matriz subestima el riesgo
#    en periodos de crisis -> justifica usar covarianzas rodantes (ventana deslizante)
#    en vez de covarianzas fijas en la optimizacion de cartera.
# 2) La funcion de perdida MSE en el MLPRegressor penaliza igual un error de 0.01
#    en un dia tranquilo que en un dia volatil. La Huber Loss es mas robusta
#    a esta asimetria. La presencia de heterocedasticidad confirma que Huber
#    es mas adecuada que MSE -> respalda la eleccion de Config A del TFG.
#
# Â¿POR QUE EL TEST ARCH DE ENGLE (LM test)?
# El test LM de Engle (1982) regresa los residuos al cuadrado sobre sus K
# propios retardos. Si los coeficientes son conjuntamente significativos ->
# la varianza pasada predice la varianza futura = efectos ARCH.
#   Hâ‚€: no hay efectos ARCH de orden K (varianza constante)
#   Hâ‚: existen efectos ARCH -> heterocedasticidad condicional
# Se prefiere sobre Breusch-Pagan y White porque estos asumen una forma
# funcional de la heterocedasticidad respecto a las variables exogenas,
# mientras que el test ARCH captura la estructura dinamica propia de las
# series temporales financieras (dependencia de la varianza en su propio pasado).
#
# GRAFICO ELEGIDO: Retornos diarios + Volatilidad Rodante (eje doble)
# Este grafico es el mas intuitivo y directo para evidenciar heterocedasticidad:
# visualmente se aprecian los periodos de calma (dispersion baja en las barras,
# volatilidad baja en la curva naranja) y de crisis (barras que sobrepasan Â±3%,
# picos en la curva). La correlacion visual entre ambas series es la "firma"
# del fenomeno ARCH. El p-valor del test ARCH queda anotado en el titulo de
# cada panel para facilitar la interpretacion estadistica.

print(f"\n  [12.3] Test ARCH de Engle y grafico de volatilidad rodante...")

_resultados_arch = {}
for _tick in sorted(seleccionadas):
    _serie = (dataset_long[dataset_long['Ticker'] == _tick][COL_RET]
              .dropna().values)
    if len(_serie) < 50:
        continue
    # het_arch(x, nlags=10): test LM de Engle con 10 retardos de xÂ²
    # Devuelve (lm_stat, lm_pvalue, f_stat, f_pvalue)
    _lm_stat, _lm_pval, _, _ = het_arch(_serie, nlags=10)
    _resultados_arch[_tick] = (_lm_stat, _lm_pval)

print(f"\n  {'Ticker':<12} {'LM-stat':>10} {'p-valor':>10} {'ARCH?':>12}")
print("  " + "-"*48)
for _tick, (_stat, _pval) in _resultados_arch.items():
    # Triple asterisco = significativo al 1%, muy comun en retornos financieros
    if _pval < 0.01:
        _sig = 'SI (***)'
    elif _pval < 0.05:
        _sig = 'SI (*)'
    else:
        _sig = 'NO'
    print(f"  {_tick:<12} {_stat:>10.3f} {_pval:>10.4f} {_sig:>12}")

# GRAFICO 12.3 -- Grid de retornos + volatilidad rodante (eje doble por panel)
# Columnas = 2 para dar espacio suficiente a los dos ejes Y de cada panel.
_n_sel       = len(seleccionadas)
_ncols_arch  = 2
_nrows_arch  = (_n_sel + _ncols_arch - 1) // _ncols_arch
fig_arch, axes_arch = plt.subplots(_nrows_arch, _ncols_arch,
                                    figsize=(12, 4 * _nrows_arch))
_axes_arch_flat = axes_arch.flatten() if _n_sel > 1 else [axes_arch]
fig_arch.suptitle(
    'Heterocedasticidad: Retornos Diarios (barras) y Volatilidad Rodante Std_M (naranja)',
    fontsize=12, fontweight='bold'
)

for _i, _tick in enumerate(sorted(seleccionadas)):
    _ax = _axes_arch_flat[_i]
    _sub = (dataset_long[dataset_long['Ticker'] == _tick]
            [['Date', COL_RET, 'Std_M']]
            .dropna(subset=[COL_RET])
            .set_index('Date'))
    if _sub.empty:
        _ax.set_visible(False)
        continue

    # Eje izquierdo: retornos diarios como barras grises
    # El ancho=1 asegura que las barras sean continuas (sin huecos entre dias)
    _ax.bar(_sub.index, _sub[COL_RET], color='lightgray',
            width=1, alpha=0.7, label='Retorno diario')
    _ax.set_ylabel('Log Return diario', fontsize=7, color='dimgray')
    _ax.tick_params(axis='y', labelsize=6, labelcolor='dimgray')

    # Eje derecho: volatilidad rodante mensual como linea naranja
    # Std_M = desviacion tipica de los ultimos 22 dias = proxy de volatilidad
    # realizada mensual. Sus picos coincidiran visualmente con los periodos
    # de mayor dispersion en las barras grises -> evidencia visual de ARCH.
    _ax2 = _ax.twinx()
    if 'Std_M' in _sub.columns:
        _ax2.plot(_sub.index, _sub['Std_M'], color='darkorange',
                  linewidth=1.2, label='Volatilidad Std_M')
    _ax2.set_ylabel('Volatilidad Std_M', fontsize=7, color='darkorange')
    _ax2.tick_params(axis='y', labelsize=6, labelcolor='darkorange')
    
    # Anotacion p-valor ARCH en titulo
    if _tick in _resultados_arch:
        _pval_arch = _resultados_arch[_tick][1]
        _arch_sig = '***' if _pval_arch < 0.01 else ('*' if _pval_arch < 0.05 else 'ns')
        _ax.set_title(f'{_tick} (ARCH p={_pval_arch:.4f} {_arch_sig})', fontsize=8)
    else:
        _ax.set_title(f'{_tick}', fontsize=8)
    
    _ax.grid(alpha=0.3)

plt.tight_layout()
ruta_arch = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '12_3_heterocedasticidad_arch.png')
plt.savefig(ruta_arch, dpi=150, bbox_inches='tight')
plt.close()
print(f"    [OK] Guardado: {ruta_arch}")


# =============================================================================
# 12.4  MATRIZ DE CORRELACIONES -- Impacto conjunto en cartera
# =============================================================================
#
# Â¿POR QUE CORRELACIONES?
# Markowitz (1952) fundamenta la diversificacion en correlaciones bajas
# entre activos. Si todos los activos se mueven juntos (Ïâ‰ˆ1), la cartera
# no diversifica riesgo -> es como comprar un unico "super-activo".
# Las correlaciones tambien definen la eficiencia de la frontera:
# activos con Ï<0.7 permiten construccion de carteras robustas.
#
# Â¿METODO?
# Matriz de correlacion de Pearson sobre retornos winzorizados (Log_Return_Wins).
# Se winsoriza porque outliers extremos sesgan correlaciones (una caida del 20%
# en 1 dia puede crear correlaciones espurias con otros activos que se movieron
# menos). Winzorizacion preserva estructura real.

print(f"\n  [12.4] Matriz de correlaciones de Pearson (Log_Return_Wins)...")

# Construir matriz de correlaciones para empresas seleccionadas
retornos_wins = (
    dataset_long[dataset_long['Ticker'].isin(seleccionadas)]
    .pivot(index='Date', columns='Ticker', values='Log_Return_Wins')
    .dropna()
)

corr_pearson = retornos_wins.corr(method='pearson')

print(f"\n  Estadisticas de correlacion entre activos:")
print(f"    Correlacion media: {corr_pearson.values[np.triu_indices_from(corr_pearson.values, k=1)].mean():.3f}")
print(f"    Correlacion min  : {corr_pearson.values[np.triu_indices_from(corr_pearson.values, k=1)].min():.3f}")
print(f"    Correlacion max  : {corr_pearson.values[np.triu_indices_from(corr_pearson.values, k=1)].max():.3f}")
print(f"    % correlaciones >0.7: {(corr_pearson.values[np.triu_indices_from(corr_pearson.values, k=1)] > 0.7).sum() / len(corr_pearson.values[np.triu_indices_from(corr_pearson.values, k=1)]) * 100:.1f}%")

# GRAFICO 12.4 -- Heatmap de correlaciones con dendrograma
from scipy.cluster.hierarchy import dendrogram as _dendro
from scipy.spatial.distance import pdist as _pdist, squareform as _squareform

# Calculamos distancias para dendrograma
_distancia_corr = np.sqrt(2 * (1 - corr_pearson.values))
_linkage_corr = linkage(_squareform(_distancia_corr), method='ward')

# Figura 12.4-a: Heatmap de correlaciones de Pearson
fig_corr_a, ax_corr_a = plt.subplots(figsize=(10, 9))
im = ax_corr_a.imshow(corr_pearson.values, cmap='RdBu_r', aspect='auto',
                       vmin=-1, vmax=1, interpolation='nearest')
ax_corr_a.set_xticks(range(len(corr_pearson.columns)))
ax_corr_a.set_yticks(range(len(corr_pearson.columns)))
ax_corr_a.set_xticklabels(corr_pearson.columns, fontsize=8, rotation=45, ha='right')
ax_corr_a.set_yticklabels(corr_pearson.columns, fontsize=8)
ax_corr_a.set_title(
    'Matriz de Correlacion de Pearson — Activos Seleccionados DAX40\n'
    '(rojo intenso = correlacion positiva alta, azul = correlacion negativa)',
    fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax_corr_a, label='Correlacion', fraction=0.046, pad=0.04)
plt.tight_layout()
ruta_corr_a = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '12_4a_matriz_correlaciones_pearson.png')
plt.savefig(ruta_corr_a, dpi=150, bbox_inches='tight')
plt.close()
print(f"    [OK] Guardado: {ruta_corr_a}")

# Figura 12.4-b: Dendrograma jerarquico basado en correlaciones
fig_corr_b, ax_corr_b = plt.subplots(figsize=(10, 7))
_dendro(
    _linkage_corr,
    labels=corr_pearson.columns,
    ax=ax_corr_b,
    orientation='right',
    leaf_font_size=9,
)
ax_corr_b.set_title(
    'Clustering Jerarquico de Activos por Similitud de Retornos\n'
    '(distancia = sqrt(2*(1 - correlacion_Pearson)), metodo Ward)',
    fontsize=12, fontweight='bold')
ax_corr_b.set_xlabel('Distancia (1 - Correlacion)')
plt.tight_layout()
ruta_corr_b = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '12_4b_dendrograma_correlaciones.png')
plt.savefig(ruta_corr_b, dpi=150, bbox_inches='tight')
plt.close()
print(f"    [OK] Guardado: {ruta_corr_b}")


# =============================================================================
# 12.5  ANALISIS DE RETORNOS POR SECTOR
# =============================================================================
#
# POR QUE SECTORES?
# Los ciclos economicos afectan sectores de forma diferenciada:
# - Ciclos expansivos: Consumo, Industriales, Tecnologia (leverage positivo)
# - Estancamiento: Utilidades, Finanzas (rentas defensivas)
# - Recesion: Salud, Consumo defensivo (resistente)
# - Inflacion: Energia, Materiales (beneficiados por precios commodities)
#
# Validar que tu cartera captura esta diversificacion sectorial = cartera
# robusta a cambios de regimen macroeconomico.

print(f"\n  [12.5] Analisis de retornos por sector...")

# Mapear tickers a sectores (dinamico, usa solo las empresas seleccionadas)
SECTORES_FINAL = construir_sectores(seleccionadas)

# Calcular retorno promedio acumulado por sector
retornos_sector = {}
_tickers_sect_presentes = []
for sector, tickers in SECTORES_FINAL.items():
    tickers_en_cartera = [t for t in tickers if t in seleccionadas]
    if tickers_en_cartera:
        _tickers_sect_presentes.append(sector)
        subset = dataset_long[dataset_long['Ticker'].isin(tickers_en_cartera)][['Date', 'Ticker', 'Log_Return_D']]
        ret_sector = subset.groupby('Date')['Log_Return_D'].mean()  # promedio por fecha
        retornos_sector[sector] = ret_sector

print(f"\n  {'Sector':<15} {'# Empresas':>12} {'Ret. Acum. (%)':>15} {'Volatilidad':>12}")
print("  " + "-"*56)
for sector in sorted(retornos_sector.keys()):
    n_emp = len([t for t in SECTORES_FINAL[sector] if t in seleccionadas])
    ret_acum = retornos_sector[sector].sum() * 100
    vol = retornos_sector[sector].std() * 100
    print(f"  {sector:<15} {n_emp:>12} {ret_acum:>14.2f}% {vol:>11.2f}%")

# GRAFICO 12.5 -- Evolucion de retornos acumulados por sector
fig_sect_ret, ax_sect_ret = plt.subplots(figsize=(16, 8))
for sector, ret_serie in sorted(retornos_sector.items()):
    ret_acum = (1 + ret_serie).cumprod() - 1
    ax_sect_ret.plot(ret_acum.index, ret_acum.values * 100, label=sector, linewidth=1.2, alpha=0.8)

ax_sect_ret.set_title('Evolucion de Retornos Acumulados por Sector\n(cartera ponderada igual)',
                      fontsize=12, fontweight='bold')
ax_sect_ret.set_xlabel('Fecha', fontsize=10)
ax_sect_ret.set_ylabel('Retorno Acumulado (%)', fontsize=10)
ax_sect_ret.legend(fontsize=9, loc='best')
ax_sect_ret.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
ruta_sect_ret = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '12_5_retornos_por_sector.png')
plt.savefig(ruta_sect_ret, dpi=150, bbox_inches='tight')
plt.close()
print(f"    [OK] Guardado: {ruta_sect_ret}")

# MODIFICACIÓN NUEVA: Beta sectorial vs DAX40 + Rotacion sectorial
# -----------------------------------------------------------------------
# Beta = sensibilidad del sector al mercado (OLS: r_sector = alpha + beta*r_DAX)
#   beta > 1: sector agresivo (amplifica movimientos del mercado)
#   beta < 1: sector defensivo (amortigua movimientos)
#   beta ~ 1: se mueve con el mercado
# Rotacion sectorial = ranking relativo de cada sector en ventanas rodantes
#   de 12 meses. Permite ver que sectores lideran y cuales quedan rezagados
#   en cada fase del ciclo economico.
# -----------------------------------------------------------------------

# --- 12.5b BETA SECTORIAL VS DAX40 ---
print(f"\n  [12.5b] Beta sectorial vs DAX40...")

# Calcular retorno diario del DAX40 desde DAX40_Close
_dax_close = dataset_long.groupby('Date')['DAX40_Close'].first().dropna()
_dax_ret = np.log(_dax_close / _dax_close.shift(1)).dropna()

_betas_sect = {}
_alphas_sect = {}
print(f"\n  {'Sector':<15} {'Beta':>8} {'Alpha (anual)':>15} {'R2':>8} {'Tipo':>12}")
print("  " + "-"*60)
for sector, ret_serie in sorted(retornos_sector.items()):
    # Alinear fechas entre sector y DAX
    _df_beta = pd.DataFrame({
        'sector': ret_serie,
        'dax': _dax_ret
    }).dropna()
    if len(_df_beta) < 60:
        continue
    _slope, _intercept, _rvalue, _pvalue, _stderr = stats.linregress(
        _df_beta['dax'].values, _df_beta['sector'].values
    )
    _betas_sect[sector] = _slope
    _alphas_sect[sector] = _intercept
    _alpha_anual = _intercept * 252 * 100  # anualizado en %
    _r2 = _rvalue ** 2
    _tipo = 'Agresivo' if _slope > 1.05 else ('Defensivo' if _slope < 0.95 else 'Neutro')
    print(f"  {sector:<15} {_slope:>8.3f} {_alpha_anual:>14.2f}% {_r2:>8.3f} {_tipo:>12}")

# GRAFICO 12.5b -- Betas sectoriales (barras horizontales)
if _betas_sect:
    _sect_sorted = sorted(_betas_sect.items(), key=lambda x: x[1], reverse=True)
    _sect_names = [s[0] for s in _sect_sorted]
    _sect_betas = [s[1] for s in _sect_sorted]
    _colors_beta = ['#ef5350' if b > 1.05 else '#66bb6a' if b < 0.95 else '#42a5f5'
                    for b in _sect_betas]

    fig_beta, ax_beta = plt.subplots(figsize=(10, max(5, len(_sect_names) * 0.6)))
    ax_beta.barh(_sect_names, _sect_betas, color=_colors_beta, edgecolor='black', linewidth=0.3)
    ax_beta.axvline(1.0, color='black', linestyle='--', linewidth=1, label='Beta = 1 (mercado)')
    ax_beta.set_xlabel('Beta vs DAX40', fontsize=11)
    ax_beta.set_title('Beta Sectorial vs DAX40\n'
                      'Rojo = Agresivo (>1.05) | Verde = Defensivo (<0.95) | Azul = Neutro',
                      fontsize=12, fontweight='bold')
    ax_beta.legend(fontsize=9)
    ax_beta.grid(alpha=0.3, axis='x')
    for i, (name, beta) in enumerate(_sect_sorted):
        ax_beta.text(beta + 0.02, i, f'{beta:.2f}', va='center', fontsize=9)
    plt.tight_layout()
    _ruta_beta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '12_5b_beta_sectorial.png')
    plt.savefig(_ruta_beta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [OK] Guardado: {_ruta_beta}")

# --- 12.5c ROTACION SECTORIAL (ranking rodante 12 meses) ---
print(f"\n  [12.5c] Rotacion sectorial (ranking rodante 252d)...")

_VENTANA_ROT = 252  # 12 meses

# Construir DataFrame de retornos rodantes por sector
_ret_sector_df = pd.DataFrame(retornos_sector)
_ret_sector_df.index = pd.to_datetime(_ret_sector_df.index)
_ret_sector_df = _ret_sector_df.sort_index().dropna(how='all')

# Retorno acumulado rodante (252 dias)
_ret_rolling = _ret_sector_df.rolling(_VENTANA_ROT).sum() * 100  # en %
_ret_rolling = _ret_rolling.dropna(how='all')

# Ranking: 1 = mejor sector, N = peor. ascending=False para que mayor retorno = rank 1
_rank_rolling = _ret_rolling.rank(axis=1, ascending=False, method='min')

# Imprimir ultimo ranking
if not _rank_rolling.empty:
    _last_date = _rank_rolling.index[-1]
    _last_ranks = _rank_rolling.loc[_last_date].sort_values()
    print(f"\n  Ranking sectorial mas reciente ({_last_date.strftime('%Y-%m-%d')}):")
    print(f"  {'Ranking':>8} {'Sector':<15} {'Ret 12m (%)':>12}")
    print("  " + "-"*38)
    for sect in _last_ranks.index:
        _r = _last_ranks[sect]
        _ret_val = _ret_rolling.loc[_last_date, sect]
        print(f"  {int(_r):>8} {sect:<15} {_ret_val:>11.2f}%")

# GRAFICO 12.5c -- Heatmap de rotacion sectorial
if not _rank_rolling.empty and len(_rank_rolling) > 50:
    # Resamplear a mensual para legibilidad del heatmap
    _rank_monthly = _rank_rolling.resample('ME').last().dropna(how='all')

    if len(_rank_monthly) > 6:
        fig_rot, ax_rot = plt.subplots(figsize=(max(14, len(_rank_monthly.columns) * 1.5),
                                                max(8, len(_rank_monthly) * 0.12)))
        # Transponer: sectores en filas, fechas en columnas
        _hm_data = _rank_monthly.T
        # Formatear fechas para etiquetas
        _xlabels = [d.strftime('%Y-%m') for d in _hm_data.columns]

        # Mostrar solo cada N etiquetas para legibilidad
        _step_lbl = max(1, len(_xlabels) // 24)
        _xlabels_show = [l if i % _step_lbl == 0 else '' for i, l in enumerate(_xlabels)]

        sns.heatmap(_hm_data, annot=False, cmap='RdYlGn_r',
                    xticklabels=_xlabels_show, yticklabels=True,
                    linewidths=0.1, linecolor='gray',
                    cbar_kws={'label': 'Ranking (1=mejor, N=peor)'},
                    ax=ax_rot)
        ax_rot.set_title('Rotacion Sectorial: Ranking Rodante 12 Meses\n'
                         'Verde oscuro = sector lider, Rojo = sector rezagado',
                         fontsize=12, fontweight='bold')
        ax_rot.set_xlabel('Fecha', fontsize=10)
        ax_rot.set_ylabel('Sector', fontsize=10)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        _ruta_rot = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '12_5c_rotacion_sectorial.png')
        plt.savefig(_ruta_rot, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    [OK] Guardado: {_ruta_rot}")

    # Resumen de rotacion
    _n_cambios_lider = 0
    _prev_lider = None
    for _d in _rank_monthly.index:
        _lider = _rank_monthly.loc[_d].idxmin()
        if _lider != _prev_lider and _prev_lider is not None:
            _n_cambios_lider += 1
        _prev_lider = _lider
    print(f"\n  Cambios de sector lider en el periodo: {_n_cambios_lider}")
    print(f"  {'Hay rotacion sectorial activa.' if _n_cambios_lider > 3 else 'Rotacion sectorial limitada (pocos cambios de lider).'}")
    print(f"  Esto {'valida' if _n_cambios_lider > 3 else 'limita'} la importancia de diversificacion sectorial en la cartera.")


