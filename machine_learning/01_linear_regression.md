# <img src="https://img.icons8.com/?size=50&id=80592&format=png&color=000000" align="center"/> Linear Regression
## Regresión Lineal → Predecir un número
> La regresión lineal es el algoritmo que encuentra la línea recta que mejor describe la relación entre una variable y otra, y la usa para predecir valores nuevos.
> Es el algoritmo de Machine Learning más simple y el más importante de entender porque todos los demás se construyen sobre él.

> **Regla de oro:**
> Si necesitas predecir un número continuo (precio, venta, temperatura, ingreso) y sospechas que hay una relación lineal,
> empieza siempre con regresión lineal. Si no funciona bien, entonces prueba algo más complejo.

---

## <img src="https://img.icons8.com/?size=40&id=81170&format=png&color=000000" align="center"/> La intuición: qué hace la regresión lineal

```
Ventas ($)
    │                              *
    │                         *  *
    │                    *  *
    │               * *
    │          * *
    │     * *
    │* *
    └──────────────────────────────→ Unidades vendidas

Regresión lineal encuentra la línea que minimiza la suma de los errores al cuadrado:

    │          */
    │         */
    │        */ ← línea de regresión
    │       */
    │      */
    │     */
    │    */
    └──────────────────────────────→

Esa línea te permite predecir: "Si vendo 50 unidades, ¿cuánto ingreso espero?"
```

**La ecuación de la línea:**

```
ŷ = β₀ + β₁×X

ŷ = valor predicho (lo que queremos saber)
β₀ = intercepto (valor de ŷ cuando X = 0)
β₁ = pendiente (cuánto cambia ŷ por cada unidad de X)
X = variable predictora (lo que ya sabemos)
```

**Con múltiples variables (regresión múltiple):**

```
ŷ = β₀ + β₁×X₁ + β₂×X₂ + ... + βₙ×Xₙ

Cada β representa el efecto de su variable manteniendo las demás constantes.
```

---

## <img src="https://img.icons8.com/?size=40&id=QaHIbDj74XXB&format=png&color=000000" align="center"/> Los dos tipos

```
REGRESIÓN LINEAL SIMPLE          REGRESIÓN LINEAL MÚLTIPLE
─────────────────────            ──────────────────────────
Una variable predictora          Varias variables predictoras

ŷ = β₀ + β₁×X                    ŷ = β₀ + β₁×X₁ + β₂×X₂ + β₃×X₃

Ejemplo:                         Ejemplo:
Predecir ventas a partir         Predecir ventas a partir
de unidades vendidas             de precio + unidades + región + mes
```

---

## <img src="https://img.icons8.com/?size=40&id=121464&format=png&color=000000" align="center"/> Implementación completa en Python

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── PASO 0: PREPARAR LOS DATOS ──

# Seleccionar features y target
X = df[['unidades_vendidas', 'precio_unitario']].copy()
y = df['ventas_total'].copy()

# Manejar nulos
X = X.dropna()
y = y[X.index]

print(f"Dataset: {X.shape[0]:,} filas, {X.shape[1]} features")
print(f"Target: {y.name}")
print(f"\nDistribución del target:")
print(f"  Media:   ${y.mean():,.2f}")
print(f"  Mediana: ${y.median():,.2f}")
print(f"  Std:     ${y.std():,.2f}")
print(f"  Min:     ${y.min():,.2f}")
print(f"  Max:     ${y.max():,.2f}")

# ── PASO 1: ANÁLISIS PREVIO → verificar linealidad ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, col in zip(axes, X.columns):
    ax.scatter(X[col], y, alpha=0.3, s=10, color='steelblue')
    # Línea de tendencia
    z = np.polyfit(X[col], y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(X[col].min(), X[col].max(), 100)
    ax.plot(x_line, p(x_line), color='red', linewidth=2, label='Tendencia')
    r, _ = stats.pearsonr(X[col], y)
    ax.set_title(f'{col} vs {y.name}\nr = {r:.3f}')
    ax.set_xlabel(col)
    ax.set_ylabel(y.name)
    ax.legend()

plt.tight_layout()
plt.savefig('images/ml_regresion_scatter_previo.png',
            dpi=150)
plt.show()

# ── PASO 2: DIVIDIR EN ENTRENAMIENTO Y PRUEBA ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,      # 20% para prueba
    random_state=42)     # semilla para reproducibilidad

print(f"\nTrain: {X_train.shape[0]:,} filas")
print(f"Test:  {X_test.shape[0]:,} filas")
print(f"Ratio: {X_train.shape[0]/X_test.shape[0]:.1f}:1")

# ── PASO 3: ESCALAR (opcional pero recomendado) ──
# StandardScaler: media=0, std=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
# CRÍTICO: fit_transform en train, solo transform en test
# Si fiteas en test → data leakage (trampa)

# ── PASO 4: ENTRENAR EL MODELO ──
modelo = LinearRegression()
modelo.fit(X_train_scaled, y_train)

print(f"\n{'='*50}")
print("PARÁMETROS DEL MODELO")
print(f"{'='*50}")
print(f"Intercepto (β₀): {modelo.intercept_:,.2f}")
for feature, coef in zip(X.columns, modelo.coef_):
    print(f"β ({feature}): {coef:,.4f}")

# Ecuación completa
print(f"\nEcuación:")
print(f"ŷ = {modelo.intercept_:.2f}", end="")
for feature, coef in zip(X.columns, modelo.coef_):
    sign = "+" if coef >= 0 else ""
    print(f" {sign}{coef:.4f}×{feature}", end="")
print()

# ── PASO 5: PREDECIR ──
y_pred_train = modelo.predict(X_train_scaled)
y_pred_test = modelo.predict(X_test_scaled)

# ── PASO 6: EVALUAR EL MODELO ──
def evaluar_modelo(y_real, y_pred, nombre=''):
    r2  = r2_score(y_real, y_pred)
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100

    print(f"\n{'='*50}")
    print(f"MÉTRICAS — {nombre}")
    print(f"{'='*50}")
    print(f"R²: {r2:.4f} → el modelo explica el {r2*100:.1f}% de la variación")
    print(f"MAE:  ${mae:,.2f} → error promedio absoluto")
    print(f"RMSE: ${rmse:,.2f} → error cuadrático medio")
    print(f"MAPE: {mape:.1f}% → error porcentual promedio")

    return {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

metricas_train = evaluar_modelo(y_train, y_pred_train, 'ENTRENAMIENTO')
metricas_test = evaluar_modelo(y_test, y_pred_test, 'PRUEBA')

# Detectar sobreajuste
diff_r2 = metricas_train['R2'] - metricas_test['R2']
print(f"\nDiferencia R² (train - test): {diff_r2:.4f}")
if diff_r2 > 0.1:
    print("✇ Posible sobreajuste → el modelo aprendió el ruido del train")
elif diff_r2 < -0.05:
    print("✇ Modelo extraño → test mejor que train")
else:
    print("✓ Sin sobreajuste evidente")
```

---

## <img src="https://img.icons8.com/?size=35&id=7MuYE737H6Vg&format=png&color=000000" align="center"/> Las métricas de evaluación: qué significan

### R²: Coeficiente de Determinación

```python
# R² = r² para regresión simple

r2 = r2_score(y_test, y_pred_test)
print(f"R² = {r2:.4f}")
print(f"→ El modelo explica el {r2*100:.1f}% de la variación")
print(f"→ El {(1-r2)*100:.1f}% restante viene de factores no incluidos en el modelo")

# Escala de interpretación (depende del contexto):
# > 0.90 → Excelente (datos muy predecibles)
# > 0.70 → Bueno
# > 0.50 → Moderado
# > 0.30 → Débil (puede ser útil en ciencias sociales)
# < 0.30 → Muy débil (considera otras variables)
```

### MAE: Error Absoluto Medio

```python
# Mean Absolute Error
# Promedio de |y_real - y_pred|
# En las mismas unidades que el target

mae = mean_absolute_error(y_test, y_pred_test)
print(f"MAE = ${mae:,.2f}")
print(f"→ En promedio, el modelo se equivoca ±${mae:,.2f}")
print(f"→ Fácil de interpretar")
print(f"→ No penaliza errores grandes")

# Interpretar como % de la media
pct_mae = mae / y_test.mean() * 100
print(f"→ = {pct_mae:.1f}% de la venta promedio")
```

### RMSE: Raíz del Error Cuadrático Medio

```python
# Root Mean Square Error
# Raíz de: promedio de (y_real - y_pred)²
# Penaliza errores grandes más que el MAE

rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"RMSE = ${rmse:,.2f}")
print(f"→ Similar al MAE pero penaliza errores grandes")
print(f"→ Si RMSE >> MAE: hay predicciones muy malas")
print(f"→ Más sensible a outliers que el MAE")

# Comparar MAE vs RMSE
print(f"\nMAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"Ratio RMSE/MAE: {rmse/mae:.2f}")
if rmse/mae > 1.5:
    print("✇ Hay predicciones muy erróneas → investigar")
else:
    print("✓ Errores relativamente uniformes")
```

### MAPE: Error Porcentual Absoluto Medio

```python
# Mean Absolute Percentage Error
# Promedio de |y_real - y_pred| / y_real × 100
# En porcentaje → fácil de comunicar a no-técnicos

mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
print(f"MAPE = {mape:.1f}%")
print(f"→ El modelo se equivoca en promedio ±{mape:.1f}%")
print(f"→ Ideal para comunicar a negocio")
print(f"→ ✇ No usar si hay valores cercanos a 0")

# Escala de interpretación:
# < 10%  → Excelente
# 10-20% → Bueno
# 20-50% → Aceptable
# > 50%  → Modelo poco útil
```

---

## <img src="https://img.icons8.com/?size=40&id=80443&format=png&color=000000" align="center"/> Visualizaciones del modelo

```python
# ── FIGURA COMPLETA DE DIAGNÓSTICO ──
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

residuos = y_test - y_pred_test

# 1. Predicciones vs valores reales
axes[0, 0].scatter(y_test, y_pred_test, alpha=0.3, s=10, color='steelblue')
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
axes[0, 0].plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2, linestyle='--', label='Predicción perfecta')
axes[0, 0].set_title(f'Predicciones vs Reales\nR²={r2:.3f}')
axes[0, 0].set_xlabel('Valores Reales')
axes[0, 0].set_ylabel('Predicciones')
axes[0, 0].legend()
# Si los puntos siguen la línea diagonal → buen modelo

# 2. Residuos vs predicciones (homocedasticidad)
axes[0, 1].scatter(y_pred_test, residuos, alpha=0.3, s=10, color='teal')
axes[0, 1].axhline(0, color='red', linewidth=2, linestyle='--')
axes[0, 1].set_title('Residuos vs Predicciones\n✓ Nube aleatoria alrededor de 0')
axes[0, 1].set_xlabel('Predicciones')
axes[0, 1].set_ylabel('Residuos')
# Si hay patrón (embudo, curva) → problema

# 3. Q-Q plot de residuos (normalidad)
stats.probplot(residuos, dist='norm', plot=axes[0, 2])
axes[0, 2].get_lines()[1].set_color('red')
axes[0, 2].set_title('Q-Q Plot de Residuos\n✓ Puntos sobre la línea diagonal')

# 4. Distribución de residuos
axes[1, 0].hist(residuos, bins=50, color='steelblue', edgecolor='white', alpha=0.8, density=True)
x_r = np.linspace(residuos.min(), residuos.max(), 200)
axes[1, 0].plot(x_r, stats.norm.pdf(x_r, residuos.mean(), residuos.std()),
                color='red', linewidth=2, label='Normal teórica')
axes[1, 0].set_title('Histograma de Residuos\n✓ Debería ser una campana')
axes[1, 0].legend()

# 5. Importancia de features (coeficientes)
coefs = pd.Series(modelo.coef_, index=X.columns)
coefs = coefs.sort_values()
colores_coef = ['#E63946' if c < 0 else '#1A7F7A' for c in coefs.values]
axes[1, 1].barh(coefs.index, coefs.values, color=colores_coef, edgecolor='white')
axes[1, 1].axvline(0, color='black', linewidth=1)
axes[1, 1].set_title('Coeficientes del Modelo\n' 'Verde=positivo, Rojo=negativo')
axes[1, 1].set_xlabel('Valor del coeficiente')

# 6. Error de predicción por rangos
y_test_reset = y_test.reset_index(drop=True)
y_pred_reset = pd.Series(y_pred_test)
residuos_abs = np.abs(y_test_reset - y_pred_reset)

rangos = pd.cut(y_test_reset, bins=5, labels=['Muy bajo', 'Bajo', 'Medio', 'Alto', 'Muy alto'])
error_por_rango = pd.DataFrame({'rango': rangos, 'error_abs': residuos_abs}).groupby('rango')['error_abs'].mean()

axes[1, 2].bar(range(len(error_por_rango)), error_por_rango.values, color='coral', edgecolor='white')
axes[1, 2].set_xticks(range(len(error_por_rango)))
axes[1, 2].set_xticklabels(error_por_rango.index, rotation=30)
axes[1, 2].set_title('Error Promedio por Rango de Venta\n¿Dónde falla más el modelo?')
axes[1, 2].set_ylabel('MAE promedio')

plt.suptitle('Diagnóstico Completo — Regresión Lineal', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/ml_regresion_diagnostico.png', dpi=150)
plt.show()
```

---

## <img src="https://img.icons8.com/?size=40&id=hKNUXmKDeI2V&format=png&color=000000" align="center"/> Interpretar los coeficientes

```python
# ── REGRESIÓN SIMPLE: interpretación directa ──
modelo_simple = LinearRegression()
X_simple = df[['unidades_vendidas']]
modelo_simple.fit(X_simple, y)

b0 = modelo_simple.intercept_
b1 = modelo_simple.coef_[0]

print(f"ŷ = {b0:.2f} + {b1:.4f} × unidades_vendidas")
print(f"\nInterpretación:")
print(f"β₀ = {b0:.2f}: venta esperada con 0 unidades")
print(f"β₁ = {b1:.4f}: por cada unidad adicional la venta aumenta ${b1:.4f}")

# Predicción con el modelo
unidades_nuevas = np.array([[10], [50], [100]])
predicciones = modelo_simple.predict(unidades_nuevas)

print(f"\nPredicciones:")
for u, p in zip(unidades_nuevas.flatten(), predicciones):
    print(f"  {u} unidades → ${p:,.2f} venta esperada")

# ── REGRESIÓN MÚLTIPLE: interpretación ceteris paribus "Todo lo demás constante" ──
modelo_mult = LinearRegression()
X_mult = df[['unidades_vendidas', 'precio_unitario', 'tamaño_producto']]
modelo_mult.fit(StandardScaler().fit_transform(X_mult), y)

print(f"\nCoeficientes estandarizados:")
for feat, coef in zip(X_mult.columns, modelo_mult.coef_):
    print(f"  {feat}: {coef:.4f}")
    print(f"  → Un aumento de 1σ en {feat} cambia la venta en ${coef:.2f}, manteniendo las otras variables constantes")
```

---

## <img src="https://img.icons8.com/?size=40&id=80923&format=png&color=000000" align="center"/> Los 4 supuestos: y qué pasa si se violan

```python
def verificar_supuestos(y_real, y_pred, X):
    """
    Verifica los 4 supuestos de regresión lineal.
    """
    residuos = y_real - y_pred

    print("="*60)
    print("VERIFICACIÓN DE SUPUESTOS")
    print("="*60)

    # SUPUESTO 1: Linealidad
    # → Los residuos no deben tener patrón sistemático
    corr_residuos = pd.Series(residuos).corr(pd.Series(y_pred))
    print(f"\n1. LINEALIDAD")
    print(f"   Correlación residuos-predicciones: "
          f"{corr_residuos:.4f}")
    print(f"   {'✓ OK' if abs(corr_residuos) < 0.1 else '☓ Patrón detectado → posible no linealidad'}")

    # SUPUESTO 2: Normalidad de residuos
    _, p_normal = stats.normaltest(residuos)
    print(f"\n2. NORMALIDAD DE RESIDUOS")
    print(f"   p-value (D'Agostino): {p_normal:.4f}")
    print(f"   {'✓ Residuos normales' if p_normal > 0.05 else '✇ Residuos no normales'} (con n grande, el TLC mitiga este problema)")

    # SUPUESTO 3: Homocedasticidad
    # Varianza constante de residuos
    # Test de Breusch-Pagan (simplificado)
    residuos_abs = np.abs(residuos)
    corr_hetero = pd.Series(residuos_abs).corr(pd.Series(y_pred))
    print(f"\n3. HOMOCEDASTICIDAD")
    print(f"   Correlación |residuos|-predicciones: {corr_hetero:.4f}")
    print(f"   {'✓ Varianza aproximadamente constante' if abs(corr_hetero) < 0.2 else '✇ Heterocedasticidad detectada'}")

    # SUPUESTO 4: Independencia (Durbin-Watson)
    # Solo relevante si los datos tienen orden temporal
    dw = np.sum(np.diff(residuos)**2) / np.sum(residuos**2)
    print(f"\n4. INDEPENDENCIA (Durbin-Watson)")
    print(f"   DW = {dw:.4f}")
    if 1.5 < dw < 2.5:
        print(f"   ✓ Sin autocorrelación significativa")
    elif dw < 1.5:
        print(f"   ✇ Autocorrelación positiva")
    else:
        print(f"   ✇ Autocorrelación negativa")

    return residuos

residuos = verificar_supuestos(y_test, y_pred_test, X_test)
```

### Qué hacer cuando se violan los supuestos

```
SUPUESTO VIOLADO         SÍNTOMA              SOLUCIÓN
────────────────────────────────────────────────────────────────────────────
Linealidad               Curva en residuos    Transformar X (log, sqrt)
                         vs predicciones      o agregar términos cuadráticos

Normalidad de residuos   Q-Q plot desviado    Transformar y (log1p)
                         del histograma       o usar más datos (TLC)
                         no es campana

Homocedasticidad         Embudo en            Transformar y (log1p)
(varianza no constante)  residuos vs          Regresión ponderada
                         predicciones         

Independencia            DW < 1.5 o > 2.5     Modelos de series de
(autocorrelación)        Datos ordenados      tiempo (ARIMA)
                         en el tiempo
```

---

## <img src="https://img.icons8.com/?size=40&id=81350&format=png&color=000000" align="center"/> Regresión con datos transformados

```python
# Para datos log-normales como ventas retail la transformación log mejora mucho el modelo

# ── COMPARAR: con y sin transformación ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Modelo 1: sin transformación
X_tr, X_te, y_tr, y_te = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

m1 = LinearRegression().fit(X_tr, y_tr)
r2_sin = r2_score(y_te, m1.predict(X_te))

# Modelo 2: con log-transformación del target
y_log_train = np.log1p(y_train)
y_log_test = np.log1p(y_test)

m2 = LinearRegression().fit(X_train_scaled, y_log_train)
y_pred_log = m2.predict(X_test_scaled)

# Deshacer la transformación para comparar
y_pred_original = np.expm1(y_pred_log)
r2_con = r2_score(y_test, y_pred_original)

print(f"R² sin transformación: {r2_sin:.4f}")
print(f"R² con log(y): {r2_con:.4f}")
print(f"Mejora: {(r2_con-r2_sin)*100:.1f} puntos")

# ── PROCESO CORRECTO ──
# 1. Entrenar con log(y)
modelo_log = LinearRegression()
modelo_log.fit(X_train_scaled, np.log1p(y_train))

# 2. Predecir en escala log
pred_log = modelo_log.predict(X_test_scaled)

# 3. Convertir predicciones de vuelta a escala original
pred_original = np.expm1(pred_log)
# expm1(x) = e^x - 1 (inverso de log1p)

# 4. Evaluar en escala original
r2_final = r2_score(y_test, pred_original)
mae_final = mean_absolute_error(y_test, pred_original)
print(f"\nModelo final (con transformación):")
print(f"R²:  {r2_final:.4f}")
print(f"MAE: ${mae_final:,.2f}")
```

---

## <img src="https://img.icons8.com/?size=40&id=p3MA8DtJTjlg&format=png&color=000000" align="center"/> Pipeline completo de ML: buenas prácticas

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (train_test_split, cross_val_score)
import numpy as np

# ── PIPELINE — evita data leakage ──
# Un Pipeline aplica cada paso en orden
# y garantiza que el scaler no "vea" el test set

pipeline = Pipeline([('scaler', StandardScaler()), ('regresion', LinearRegression())])

# ── VALIDACIÓN CRUZADA: más confiable que un split ──
# Divide los datos en k grupos, entrena en k-1, prueba en el restante, repite k veces

scores = cross_val_score(
    pipeline, X, y,
    cv=5,                        # 5 folds
    scoring='r2')

print(f"R² por fold: {scores.round(3)}")
print(f"R² promedio: {scores.mean():.4f}")
print(f"Std:         {scores.std():.4f}")
print(f"IC 95%: [{scores.mean()-2*scores.std():.4f}, {scores.mean()+2*scores.std():.4f}]")

# Si std es grande → modelo inestable
# Si std es pequeña → modelo consistente

# ── ENTRENAR EL MODELO FINAL ──
# Con TODOS los datos (train + test) para producción
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

# Evaluación final en test (solo una vez)
y_pred = pipeline.predict(X_test)
print(f"\nEvaluación final en test:")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")
print(f"MAE:  ${mean_absolute_error(y_test, y_pred):,.2f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")

# ── PREDECIR NUEVOS DATOS ──
# Datos nuevos que nunca ha visto el modelo
nuevas_transacciones = pd.DataFrame({
    'unidades_vendidas': [5, 20, 100],
    'precio_unitario':   [15.0, 8.50, 3.20]
})

predicciones_nuevas = pipeline.predict(nuevas_transacciones)
print(f"\nPredicciones para nuevas transacciones:")
for i, (_, row) in enumerate(nuevas_transacciones.iterrows()):
    print(f"  {row['unidades_vendidas']:.0f} uds a ${row['precio_unitario']:.2f}c/u "
          f"→ venta predicha: ${predicciones_nuevas[i]:,.2f}")
```

---

## <img src="https://img.icons8.com/?size=40&id=81311&format=png&color=000000" align="center"/> Regresión Simple vs Múltiple

```python
# ── COMPARAR MODELOS CON DISTINTAS VARIABLES ──
from sklearn.model_selection import cross_val_score

configs = {
    'Simple (unidades)': ['unidades_vendidas'],
    'Simple (precio)':   ['precio_unitario'],
    'Doble':             ['unidades_vendidas', 'precio_unitario'],
    'Triple':            ['unidades_vendidas', 'precio_unitario', 'tamaño_producto'],
}

print("Comparación de modelos:")
print("-" * 50)

resultados = {}
for nombre, features in configs.items():
    X_config = df[features].dropna()
    y_config = y[X_config.index]

    pipeline = Pipeline([('scaler', StandardScaler()), ('modelo', LinearRegression())])

    scores = cross_val_score(pipeline, X_config, y_config, cv=5, scoring='r2')
    resultados[nombre] = scores.mean()
    print(f"{nombre:<30} R²={scores.mean():.4f} "
          f"(±{scores.std():.4f})")

# El modelo más simple con mejor R² es generalmente el mejor
mejor = max(resultados, key=resultados.get)
print(f"\n✓ Mejor modelo: {mejor}")
print(f"   Agregar más variables no siempre mejora el modelo")
```

---

## <img src="https://img.icons8.com/?size=40&id=80435&format=png&color=000000" align="center"/> Regresión Lineal vs otros algoritmos

```
CUÁNDO USAR REGRESIÓN LINEAL:
✓ Relación entre variables es aproximadamente lineal
✓ Necesitas interpretabilidad (explicas cada β)
✓ Dataset pequeño o mediano
✓ Primer modelo de referencia (baseline)
✓ Necesitas intervalos de confianza en predicciones
✓ El tiempo de entrenamiento importa

CUÁNDO USAR OTRO ALGORITMO:
✗ Relación es claramente no lineal → Árbol de decisión
✗ Muchas interacciones entre variables → Random Forest
✗ Datos de imagen o texto → Redes neuronales
✗ Muchas variables con multicolinealidad → Ridge/Lasso
✗ R² < 0.4 con datos limpios → revisar si es lineal

LA REGLA DEL BASELINE:
Siempre entrena regresión lineal primero. Si tu modelo complejo no supera a la regresión lineal en al menos 5-10% de R², no vale la complejidad adicional.
```

---

## <img src="https://img.icons8.com/?size=45&id=NGU2KSe3KO8r&format=png&color=000000" align="center"/> Analogía con Medicina Tradicional China

La regresión lineal en MTC sería como encontrar la relación de proporcionalidad entre dosis y respuesta en acupuntura:

| Regresión Lineal | Acupuntura MTC |
|---|---|
| Variable X | Número de sesiones de tratamiento |
| Variable Y | Intensidad del síntoma (0-10) |
| β₁ (pendiente) | Reducción de síntoma por sesión |
| β₀ (intercepto) | Intensidad inicial sin tratamiento |
| R² | Qué % de la mejoría explica el tratamiento |
| Residuos | Factores no controlados: dieta, estrés, sueño |
| Supuestos de linealidad | Asume respuesta proporcional → no siempre válido |
| Transformación log | Cuando la respuesta no es lineal sino exponencial |
| Modelo múltiple | Acupuntura + moxibustión + Qì Gōng → mejor R² |

> Si tratas 100 pacientes con el mismo protocolo y graficas sesiones vs mejoría, la regresión lineal encuentra la "tasa promedio de mejoría por sesión" → el β₁.
> El R² te dice qué tan predecible es la respuesta al tratamiento.
> Un R² bajo significa que hay muchos otros factores que determinan la mejoría.

---

## <img src="https://img.icons8.com/?size=40&id=9nyiB6VEUhJy&format=png&color=000000" align="center"/>  Cheat Sheet rápido

```
ECUACIÓN:
→ ŷ = β₀ + β₁X₁ + β₂X₂ + ...
→ β₀ = intercepto (valor base)
→ β₁ = pendiente (cambio por unidad de X)

FLUJO COMPLETO:
1. Scatter previo — ¿hay relación lineal?
2. Train/test split (80/20)
3. Escalar con StandardScaler (fit solo en train)
4. Entrenar: modelo.fit(X_train, y_train)
5. Predecir: modelo.predict(X_test)
6. Evaluar: R², MAE, RMSE, MAPE
7. Verificar supuestos (residuos)
8. Usar cross-validation para confirmar

MÉTRICAS:
→ R² = % de variación explicada (0 a 1)
→ MAE = error promedio en unidades del target
→ RMSE = MAE pero penaliza errores grandes
→ MAPE = error en % — fácil de comunicar

SUPUESTOS (verificar siempre):
Linealidad       → scatter residuos vs predicciones
Normalidad       → Q-Q plot de residuos
Homocedasticidad → sin embudo en residuos
Independencia    → Durbin-Watson ≈ 2

SEÑALES DE PROBLEMA:
R² train >> R² test → sobreajuste
RMSE >> MAE         → hay predicciones muy malas
Patrón en residuos  → relación no es lineal
Embudo en residuos  → heterocedasticidad

CON DATOS LOG-NORMALES (ventas, salarios):
Transformar: y_log = np.log1p(y)
Entrenar con y_log
Deshacer: y_pred = np.expm1(pred_log)
```

---
*Nota creada: 2026 · Serie: Machine Learning para Ciencia de Datos* 

*Parte del certificado Científico de Datos — EBAC* <img src="https://img.icons8.com/?size=25&id=42810&format=png&color=000000" align="center"/>

*<img src="https://img.icons8.com/?size=25&id=Gh99xffskT40&format=png&color=000000" align="center"/> github.com/ReginaPema* 
