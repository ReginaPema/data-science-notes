# Normal Distribution
## Distribución Normal — La campana que gobierna la estadística

> La distribución normal es la forma que toman los datos cuando muchos factores independientes y aleatorios se suman.
> Es la distribución más importante en estadística porque es la base de la mayoría de las pruebas,
> modelos y supuestos que se usan en Ciencia de Datos.

> **Por qué importa tanto:**
> La prueba Z, los intervalos de confianza, el p-value, la regresión lineal, y docenas de algoritmos de ML
> asumen que los datos siguen (o se aproximan a) una distribución normal.
> Si no la entiendes, no puedes interpretar ninguno de esos resultados correctamente.

---

## 🧱 Anatomía de la distribución normal

```
Frecuencia
    │
    │           ████
    │          ██████
    │         ████████
    │        ██████████
    │       ████████████
    │      ██████████████
    │     ████████████████
    │    ██████████████████
    │___████████████████████___
    └─────────────────────────→ Valor
  μ-3σ μ-2σ μ-σ  μ  μ+σ μ+2σ μ+3σ

    μ = media (centro de la campana)
    σ = desviación estándar (ancho de la campana)
```

| Elemento | Símbolo | Qué significa |
|---|---|---|
| Media | μ (mu) | El centro → donde está el pico |
| Desviación estándar | σ (sigma) | El ancho → qué tan dispersa es |
| Varianza | σ² | La dispersión al cuadrado |
| Campana simétrica | → | Media = Mediana = Moda |
| Cola izquierda | → | Valores bajos poco frecuentes |
| Cola derecha | → | Valores altos poco frecuentes |

**Propiedades fundamentales:**
- Perfectamente simétrica alrededor de la media
- Media = Mediana = Moda (siempre)
- Las colas se acercan a 0 pero nunca lo tocan
- El área total bajo la curva = 1 (100%)

---

## 🐍 Cómo trabajar con ella en Python

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# ── GENERAR DATOS NORMALES ──
np.random.seed(42)

# Normal estándar: media=0, std=1
datos_estandar = np.random.normal(loc=0, scale=1, size=10000)

# Normal con parámetros específicos
datos_custom = np.random.normal(loc=500,     # media = $500
                                 scale=100,  # std = $100
                                 size=10000)

# ── VERIFICAR SI TUS DATOS SON NORMALES ──
def verificar_normalidad(datos, nombre='Variable'):
    """
    Prueba múltiples métodos para verificar normalidad.
    Ningún método es definitivo solo,  úsalos juntos.
    """
    print(f"\n{'='*50}")
    print(f"PRUEBAS DE NORMALIDAD: {nombre}")
    print(f"{'='*50}")

    # 1. Estadísticas descriptivas
    media    = np.mean(datos)
    mediana  = np.median(datos)
    std      = np.std(datos)
    sesgo    = stats.skew(datos)
    curtosis = stats.kurtosis(datos)

    print(f"\nEstadísticas descriptivas:")
    print(f"  Media:    {media:.4f}")
    print(f"  Mediana:  {mediana:.4f}")
    print(f"  Std:      {std:.4f}")
    print(f"  Sesgo:    {sesgo:.4f}",
          "✓" if abs(sesgo) < 0.5 else "✇ sesgado")
    print(f"  Curtosis: {curtosis:.4f}",
          "✓" if abs(curtosis) < 1 else "✇ colas pesadas")

    # 2. Prueba de Shapiro-Wilk (mejor para n < 5000)
    if len(datos) <= 5000:
        stat_sw, p_sw = stats.shapiro(datos)
        print(f"\n Shapiro-Wilk:")
        print(f"  Estadístico: {stat_sw:.4f}")
        print(f"  p-value:     {p_sw:.4f}")
        print(f"  Resultado:   {'✓ Normal' if p_sw > 0.05 else '✗ No normal'}")

    # 3. Prueba de D'Agostino-Pearson (para n grande)
    stat_dp, p_dp = stats.normaltest(datos)
    print(f"\n D'Agostino-Pearson:")
    print(f"  Estadístico: {stat_dp:.4f}")
    print(f"  p-value:     {p_dp:.4f}")
    print(f"  Resultado:   {'✓ Normal' if p_dp > 0.05 else '✗ No normal'}")

    # 4. Prueba de Kolmogorov-Smirnov
    stat_ks, p_ks = stats.kstest(datos, 'norm', args=(media, std))
    print(f"\n Kolmogorov-Smirnov:")
    print(f"  Estadístico: {stat_ks:.4f}")
    print(f"  p-value:     {p_ks:.4f}")
    print(f"  Resultado:   {'✓ Normal' if p_ks > 0.05 else '✗ No normal'}")

    return media, mediana, std, sesgo

# Usar la función
verificar_normalidad(datos_estandar, 'Datos simulados normales')
verificar_normalidad(df['ventas_total'].dropna(), 'Ventas retail')

# ── VISUALIZAR LA DISTRIBUCIÓN ──
def visualizar_normalidad(datos, nombre='Variable'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    media = np.mean(datos)
    std   = np.std(datos)

    # 1. Histograma + curva normal teórica
    axes[0].hist(datos, bins=50, density=True,
                 color='steelblue', edgecolor='white',
                 alpha=0.7, label='Datos reales')

    x_range = np.linspace(datos.min(), datos.max(), 200)
    curva_normal = stats.norm.pdf(x_range, media, std)
    axes[0].plot(x_range, curva_normal,
                 color='red', linewidth=2.5,
                 label='Normal teórica')

    axes[0].axvline(media, color='orange', linestyle='--',
                    linewidth=2, label=f'Media: {media:.2f}')
    axes[0].set_title(f'Histograma vs Normal Teórica\n{nombre}')
    axes[0].legend(fontsize=8)

    # 2. Q-Q Plot (Quantile-Quantile)
    # Si los puntos siguen la línea diagonal → normal
    stats.probplot(datos, dist='norm', plot=axes[1])
    axes[1].set_title(f'Q-Q Plot\n{nombre}')
    axes[1].get_lines()[1].set_color('red')

    # 3. Box-Cox transformación (si no es normal)
    if stats.normaltest(datos)[1] < 0.05:
        datos_positivos = datos - datos.min() + 1
        datos_transformados, lambda_optimo = (
            stats.boxcox(datos_positivos))
        axes[2].hist(datos_transformados, bins=50, density=True,
                     color='teal', edgecolor='white', alpha=0.7)
        x_t = np.linspace(datos_transformados.min(), datos_transformados.max(), 200)
        m_t = np.mean(datos_transformados)
        s_t = np.std(datos_transformados)
        axes[2].plot(x_t, stats.norm.pdf(x_t, m_t, s_t), color='red', linewidth=2)
        axes[2].set_title(f'Después de Box-Cox\n (λ={lambda_optimo:.3f})')
    else:
        axes[2].text(0.5, 0.5, '✓ Ya es normal no necesita transformación',
                     ha='center', va='center', transform=axes[2].transAxes,
                     fontsize=14, color='green')
        axes[2].set_title('Transformación')

    plt.suptitle(f'Análisis de Normalidad — {nombre}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'images/normalidad_{nombre.lower()}.png', dpi=150)
    plt.show()

visualizar_normalidad(datos_estandar, 'Normal simulada')
```

---

## 📐 La Regla Empírica: 68-95-99.7

Esta es la regla más importante de la distribución normal.
Todo lo que sigue (Z-score, intervalos de confianza,
p-value) se basa en ella.

```python
# ── VISUALIZAR LA REGLA EMPÍRICA ──
fig, ax = plt.subplots(figsize=(14, 7))

x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x, 0, 1)

ax.plot(x, y, 'k-', linewidth=2.5, zorder=5)

# Áreas coloreadas
zonas = [
    (-1, 1, '#2196F3', '68.27%\n(μ ± 1σ)'),
    (-2, 2, '#64B5F6', '95.45%\n(μ ± 2σ)'),
    (-3, 3, '#BBDEFB', '99.73%\n(μ ± 3σ)'),
]

for x_min, x_max, color, label in zonas:
    x_fill = np.linspace(x_min, x_max, 500)
    y_fill = stats.norm.pdf(x_fill, 0, 1)
    ax.fill_between(x_fill, y_fill, alpha=0.5, color=color, label=label)

# Líneas verticales
for sigma, label in [(-3,'-3σ'),(-2,'-2σ'),(-1,'-1σ'), (0,'μ'),(1,'+1σ'),(2,'+2σ'),(3,'+3σ')]:
    ax.axvline(sigma, color='gray', linestyle='--', alpha=0.6, linewidth=1)
    ax.text(sigma, -0.015, label, ha='center', fontsize=9,
            fontweight='bold' if sigma == 0 else 'normal')

ax.set_title('Regla Empírica 68-95-99.7\nDistribución Normal Estándar (μ=0, σ=1)', fontsize=14)
ax.set_xlabel('Desviaciones estándar (σ)', fontsize=12)
ax.set_ylabel('Densidad de probabilidad', fontsize=12)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(-0.03, 0.45)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/regla_empirica.png', dpi=150)
plt.show()
```

```
68.27%  de los datos están entre  μ - 1σ  y  μ + 1σ
95.45%  de los datos están entre  μ - 2σ  y  μ + 2σ
99.73%  de los datos están entre  μ - 3σ  y  μ + 3σ

Fuera de μ ± 3σ → solo el 0.27% de los datos
→ Esos son tus outliers estadísticos
```

**Aplicación práctica inmediata:**

```python
media = df['ventas_total'].mean()
std   = df['ventas_total'].std()

rango_68 = (media - std,   media + std)
rango_95 = (media - 2*std, media + 2*std)
rango_99 = (media - 3*std, media + 3*std)

print(f"68% de ventas entre: ${rango_68[0]:,.2f} - ${rango_68[1]:,.2f}")
print(f"95% de ventas entre: ${rango_95[0]:,.2f} - ${rango_95[1]:,.2f}")
print(f"99% de ventas entre: ${rango_99[0]:,.2f} - ${rango_99[1]:,.2f}")

# Contar cuántos realmente caen en cada rango
n_total = len(df)
for rango, pct_teorico in [(rango_68, 68), (rango_95, 95), (rango_99, 99)]:
    n_rango = ((df['ventas_total'] >= rango[0]) & (df['ventas_total'] <= rango[1])).sum()
    pct_real = n_rango / n_total * 100
    print(f"Teórico {pct_teorico}% | Real {pct_real:.1f}%",
          "✓" if abs(pct_real - pct_teorico) < 5 else "✇")
```

---

## 📏 Z-Score: estandarizar para comparar

### Qué es

El Z-score convierte cualquier valor a una escala universal medida en desviaciones estándar.
Permite comparar valores de distribuciones con distintas medias y escalas.

```
Z = (X - μ) / σ

X = valor que quieres convertir
μ = media de la distribución
σ = desviación estándar
```

```python
# ── CALCULAR Z-SCORE ──

# Método 1: Manual
def z_score(valor, media, std):
    return (valor - media) / std

# Método 2: Scipy (para todo un array)
z_scores = stats.zscore(df['ventas_total'])

# Método 3: Sklearn (para ML — más usado)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
z_scores_scaled = scaler.fit_transform(df[['ventas_total']])

# ── INTERPRETAR EL Z-SCORE ──
def interpretar_z(z):
    abs_z = abs(z)
    if abs_z < 1:
        return f"Z={z:.2f} → Dentro de 1σ — muy común"
    elif abs_z < 2:
        return f"Z={z:.2f} → Entre 1σ y 2σ — inusual"
    elif abs_z < 3:
        return f"Z={z:.2f} → Entre 2σ y 3σ — raro"
    else:
        return f"Z={z:.2f} → Más de 3σ — outlier estadístico"

# Ejemplos
media_v = df['ventas_total'].mean()
std_v   = df['ventas_total'].std()

for venta in [16.81, 90.51, 500, 1000, 5000, 12236]:
    z = z_score(venta, media_v, std_v)
    print(f"Venta ${venta:,.2f}: {interpretar_z(z)}")

# ── USAR Z-SCORE PARA DETECTAR OUTLIERS ──
z_scores_array = np.abs(stats.zscore(
    df['ventas_total'].dropna()))

outliers_z = df[z_scores_array > 3]
print(f"\nOutliers (|Z| > 3): {len(outliers_z):,}")
print(f"Porcentaje: {len(outliers_z)/len(df)*100:.2f}%")
# Teóricamente debería ser ~0.27%
# Si es mucho más → datos no son normales

# ── VISUALIZAR Z-SCORES ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribución original
axes[0].hist(df['ventas_total'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_title(f'Distribución Original\nμ={media_v:.2f}, σ={std_v:.2f}')
axes[0].set_xlabel('Ventas ($)')

# Distribución estandarizada (Z-scores)
axes[1].hist(z_scores_array * np.sign(
    stats.zscore(df['ventas_total'].dropna())), bins=50,
             color='teal', edgecolor='white', alpha=0.8)
axes[1].axvline(-3, color='red', linestyle='--', label='±3σ (outliers)')
axes[1].axvline(3, color='red', linestyle='--')
axes[1].set_title('Z-Scores\nμ=0, σ=1 (estandarizado)')
axes[1].set_xlabel('Z-Score (desviaciones estándar)')
axes[1].legend()

plt.suptitle('Estandarización con Z-Score', fontsize=13)
plt.tight_layout()
plt.savefig('images/z_scores.png', dpi=150)
plt.show()
```

### Para qué sirve el Z-Score

```
1. DETECTAR OUTLIERS:
   |Z| > 3 → outlier estadístico (solo 0.27% teórico)

2. COMPARAR VARIABLES CON DISTINTAS ESCALAS:
   "¿Esta venta de $500 es más inusual que este tiempo de espera de 2 horas?"
   Z-score de ambos → comparables directamente

3. PREPARAR DATOS PARA ML:
   Muchos algoritmos (SVM, KNN, regresión logística) requieren variables en la misma escala.
   StandardScaler aplica Z-score a todo el dataset.

4. CALCULAR PROBABILIDADES:
   ¿Qué % de ventas son mayores a $500?
   → Calcula Z, busca en tabla o usa scipy
```

---

## 📊 P-Value: la probabilidad de estar equivocado

### Qué es

El p-value es la probabilidad de obtener un resultado tan extremo como el observado
**si la hipótesis nula fuera verdadera**.

```
p-value pequeño (< 0.05) →  Resultado muy improbable por azar
                         →  Rechazar hipótesis nula
                         →  "Hay evidencia estadística"

p-value grande (> 0.05)  →  Resultado puede ser por azar
                         →  No rechazar hipótesis nula
                         →  "No hay evidencia suficiente"
```

### La lógica detrás del p-value

```python
# Analogía simple:
# Lanzas una moneda 100 veces y obtienes 95 caras.
# ¿Es la moneda justa?

# Hipótesis nula (H0): la moneda es justa (p = 0.5)
# Hipótesis alternativa (H1): la moneda está cargada

# ¿Qué tan probable es obtener 95+ caras
# si la moneda fuera justa?

from scipy.stats import binom

p_valor = 1 - binom.cdf(94, 100, 0.5)
print(f"p-value = {p_valor:.10f}")
# p ≈ 0.0000000001
# → Imposible por azar → la moneda está cargada

# Ahora con 55 caras:
p_valor_55 = 1 - binom.cdf(54, 100, 0.5)
print(f"p-value (55 caras) = {p_valor_55:.4f}")
# p ≈ 0.1841
# → Podría ser azar → no podemos afirmar que está cargada
```

### Cómo calcularlo en análisis real

```python
# ── PRUEBA T: comparar medias de dos grupos ──
grupo_mexico = df[df['region']=='Mexico']['ventas_total']
grupo_area2  = df[df['region']=='Area 2']['ventas_total']

t_stat, p_value = stats.ttest_ind(grupo_mexico, grupo_area2)

print(f"Media México:  ${grupo_mexico.mean():,.2f}")
print(f"Media Area 2:  ${grupo_area2.mean():,.2f}")
print(f"t-estadístico: {t_stat:.4f}")
print(f"p-value:       {p_value:.6f}")

if p_value < 0.05:
    print("✅ Diferencia ESTADÍSTICAMENTE SIGNIFICATIVA")
    print("   La diferencia no es por azar")
else:
    print("⚠️ Diferencia NO significativa")
    print("   Puede ser por azar")

# ── PRUEBA DE CORRELACIÓN con p-value ──
r, p = stats.pearsonr(df['col1'], df['col2'])
print(f"\nr = {r:.4f}")
print(f"p = {p:.6f}")
print(f"¿Correlación significativa? "
      f"{'Sí' if p < 0.05 else 'No'}")

# ── ANOVA: comparar medias de múltiples grupos ──
grupos = [df[df['segmento']==s]['ventas_total']
          for s in df['segmento'].unique()]

f_stat, p_anova = stats.f_oneway(*grupos)
print(f"\nANOVA:")
print(f"F-estadístico: {f_stat:.4f}")
print(f"p-value:       {p_anova:.6f}")

if p_anova < 0.05:
    print("✓ Al menos un segmento tiene media diferente")
else:
    print("✇ No hay diferencia significativa entre segmentos")
```

### Las advertencias sobre el p-value

```python
# ADVERTENCIA 1: p < 0.05 no significa "efecto grande"
# Solo significa "estadísticamente significativo"
# Con n grande, diferencias triviales son significativas

n_grande = 100000
x1 = np.random.normal(100, 10, n_grande)
x2 = np.random.normal(100.1, 10, n_grande)  # diferencia de 0.1

t, p = stats.ttest_ind(x1, x2)
print(f"Diferencia: 0.1 unidades")
print(f"p-value: {p:.6f}")
# → p < 0.05 aunque la diferencia sea irrelevante
# → Con n=100,000 casi todo es "significativo"

# ADVERTENCIA 2: p > 0.05 no significa "sin efecto"
# Solo significa "no hay evidencia suficiente"
# Puede ser por muestra pequeña

n_pequeno = 20
x1 = np.random.normal(100, 10, n_pequeno)
x2 = np.random.normal(115, 10, n_pequeno)  # diferencia real de 15

t, p = stats.ttest_ind(x1, x2)
print(f"\nDiferencia real: 15 unidades")
print(f"p-value: {p:.4f}")
# → p puede ser > 0.05 con n=20 aunque haya efecto real

# SOLUCIÓN: reportar siempre TAMAÑO DEL EFECTO + p-value
def cohen_d(grupo1, grupo2):
    """Tamaño del efecto: qué tan grande es la diferencia"""
    diff = grupo1.mean() - grupo2.mean()
    pooled_std = np.sqrt(
        (grupo1.std()**2 + grupo2.std()**2) / 2)
    return diff / pooled_std

d = cohen_d(grupo_mexico, grupo_area2)
print(f"\nCohen's d = {d:.3f}")
print("Interpretación:",
      "pequeño" if abs(d) < 0.5
      else "mediano" if abs(d) < 0.8
      else "grande")
```

---

## 📊 Intervalos de Confianza

### Qué es

Un intervalo de confianza del 95% significa:
**"Si repitiéramos este estudio 100 veces, en 95 de ellas el intervalo contendría el valor real de la población."**

**Lo que NO significa:**
"Hay 95% de probabilidad de que el valor real esté en este intervalo" ← esa es la interpretación más común y más incorrecta.

```python
# ── CALCULAR INTERVALO DE CONFIANZA ──
def intervalo_confianza(datos, confianza=0.95):
    """
    Calcula el IC para la media de una muestra.
    """
    n     = len(datos)
    media = np.mean(datos)
    std   = np.std(datos, ddof=1)   # ddof=1 para muestra
    error = std / np.sqrt(n)        # error estándar

    # Valor crítico t (mejor que Z para muestras finitas)
    t_critico = stats.t.ppf((1 + confianza) / 2, df=n-1)

    margen = t_critico * error
    limite_inf = media - margen
    limite_sup = media + margen

    return media, limite_inf, limite_sup, margen

media, ic_inf, ic_sup, margen = intervalo_confianza(df['ventas_total'], 0.95)

print(f"Media muestral:     ${media:,.2f}")
print(f"IC 95%:             [${ic_inf:,.2f}, ${ic_sup:,.2f}]")
print(f"Margen de error:    ±${margen:,.2f}")
print(f"\nInterpretación:")
print(f"Estamos 95% seguros de que la media REAL")
print(f"de ventas por transacción está entre")
print(f"${ic_inf:,.2f} y ${ic_sup:,.2f}")

# ── COMPARAR ICs con distintos niveles ──
fig, ax = plt.subplots(figsize=(10, 6))

niveles = [0.80, 0.90, 0.95, 0.99]
colores = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

for i, (nivel, color) in enumerate(
        zip(niveles, colores)):
    m, inf, sup, _ = intervalo_confianza(
        df['ventas_total'].sample(1000), nivel)
    ax.errorbar(nivel, m,
                yerr=[[m-inf], [sup-m]],
                fmt='o', color=color,
                capsize=8, capthick=2,
                linewidth=2, markersize=8,
                label=f'IC {nivel*100:.0f}%: '
                      f'[${inf:.2f}, ${sup:.2f}]')

ax.set_xlabel('Nivel de confianza', fontsize=12)
ax.set_ylabel('Ventas ($)', fontsize=12)
ax.set_title('Intervalos de Confianza\nMayor confianza = intervalo más ancho', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/intervalos_confianza.png', dpi=150)
plt.show()
```

### La relación entre IC, p-value y significancia

```
IC 95%  ←→  p-value < 0.05  ←→  nivel de significancia α = 0.05

Son tres formas de decir lo mismo:

Si el IC 95% NO incluye el 0 (en diferencias)
→ la diferencia es estadísticamente significativa
→ p-value < 0.05

Si el IC 95% SÍ incluye el 0
→ la diferencia NO es significativa
→ p-value > 0.05

Ejemplo:
Diferencia de medias México vs Area 2 = $220
IC 95% de la diferencia: [$180, $260]
→ No incluye 0 → significativa → p < 0.05 ✓
```

---

## 🔄 Teorema del Límite Central (TLC)

### Qué es

El teorema más importante de la estadística:

> **Si tomas muestras suficientemente grandes de CUALQUIER distribución,
> la distribución de las MEDIAS muestrales será aproximadamente normal.**

```
Distribución original:      Distribución de medias muestrales:
puede ser cualquier forma   siempre se acerca a una campana

█                               ████
█ █                            ██████
██ █  ← sesgada                ████████
██████                         ██████████
                               ████████
                               ██████
                                ████
```

```python
# ── DEMOSTRAR EL TLC VISUALMENTE ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

np.random.seed(42)

# Distribuciones originales NO normales
distribuciones = {
    'Exponencial\n(muy sesgada)':
        np.random.exponential(scale=2, size=100000),
    'Uniforme\n(completamente plana)':
        np.random.uniform(0, 10, size=100000),
    'Ventas retail\n(sesgo extremo)':
        df['ventas_total'].dropna().values
}

for col, (nombre, datos_orig) in enumerate(distribuciones.items()):

    # Fila 1: Distribución original
    axes[0, col].hist(datos_orig[:5000], bins=50, color='steelblue',
                      edgecolor='white', alpha=0.8, density=True)
    axes[0, col].set_title(f'Distribución original\n{nombre}', fontsize=10)

    # Fila 2: Distribución de medias muestrales (TLC)
    n_muestras = 1000
    tamanio_muestra = 30
    medias = [np.mean(np.random.choice(datos_orig, tamanio_muestra))
              for _ in range(n_muestras)]

    axes[1, col].hist(medias, bins=40, color='teal', edgecolor='white',
                      alpha=0.8, density=True)

    # Curva normal teórica sobre las medias
    x_range = np.linspace(min(medias), max(medias), 200)
    axes[1, col].plot(
        x_range,
        stats.norm.pdf(x_range, np.mean(medias), np.std(medias)),
                       color='red', linewidth=2.5)

    axes[1, col].set_title(
        f'Medias de {n_muestras} muestras (n={tamanio_muestra})\n'
        f'→ Se vuelve normal (TLC)',
        fontsize=10)

plt.suptitle(
    'Teorema del Límite Central\n'
    'Cualquier distribución → medias muestrales normales',
    fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/teorema_limite_central.png', dpi=150)
plt.show()

print("✓ El TLC explica por qué las pruebas estadísticas funcionan incluso con datos no normales, ")
print("   siempre que la muestra sea suficientemente grande.")
```

### Por qué el TLC importa en la práctica

```
1. JUSTIFICA EL USO DE PRUEBAS ESTADÍSTICAS:
   Tus datos de ventas NO son normales.
   Pero con n=122,002 transacciones, la distribución de medias SÍ es normal.
   → Puedes usar pruebas t, Z, ANOVA legítimamente.

2. DEFINE "SUFICIENTEMENTE GRANDE":
   → n ≥ 30: el TLC ya funciona para distribuciones moderadamente sesgadas
   → n ≥ 100: funciona para distribuciones muy sesgadas como ventas retail
   → n ≥ 1000: funciona para casi cualquier cosa

3. EXPLICA POR QUÉ LA NORMALIDAD IMPORTA EN ML:
   Muchos algoritmos asumen que los RESIDUOS (errores del modelo) son normales, no que los datos sean normales.
   El TLC garantiza que con suficientes datos, los residuos tienden a normalidad.
```

---

## 🤖 Supuestos de normalidad en ML

### Qué algoritmos asumen normalidad

```python
# Algoritmos que REQUIEREN o SE BENEFICIAN de normalidad:
supuestos_normalidad = {
    'Regresión Lineal': {
        'qué asume': 'Residuos normales (no los datos)',
        'cómo verificar': 'Q-Q plot de residuos',
        'qué pasa si no': 'Intervalos de confianza incorrectos'
    },
    'Regresión Logística': {
        'qué asume': 'Nada sobre los datos, pero...',
        'cómo verificar': 'Verificar multicolinealidad',
        'qué pasa si no': 'Estimadores menos eficientes'
    },
    'LDA (Linear Discriminant Analysis)': {
        'qué asume': 'Datos normales por clase',
        'cómo verificar': 'Shapiro-Wilk por clase',
        'qué pasa si no': 'Clasificación menos precisa'
    },
    'Naive Bayes Gaussiano': {
        'qué asume': 'Features normales por clase',
        'cómo verificar': 'Histogramas por clase',
        'qué pasa si no': 'Probabilidades incorrectas'
    },
    'Pruebas t y ANOVA': {
        'qué asume': 'Normalidad o n grande (TLC)',
        'cómo verificar': 'Shapiro-Wilk o n ≥ 30',
        'qué pasa si no': 'p-values incorrectos'
    }
}

# Algoritmos ROBUSTOS que NO asumen normalidad:
robustos = [
    'Árboles de decisión',
    'Random Forest',
    'Gradient Boosting',
    'KNN',
    'SVM con kernel RBF',
    'Redes neuronales'
]

print("✓ Algoritmos robustos (no asumen normalidad):")
for alg in robustos:
    print(f"   → {alg}")
```

### Cómo verificar supuestos de regresión lineal

```python
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Ajustar modelo
X = df[['precio_unitario', 'unidades_vendidas']]
y = df['ventas_total']

modelo = LinearRegression()
modelo.fit(X, y)
y_pred = modelo.predict(X)
residuos = y - y_pred

# ── LOS 4 SUPUESTOS DE REGRESIÓN LINEAL ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# SUPUESTO 1: Residuos con media = 0
# (sin sesgo sistemático)
axes[0, 0].scatter(y_pred, residuos,
                   alpha=0.3, s=10, color='steelblue')
axes[0, 0].axhline(0, color='red', linestyle='--',
                   linewidth=2)
axes[0, 0].set_title('Supuesto 1: Residuos vs Ajustados\n'
                     '✓ Puntos deben distribuirse\naleatoriamente alrededor de 0')
axes[0, 0].set_xlabel('Valores ajustados (ŷ)')
axes[0, 0].set_ylabel('Residuos')

# SUPUESTO 2: Normalidad de residuos (Q-Q plot)
stats.probplot(residuos, dist='norm', plot=axes[0, 1])
axes[0, 1].get_lines()[1].set_color('red')
axes[0, 1].set_title('Supuesto 2: Normalidad de Residuos\n'
                     '✓ Puntos deben seguir la línea diagonal')

# SUPUESTO 3: Homocedasticidad
# (varianza constante de residuos)
axes[1, 0].scatter(y_pred, np.abs(residuos),
                   alpha=0.3, s=10, color='teal')
axes[1, 0].set_title('Supuesto 3: Homocedasticidad\n'
                     '✓ Dispersión debe ser\nuniforme en todo el rango')
axes[1, 0].set_xlabel('Valores ajustados (ŷ)')
axes[1, 0].set_ylabel('|Residuos|')

# SUPUESTO 4: Distribución de residuos
axes[1, 1].hist(residuos, bins=50, color='steelblue', edgecolor='white',
                alpha=0.8, density=True)
x_r = np.linspace(residuos.min(), residuos.max(), 200)
axes[1, 1].plot(x_r,
                stats.norm.pdf(x_r,
                               residuos.mean(),
                               residuos.std()),
                color='red', linewidth=2.5,
                label='Normal teórica')
axes[1, 1].set_title('Supuesto 4: Histograma de Residuos\n'
                     '✓ Debe parecerse a una campana')
axes[1, 1].legend()

plt.suptitle('Los 4 Supuestos de Regresión Lineal\n(Verificación visual)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/supuestos_regresion.png', dpi=150)
plt.show()

# Prueba formal de normalidad en residuos
_, p_normalidad = stats.normaltest(residuos)
print(f"\nPrueba de normalidad en residuos:")
print(f"p-value = {p_normalidad:.6f}")
print("✓ Residuos normales" if p_normalidad > 0.05
      else "✇ Residuos no normales, considerar transformación")
```

---

## 🔄 Transformaciones para normalizar datos

Cuando tus datos no son normales y el algoritmo lo requiere:

```python
# ── LAS 4 TRANSFORMACIONES MÁS COMUNES ──
fig, axes = plt.subplots(2, 4, figsize=(20, 8))

datos = df['ventas_total'].dropna()
datos_pos = datos + 1  # asegurar positivos

transformaciones = [
    ('Original', datos, 'steelblue'),
    ('Log (ln)', np.log(datos_pos), 'teal'),
    ('Log10', np.log10(datos_pos), 'coral'),
    ('Raíz cuadrada', np.sqrt(datos_pos), 'gold'),
]

for col, (nombre, datos_t, color) in enumerate(transformaciones):
    # Histograma
    axes[0, col].hist(datos_t, bins=50, color=color, edgecolor='white', alpha=0.8, density=True)
    axes[0, col].set_title(f'{nombre}', fontsize=10)

    # Q-Q plot
    stats.probplot(datos_t, dist='norm', plot=axes[1, col])
    axes[1, col].get_lines()[1].set_color('red')
    axes[1, col].set_title(f'Q-Q: {nombre}', fontsize=9)

    # Prueba de normalidad
    _, p = stats.normaltest(datos_t)
    estado = '✓' if p > 0.05 else '✗'
    axes[0, col].set_xlabel(f'{estado} p={p:.3f}', fontsize=9)

plt.suptitle(
    'Transformaciones para Normalizar Datos\n'
    '(Ventas Retail: distribución original muy sesgada)',
    fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('images/transformaciones_normalidad.png', dpi=150)
plt.show()
```

---

## <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzMxY200dzhyc2RhbG5zbnk3MWZoZHE2ODZnM3I3Y2lseHNtYXI4diZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/WmvhTzEl9ihy9TqPme/giphy.gif" width="40"> Analogía con Medicina Tradicional China

La distribución normal en MTC es como el concepto de equilibrio del Qi:

| Distribución Normal | Equilibrio en MTC |
|---|---|
| Media μ | El estado de equilibrio → ni exceso ni deficiencia |
| ±1σ (68%) | Variación normal → dentro del rango saludable |
| ±2σ (95%) | Variación leve → requiere atención preventiva |
| ±3σ (99.7%) | Variación severa → condición que requiere tratamiento |
| Outlier (> 3σ) | Desequilibrio extremo → urgencia clínica |
| Distribución sesgada | Patrón de exceso o deficiencia crónica |
| TLC | El cuerpo tiende al equilibrio con suficiente tiempo y muestras |
| Transformar a normal | Restaurar el equilibrio → el objetivo del tratamiento |

> En MTC, el diagnóstico de "exceso de Yang" es estadísticamente una distribución sesgada hacia la derecha,
> la mayoría de los síntomas están en la zona de exceso (por encima de la media).
> El tratamiento busca llevar la distribución de vuelta a la campana centrada en μ.

---

## ⚡ Cheat Sheet rápido

```
DISTRIBUCIÓN NORMAL:
→ Campana simétrica: media = mediana = moda
→ Completamente definida por μ y σ
→ Área total = 1 (100%)

REGLA EMPÍRICA:
→ μ ± 1σ = 68.27% de los datos
→ μ ± 2σ = 95.45% de los datos
→ μ ± 3σ = 99.73% de los datos
→ > 3σ   = outlier estadístico (0.27%)

Z-SCORE:
→ Z = (X - μ) / σ
→ |Z| < 1 = muy común
→ |Z| 1-2 = inusual
→ |Z| 2-3 = raro
→ |Z| > 3 = outlier estadístico

P-VALUE:
→ p < 0.05 → significativo → rechazar H0
→ p > 0.05 → no significativo → no rechazar H0
→ p pequeño ≠ efecto grande (con n grande todo es sig.)
→ p grande ≠ sin efecto (puede ser muestra chica)
→ SIEMPRE reportar tamaño del efecto además del p-value

INTERVALOS DE CONFIANZA:
→ IC 95% = resultado más común
→ Más confianza = intervalo más ancho
→ IC 95% que no incluye 0 → p < 0.05

TLC:
→ Cualquier distribución → medias muestrales normales
→ Necesitas n ≥ 30 (mínimo)
→ Justifica usar pruebas estadísticas con datos no normales

VERIFICAR NORMALIDAD:
→ Visual: histograma + Q-Q plot (siempre primero)
→ Formal: Shapiro-Wilk (n < 5000) o D'Agostino (n grande)
→ Si no es normal: transformar (log, raíz) o usar algoritmos robustos

SUPUESTOS DE REGRESIÓN LINEAL:
→ Residuos normales (no los datos)
→ Residuos con media = 0
→ Homocedasticidad (varianza constante)
→ Independencia de residuos
```

---

## 🔗 Aplicado a mis proyectos

### Proyecto EDA: Ventas Retail

```python
# Normalidad en mi dataset:

# ventas_total: NO normal
# → Sesgo extremo positivo (media=$90 vs mediana=$16)
# → Shapiro-Wilk: p ≈ 0 → rechaza normalidad
# → Transformación log → se acerca a normal
# → Implicación: usar Spearman en lugar de Pearson para correlaciones más robustas
# → Para ML: usar algoritmos robustos (Random Forest, Gradient Boosting) o transformar antes de regresión

# Z-scores para detectar outliers:
z = np.abs(stats.zscore(df['ventas_total']))
outliers_z3 = (z > 3).sum()
print(f"Outliers |Z|>3: {outliers_z3:,} "
      f"({outliers_z3/len(df)*100:.2f}%)")
# → Teórico: 0.27% | Real: mucho más
# → Confirma que los datos no son normales
# → Los "outliers" son transacciones mayoristas reales

# TLC aplicado:
# → Con n=122,002 el TLC garantiza que las pruebas estadísticas son válidas aunque los datos no sean normales
# → La distribución de medias muestrales sí es normal con n tan grande
```

### Posibles proyectos futuros
| Dataset | Normalidad esperada | Transformación probable |
|---|---|---|
| SEMARNAT emisiones | No normal → sesgo positivo | Log transform |
| OMS esperanza de vida | Aproximadamente normal | Ninguna |
| INEGI ingresos | No normal → muy sesgado | Log transform |
| Kaggle fraude | No normal → binaria/discreta | Ninguna → usar robustos |
| Temperatura CDMX | Aproximadamente normal | Ninguna |

---

*Nota creada: 2026 · Serie: Estadística para Ciencia de Datos*
*Parte del certificado Científico de Datos — EBAC*
*🪷 github.com/ReginaPema*
