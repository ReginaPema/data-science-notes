# Correlation & R²
## Correlación y Coeficiente de Determinación

> La correlación te dice si dos variables se mueven juntas y en qué dirección.
> La R² te dice qué tanto de lo que le pasa a una variable es explicado por la otra.

---

## 🔢 Parte 1: Coeficiente de Correlación (r)

### Qué es
El coeficiente de correlación de Pearson (r) mide la **fuerza y dirección** de la relación lineal entre dos variables numéricas.

```
r = covarianza(X, Y) / (std(X) × std(Y))
```

Siempre está entre -1 y +1:

```
-1.0  ←──────────────0──────────────→  +1.0

Negativa           Sin             Positiva
perfecta        correlación        perfecta
```

### Cómo calcularlo

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ── MÉTODO 1: Pandas (más simple) ──
r = df['col1'].corr(df['col2'])
print(f"r = {r:.4f}")

# ── MÉTODO 2: Scipy (incluye p-value) ──
r, p_value = stats.pearsonr(df['col1'], df['col2'])
print(f"r = {r:.4f}")
print(f"p-value = {p_value:.6f}")
print(f"¿Significativo? {'Sí' if p_value < 0.05 else 'No'}")

# ── MÉTODO 3: Matriz completa (todas vs todas) ──
matriz = df.select_dtypes(include='number').corr()
print(matriz.round(3))

# ── VISUALIZAR la correlación entre dos variables ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter con línea de tendencia
axes[0].scatter(df['col1'], df['col2'], alpha=0.4, s=20, color='steelblue')
z = np.polyfit(df['col1'], df['col2'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['col1'].min(), df['col1'].max(), 100)
axes[0].plot(x_line, p(x_line), color='red', linewidth=2)
axes[0].set_title(f'Scatter — r = {r:.3f}')
axes[0].set_xlabel('Variable 1')
axes[0].set_ylabel('Variable 2')

# Heatmap de la matriz completa
mask = np.triu(np.ones_like(matriz, dtype=bool))
sns.heatmap(matriz, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            mask=mask, square=True, ax=axes[1])
axes[1].set_title('Matriz de Correlación')

plt.tight_layout()
plt.savefig('images/correlacion_completa.png', dpi=150)
plt.show()
```

---

### Cómo interpretar r
```
r = +1.00  →  Perfecta positiva      → imposible en datos reales
r = +0.90  →  Muy fuerte positiva    → variables casi idénticas
r = +0.70  →  Fuerte positiva        → relación clara y confiable
r = +0.50  →  Moderada positiva      → relación visible pero con ruido
r = +0.30  →  Débil positiva         → tendencia leve
r = +0.10  →  Muy débil              → casi sin relación
r =  0.00  →  Sin correlación lineal
r = -0.10  →  Muy débil negativa
r = -0.30  →  Débil negativa
r = -0.50  →  Moderada negativa
r = -0.70  →  Fuerte negativa
r = -0.90  →  Muy fuerte negativa
r = -1.00  →  Perfecta negativa      → imposible en datos reales
```

**En la práctica en negocios:**

| Rango | Interpretación práctica |
|---|---|
| `|r| > 0.8` | Relación muy fuerte → cuidado con multicolinealidad en ML |
| `0.6 < |r| ≤ 0.8` | Relación fuerte → útil para predicción |
| `0.4 < |r| ≤ 0.6` | Relación moderada → hay relación pero con mucho ruido |
| `0.2 < |r| ≤ 0.4` | Relación débil → presente pero poco confiable |
| `|r| ≤ 0.2` | Sin relación lineal significativa |

---

### Los 3 tipos de correlación

```python
# ── PEARSON: para datos numéricos continuos ──
# El más común. Asume relación lineal.
r_pearson, p = stats.pearsonr(df['col1'], df['col2'])
print(f"Pearson r = {r_pearson:.3f}")

# ── SPEARMAN: para datos ordinales o no lineales ──
# Mide correlación de rangos — más robusto ante outliers
r_spearman, p = stats.spearmanr(df['col1'], df['col2'])
print(f"Spearman r = {r_spearman:.3f}")

# ── KENDALL: para muestras pequeñas ──
r_kendall, p = stats.kendalltau(df['col1'], df['col2'])
print(f"Kendall τ = {r_kendall:.3f}")
```

| Tipo | Cuándo usarlo |
|---|---|
| **Pearson** | Datos continuos, relación lineal, sin muchos outliers |
| **Spearman** | Datos ordinales, relación no lineal, con outliers extremos |
| **Kendall** | Muestras pequeñas (< 30 filas), datos con empates |

```python
# Comparar los tres: si son muy distintos, la relación puede no ser lineal o hay outliers
print(f"Pearson: {stats.pearsonr(x, y)[0]:.3f}")
print(f"Spearman: {stats.spearmanr(x, y)[0]:.3f}")
print(f"Kendall: {stats.kendalltau(x, y)[0]:.3f}")

# Si Pearson << Spearman → hay outliers distorsionando
# Si todos son similares → la relación es lineal y limpia
```

---

### La advertencia más importante: Correlación ≠ Causalidad

```python
# Ejemplos reales de correlaciones sin causalidad:

correlaciones_espurias = {
    'helados_vs_ahogamientos': 0.97,
    # → Ambas suben en verano. El calor es la causa real.

    'tv_per_capita_vs_esperanza_vida': 0.89,
    # → Ambas indican nivel de desarrollo económico.

    'nicolas_cage_peliculas_vs_ahogados_en_piscina': 0.67,
    # → Correlación espuria pura: coincidencia estadística.

    'margarina_consumida_vs_divorcios_maine': 0.99,
    # → El ejemplo clásico de correlación sin sentido.
}

# Preguntas que SIEMPRE debes hacerte:
preguntas = [
    "¿Hay una tercera variable que explique ambas?",
    "¿La relación tiene sentido lógico/causal?",
    "¿Es una coincidencia estadística?",
    "¿La relación se mantiene en subgrupos?",
    "¿Existe un mecanismo causal plausible?"
]
```

**Regla práctica:**
La correlación identifica candidatos para investigar causalidad, no la prueba.
La causalidad requiere experimentos controlados o conocimiento del dominio.

---

### El Cuarteto de Anscombe: por qué SIEMPRE graficar primero

Cuatro datasets completamente distintos con exactamente la misma correlación:

```python
# Los cuatro datasets de Anscombe tienen todos: r = 0.816, media_x = 9, media_y = 7.5

import matplotlib.pyplot as plt
import numpy as np

# Dataset I: correlación lineal real
x1 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]

# Dataset II: relación curvilínea
x2 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]

# Dataset III: correlación con outlier
x3 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]

# Dataset IV: outlier extremo
x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
datasets = [(x1,y1,'I — Lineal real'),
            (x2,y2,'II — Curvilínea'),
            (x3,y3,'III — Outlier distorsiona'),
            (x4,y4,'IV — Solo un outlier')]

for ax, (x, y, titulo) in zip(axes.flatten(), datasets):
    r_val = np.corrcoef(x, y)[0, 1]
    ax.scatter(x, y, color='steelblue', s=60, edgecolors='white', zorder=3)

    # Línea de regresión
    z = np.polyfit(x, y, 1)
    p_line = np.poly1d(z)
    x_range = np.linspace(min(x)-0.5, max(x)+0.5, 100)
    ax.plot(x_range, p_line(x_range), color='red', linewidth=2)

    ax.set_title(f'Dataset {titulo}\nr = {r_val:.3f}', fontsize=11)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)

plt.suptitle('Cuarteto de Anscombe | r = 0.816 en los 4 casos\n'
             '¡Siempre grafica antes de calcular r!', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/anscombe_quartet.png', dpi=150)
plt.show()

print("Moraleja: el número solo no es suficiente.")
print("El scatter plot revela lo que r oculta.")
```

---

## 🔢 Parte 2: Coeficiente de Determinación (R²)

### Qué es

R² (R cuadrada) es simplemente r al cuadrado:

```
R² = r²
```

Pero su interpretación es completamente distinta y mucho más intuitiva:

> **R² = el porcentaje de la variación de Y que es explicado por X**

```
Si r = 0.92  →  R² = 0.92² = 0.846
→ El 84.6% de la variación en Y es explicada por X
→ El 15.4% restante viene de otros factores no considerados
```

### Cómo calcularlo

```python
# ── MÉTODO 1: Directo desde r ──
r, p = stats.pearsonr(df['col1'], df['col2'])
r_cuadrada = r ** 2
print(f"r  = {r:.4f}")
print(f"R² = {r_cuadrada:.4f}")
print(f"R² = {r_cuadrada*100:.1f}%")

# ── MÉTODO 2: Con sklearn (más usado en ML) ──
from sklearn.metrics import r2_score

# Primero necesitas predicciones de un modelo
from sklearn.linear_model import LinearRegression

X = df[['col1']]    # feature, debe ser 2D
y = df['col2']      # target

modelo = LinearRegression()
modelo.fit(X, y)
y_pred = modelo.predict(X)

r2 = r2_score(y, y_pred)
print(f"R² del modelo = {r2:.4f} ({r2*100:.1f}%)")

# ── VISUALIZAR r vs R² ──
valores_r = np.arange(0, 1.01, 0.1)
valores_r2 = valores_r ** 2

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(valores_r, valores_r2, width=0.08, color='steelblue', edgecolor='white', alpha=0.8)

# Línea diagonal para comparar
ax.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1.5, label='r = R² (referencia)')

ax.set_xlabel('Correlación (r)', fontsize=12)
ax.set_ylabel('R²', fontsize=12)
ax.set_title('Relación entre r y R²\n' 'R² siempre es menor o igual que r', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)

# Anotaciones clave
for r_val in [0.3, 0.5, 0.7, 0.9]:
    r2_val = r_val**2
    ax.annotate(f'r={r_val}\nR²={r2_val:.2f}', xy=(r_val, r2_val),
                xytext=(r_val + 0.05, r2_val - 0.08), fontsize=8,
                arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.savefig('images/r_vs_r_cuadrada.png', dpi=150)
plt.show()
```

---

### Cómo interpretar R²

```
R² = 1.00  →  100% → modelo perfecto (imposible en realidad)
R² = 0.90  →  90%  → excelente: X explica casi todo
R² = 0.70  →  70%  → bueno: relación fuerte y útil
R² = 0.50  →  50%  → moderado: X explica la mitad
R² = 0.30  →  30%  → débil: hay relación pero hay mucho más
R² = 0.10  →  10%  → muy débil: X explica muy poco
R² = 0.00  →  0%   → X no explica nada de Y
```

**Por industria, qué se considera "bueno":**

| Industria | R² "bueno" | Por qué |
|---|---|---|
| Física / Ingeniería | > 0.95 | Sistemas controlados y predecibles |
| Finanzas | > 0.70 | Mercados con algo de ruido |
| Marketing | > 0.50 | Comportamiento humano es variable |
| Ciencias sociales | > 0.30 | Alta variabilidad humana y cultural |
| Salud / Medicina | > 0.40 | Muchos factores confundidores |
| Medio ambiente | > 0.50 | Sistemas complejos pero medibles |

> No existe un R² "bueno" universal. Depende completamente del contexto y de qué tan complejo es el fenómeno.
> Un R² de 0.30 puede ser excelente en predicción de comportamiento humano y terrible en ingeniería mecánica.

---

### La parte que R² no te dice

```python
# R² alto NO garantiza un buen modelo
# R² bajo NO significa que el modelo sea inútil

# Ejemplo 1: R² alto pero modelo incorrecto (relación curvilínea modelada linealmente)
x_curva = np.linspace(-3, 3, 100)
y_curva = x_curva**2 + np.random.normal(0, 0.5, 100)

modelo_lineal = LinearRegression()
modelo_lineal.fit(x_curva.reshape(-1, 1), y_curva)
y_pred_lineal = modelo_lineal.predict(x_curva.reshape(-1, 1))
r2_incorrecto = r2_score(y_curva, y_pred_lineal)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(x_curva, y_curva, alpha=0.5, color='steelblue', s=20)
axes[0].plot(x_curva, y_pred_lineal, color='red', linewidth=2)
axes[0].set_title(f'R² = {r2_incorrecto:.2f}\n'
                  f'✗ Modelo incorrecto: relación es curvilínea')

# Ejemplo 2: R² bajo pero relación real y útil
x_ruido = np.random.uniform(0, 10, 200)
y_ruido = 2 * x_ruido + np.random.normal(0, 8, 200)
r2_bajo = r2_score(y_ruido, LinearRegression()
                   .fit(x_ruido.reshape(-1,1), y_ruido)
                   .predict(x_ruido.reshape(-1,1)))

axes[1].scatter(x_ruido, y_ruido, alpha=0.4,
                color='teal', s=20)
axes[1].set_title(f'R² = {r2_bajo:.2f}\n'
                  f'✓ R² bajo pero tendencia real y útil')

plt.suptitle('R² alto ≠ buen modelo | R² bajo ≠ sin relación', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('images/r2_advertencias.png', dpi=150)
plt.show()
```

---

### R² ajustada, para modelos con múltiples variables

Cuando tienes más de una variable predictora,
usa R² ajustada en lugar de R²:

```python
# R² simple siempre sube al agregar variables aunque sean inútiles, es una trampa

# R² ajustada penaliza por variables innecesarias. Baja si agregas variables que no aportan

def r2_ajustada(r2, n, k):
    """
    r2 = R² del modelo
    n  = número de observaciones
    k  = número de variables predictoras
    """
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

# Ejemplo:
n = len(df)           # filas del dataset
k = 3                 # número de features usados
r2 = 0.75             # R² del modelo

r2_adj = r2_ajustada(r2, n, k)
print(f"R²          = {r2:.4f}")
print(f"R² ajustada = {r2_adj:.4f}")

# Si R² ajustada << R² → tienes variables inútiles
# Si R² ajustada ≈ R² → todas tus variables aportan
```

---

## 🔗 Cómo se conectan r y R²

```
r (correlación)                R² (determinación)
────────────────               ─────────────────
Mide: fuerza y                 Mide: % de varianza
dirección de la                de Y explicado por X
relación lineal

Va de -1 a +1                  Va de 0 a 1
(tiene dirección)              (sin dirección)

Útil para: entender            Útil para: evaluar
si dos variables               qué tan bien predice
se relacionan                  un modelo

Se calcula primero             Se calcula a partir de r
                               o de las predicciones del modelo

r = 0.92               →       R² = 0.92² = 0.846
"correlación                   "el modelo explica
muy fuerte"                    el 84.6% de la variación"
```

```python
# Visualización de la relación entre los dos
fig, ax = plt.subplots(figsize=(10, 6))

r_valores = np.linspace(-1, 1, 200)
r2_valores = r_valores ** 2

ax.plot(r_valores, r2_valores, color='steelblue', linewidth=3)
ax.fill_between(r_valores, 0, r2_valores, alpha=0.2, color='steelblue')

# Marcar puntos clave
puntos_clave = [-0.9, -0.7, -0.5, 0.5, 0.7, 0.9]
for r_punto in puntos_clave:
    r2_punto = r_punto ** 2
    ax.plot(r_punto, r2_punto, 'ro', markersize=8)
    ax.annotate(f'r={r_punto}\nR²={r2_punto:.2f}', xy=(r_punto, r2_punto),
                xytext=(r_punto + 0.05, r2_punto + 0.05), fontsize=8)

ax.set_xlabel('Correlación (r)', fontsize=12)
ax.set_ylabel('R²', fontsize=12)
ax.set_title('De r a R² — la curva de conversión', fontsize=13)
ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='R² = 0.50')
ax.axvline(0.707, color='red', linestyle='--', alpha=0.5, label='r = 0.707 → R² = 0.50')
ax.axvline(-0.707, color='red', linestyle='--', alpha=0.5)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/r_a_r2_curva.png', dpi=150)
plt.show()

# La curva muestra algo importante:
# r = 0.70 → R² = 0.49 (solo explica el 49%)
# r = 0.90 → R² = 0.81 (explica el 81%)
# Necesitas r muy alto para tener R² realmente útil
```

---

## 🌍 Casos reales por industria

### 💰 Ventas y Retail

```python
# Escenario de tu proyecto EDA:
r_unidades_ventas = 0.92
r2 = r_unidades_ventas ** 2

print(f"r  = {r_unidades_ventas}")
print(f"R² = {r2:.3f} ({r2*100:.1f}%)")

# Interpretación:
# → El 84.6% de la variación en el valor de venta
#   es explicado por las unidades vendidas
# → El 15.4% restante viene de otros factores:
#   precio unitario, descuentos, mezcla de productos

# Insight de negocio:
# → Unidades es el driver principal del ingreso
# → Para aumentar ventas: aumentar volumen es más efectivo que subir precios
#   (confirmado por la elasticidad del scatter)
```

### 🌿 Medio Ambiente

```python
# Hipotético: PIB estatal vs emisiones CO₂
r_pib_emisiones = 0.78
r2_pib_emisiones = r_pib_emisiones ** 2

print(f"R² = {r2_pib_emisiones:.2f}")
# → El PIB explica el 60.8% de las emisiones
# → El 39.2% restante: tipo de industria, eficiencia energética,
#   mix energético, regulación ambiental

# Insight de política pública:
# → No basta controlar el PIB para predecir emisiones
# → Hay factores estructurales independientes que explican el 40% restante
```

### 🏥 Salud

```python
# Hipotético: edad vs presión arterial sistólica
r_edad_presion = 0.55
r2_edad_presion = r_edad_presion ** 2

print(f"R² = {r2_edad_presion:.2f}")
# → La edad explica solo el 30.25% de la variación en presión arterial
# → El 69.75% restante: genética, dieta, ejercicio, estrés, medicación, peso

# Insight clínico:
# → La edad es un factor importante pero no dominante
# → No puedes predecir presión arterial solo con la edad, necesitas más variables
# → R² de 0.30 es aceptable en medicina dado el número de factores involucrados
```

---

## <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzMxY200dzhyc2RhbG5zbnk3MWZoZHE2ODZnM3I3Y2lseHNtYXI4diZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/WmvhTzEl9ihy9TqPme/giphy.gif" width="40"> Analogía con Medicina Tradicional China

La correlación y R² son como el diagnóstico de la relación entre síntomas en MTC:

| Estadística | MTC |
|---|---|
| r alto positivo | Síntomas que siempre aparecen juntos → mismo síndrome |
| r alto negativo | Síntomas con relación Yin-Yang → uno sube cuando el otro baja |
| r ≈ 0 | Síntomas independientes → diferentes sistemas |
| R² = 0.80 | Un síntoma explica el 80% de la presencia del otro |
| R² = 0.20 | Hay relación pero hay muchos otros factores |
| Correlación ≠ causalidad | Un síntoma no causa al otro → ambos son manifestaciones del mismo desequilibrio |
| R² ajustada | Diagnóstico que considera solo los factores relevantes sin sobrediagnosticar |

> En MTC, cuando la fiebre y la sed correlacionan r = 0.85 en 1,000 pacientes, el R² = 0.72 te dice que el 72%
> de la variación en la intensidad de la sed es explicado por la intensidad de la fiebre.
> El 28% restante viene de otros factores: clima, constitución del paciente, medicación, hidratación previa.

---

## ⚡ Cheat Sheet rápido

```
CORRELACIÓN (r):
→ Mide fuerza y dirección de relación lineal
→ Va de -1 a +1
→ |r| > 0.7 = fuerte  |  0.4-0.7 = moderada  |  < 0.4 = débil
→ Positiva = ambas suben juntas
→ Negativa = cuando una sube la otra baja
→ SIEMPRE grafica el scatter antes de interpretar r

R² (coeficiente de determinación):
→ R² = r²
→ Va de 0 a 1 (o 0% a 100%)
→ Interpretación: "X explica el R²% de la variación en Y"
→ Lo que falta hasta 100% = otros factores no considerados
→ No hay un R² "bueno" universal — depende del contexto

RELACIÓN ENTRE LOS DOS:
→ r = 0.50  →  R² = 0.25 (X explica solo 25%)
→ r = 0.70  →  R² = 0.49 (X explica casi la mitad)
→ r = 0.90  →  R² = 0.81 (X explica la mayoría)
→ Necesitas r > 0.70 para tener R² > 0.50

ADVERTENCIAS:
→ Correlación ≠ Causalidad (SIEMPRE)
→ r mide solo relaciones LINEALES
→ Anscombe: misma r, distribuciones completamente distintas
→ Con muchas variables: usa R² ajustada
→ R² alto ≠ modelo correcto
→ R² bajo ≠ sin relación (puede ser no lineal)

TIPOS DE CORRELACIÓN:
→ Pearson  = datos continuos, relación lineal ← el más común
→ Spearman = datos ordinales o con outliers
→ Kendall  = muestras pequeñas
```

---

## 🔗 Aplicado a mis proyectos

### Proyecto EDA: Ventas Retail

```python
# Correlaciones que encontré y su R²:

correlaciones_eda = {
    'TOTAL_VALUE_SALES ↔ TOTAL_UNIT_SALES': {
        'r': 0.92,
        'r2': 0.92**2,
        'interpretacion': 'Las unidades explican el 84.6% del ingreso'
    },
    'VAR_PCT ↔ ABOVE_AVG': {
        'r': 0.83,
        'r2': 0.83**2,
        'interpretacion': 'Variables casi redundantes: 68.9% compartido'
    },
    'VAR_WEEKLY_AVG ↔ TOTAL_UNIT_AVG': {
        'r': -0.78,
        'r2': 0.78**2,
        'interpretacion': 'Relación inversa: grandes compradores son más consistentes (60.8%)'
    },
    'PRICE ↔ HIGH_VALUE': {
        'r': 0.68,
        'r2': 0.68**2,
        'interpretacion': 'Precio explica el 46.2% de ser clasificado como alto valor'
    }
}

for par, datos in correlaciones_eda.items():
    print(f"\n{par}")
    print(f"  r  = {datos['r']:.2f}")
    print(f"  R² = {datos['r2']:.3f} ({datos['r2']*100:.1f}%)")
    print(f"  → {datos['interpretacion']}")
```

### Posibles proyectos futuros
| Dataset | Correlación esperada | R² esperada | Por qué |
|---|---|---|---|
| SEMARNAT | PIB ↔ Emisiones CO₂ | r≈0.75, R²≈0.56 | Más industria = más emisiones |
| OMS | PIB per cápita ↔ Esperanza de vida | r≈0.80, R²≈0.64 | Desarrollo → salud |
| INEGI | Educación ↔ Ingreso | r≈0.65, R²≈0.42 | Relación real pero con mucho ruido |
| Kaggle fraude | Monto ↔ Es fraude | r≈0.30, R²≈0.09 | Fraude no depende solo del monto |

---

*Nota creada: 2026 · Serie: Estadística para Ciencia de Datos*
*Parte del certificado Científico de Datos — EBAC*
*🪷 github.com/ReginaPema*
