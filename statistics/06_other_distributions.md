# Other Distributions
## Otras Distribuciones — Cuando tus datos no son una campana

> **En una frase:**
> La distribución normal es la más famosa, pero en datos reales es la excepción, no la regla.
> Aprender a reconocer qué distribución siguen tus datos es lo que separa un análisis correcto
> de uno que parece correcto pero está mal.

> **Regla de oro:**
> Antes de aplicar cualquier modelo o prueba, pregúntate: ¿qué distribución tienen mis datos?
> La respuesta determina qué herramientas puedes usar.

---

## 🗺️ Mapa de distribuciones: cuál aplica cuándo

```
¿Qué tipo de datos tienes?
│
├── Continuos (valores en cualquier punto de un rango)
│   ├── Simétricos, fenómenos naturales   → Normal
│   ├── Sesgados positivos (ventas, precios,
│   │   ingresos, tiempos)                → Log-Normal
│   ├── Tiempo entre eventos              → Exponencial
│   └── Cualquier valor en un rango fijo  → Uniforme
│
├── Discretos (valores enteros, conteos)
│   ├── Éxito/fracaso en n intentos       → Binomial
│   ├── Número de eventos en un período   → Poisson
│   └── Intentos hasta el primer éxito    → Geométrica
│
└── Especiales
    ├── Conteos con mucha variabilidad    → Binomial Negativa
    ├── Datos de proporciones (0 a 1)     → Beta
    └── Varianzas y pruebas de hipótesis  → Chi-cuadrada
```

---

## 📦 1. Log-Normal: La distribución de mis datos de ventas

### Qué es

Una variable sigue una distribución log-normal cuando su **logaritmo** sigue una distribución normal.
Dicho de otra forma: si tomas log(X) y obtienes una campana, X es log-normal.

```
Variable original (sesgada):    Log de la variable (normal):

█                                          ████
██                                       ████████
███                                     ██████████
████░░░░░░                               ████████
                                           ████
```

### Características clave

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ── GENERAR DATOS LOG-NORMALES ──
np.random.seed(42)
datos_lognormal = np.random.lognormal(mean=3, sigma=1.5, size=10000)

# ── PROPIEDADES ──
media   = datos_lognormal.mean()
mediana = np.median(datos_lognormal)
moda    = np.exp(3 - 1.5**2)   # moda = e^(μ - σ²)

print(f"Media:   {media:,.2f}")
print(f"Mediana: {mediana:,.2f}")
print(f"Moda:    {moda:,.2f}")
print(f"Media > Mediana > Moda: sesgo positivo extremo")

# ── VERIFICAR SI TUS DATOS SON LOG-NORMALES ──
def verificar_lognormal(datos, nombre='Variable'):
    log_datos = np.log(datos[datos > 0])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Original: sesgado
    axes[0].hist(datos, bins=50, color='steelblue', edgecolor='white', alpha=0.8, density=True)
    axes[0].set_title(f'Distribución original\n{nombre}')
    axes[0].set_xlabel('Valor')

    # Log-transformado: debería ser campana
    axes[1].hist(log_datos, bins=50, color='teal', edgecolor='white', alpha=0.8, density=True)
    x_range = np.linspace(log_datos.min(), log_datos.max(), 200)
    axes[1].plot(x_range, stats.norm.pdf(x_range, log_datos.mean(), log_datos.std()),
                 color='red', linewidth=2.5, label='Normal teórica')
    axes[1].set_title(f'Log({nombre})\n¿se parece a una campana?')
    axes[1].legend()

    # Q-Q plot del log
    stats.probplot(log_datos, dist='norm', plot=axes[2])
    axes[2].get_lines()[1].set_color('red')
    axes[2].set_title('Q-Q del log\n✓ = puntos sobre la línea')

    # Prueba formal
    _, p = stats.normaltest(log_datos)
    print(f"\n{nombre}:")
    print(f"  p-value de normalidad en log: {p:.4f}")
    print(f"  {'✓ Log-normal' if p > 0.05 else '✗ No log-normal'}")

    plt.suptitle(f'Verificación Log-Normal — {nombre}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'images/dist_lognormal_{nombre.lower()}.png', dpi=150)
    plt.show()

# Aplicar a tus datos de ventas
verificar_lognormal(df['ventas_total'].dropna(), 'Ventas Retail')
```

### Cuándo aparece en la práctica

| Área | Variable | Por qué es log-normal |
|---|---|---|
| **Retail** | Ventas por transacción | Pocas transacciones enormes + muchas pequeñas |
| **Economía** | Ingresos y salarios | Desigualdad → el ingreso no puede ser negativo |
| **Inmobiliario** | Precios de vivienda | Concentración en rangos medios, cola alta |
| **Ambiente** | Emisiones de CO₂ | Pocas industrias gigantes + muchas pequeñas |
| **Salud** | Tiempo de recuperación | Mayoría rápido, algunos muy largo |
| **Fraude** | Monto de transacciones fraudulentas | Mayoría moderadas, algunas enormes |
| **Digital** | Tiempo en app por sesión | Mayoría sesiones cortas, pocas muy largas |

### Cómo trabajar con datos log-normales

```python
# ── TRANSFORMAR PARA ANÁLISIS ──

# 1. Usar escala log en gráficas
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['ventas_total'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_title('Escala normal — ilegible')

axes[1].hist(df['ventas_total'], bins=50, color='teal', edgecolor='white', alpha=0.8, log=True)
axes[1].set_title('Escala log: informativa ✓')

plt.tight_layout()
plt.show()

# 2. Transformar antes de ML que asume normalidad
df['log_ventas'] = np.log1p(df['ventas_total'])
# log1p = log(1 + x) → maneja ceros sin error

# 3. Usar mediana en lugar de media
# La media es engañada por los valores extremos
print(f"Media:   ${df['ventas_total'].mean():,.2f}  ← sobreestima")
print(f"Mediana: ${df['ventas_total'].median():,.2f} ← más representativa")

# 4. Usar Spearman en lugar de Pearson para correlaciones
from scipy import stats
r_pearson, _ = stats.pearsonr(
    df['ventas_total'], df['precio_unitario'])
r_spearman, _ = stats.spearmanr(
    df['ventas_total'], df['precio_unitario'])
print(f"\nPearson: {r_pearson:.3f} (sensible a outliers)")
print(f"Spearman: {r_spearman:.3f} (más robusto ✓)")
```

---

## 📦 2. Binomial: Éxito o fracaso en n intentos

### Qué es

Modela el número de éxitos en **n intentos independientes**, donde cada intento tiene probabilidad **p** de éxito.

```
¿Cuántas de 10 transacciones son fraude si la probabilidad de fraude es 2%?

P(X = k) = C(n,k) × p^k × (1-p)^(n-k)

n = 10  (intentos)
p = 0.02 (probabilidad de fraude)
k = 0, 1, 2, ... (número de fraudes)
```

### Características

```python
from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np

# ── CALCULAR PROBABILIDADES ──
n = 100    # transacciones analizadas
p = 0.02   # 2% probabilidad de fraude

# P(exactamente 3 fraudes)
prob_3 = binom.pmf(3, n, p)
print(f"P(exactamente 3 fraudes) = {prob_3:.4f} ({prob_3*100:.2f}%)")

# P(3 o menos fraudes)
prob_hasta_3 = binom.cdf(3, n, p)
print(f"P(3 o menos fraudes) = {prob_hasta_3:.4f} ({prob_hasta_3*100:.2f}%)")

# P(más de 5 fraudes) — alerta
prob_mas_5 = 1 - binom.cdf(5, n, p)
print(f"P(más de 5 fraudes) = {prob_mas_5:.4f} ← ✇ raro")

# ── VISUALIZAR ──
fig, ax = plt.subplots(figsize=(12, 6))

k_valores = np.arange(0, 15)
probabilidades = binom.pmf(k_valores, n, p)

barras = ax.bar(k_valores, probabilidades, color='steelblue', edgecolor='white', alpha=0.8)

# Destacar zona de alerta (> 5)
for i, barra in enumerate(barras):
    if k_valores[i] > 5:
        barra.set_color('#E63946')
        barra.set_alpha(0.9)

ax.set_title(f'Distribución Binomial\n'
             f'n={n} transacciones, p={p} fraude\n'
             f'Rojo = zona de alerta (>5 fraudes)', fontsize=12)
ax.set_xlabel('Número de fraudes detectados')
ax.set_ylabel('Probabilidad')
ax.set_xticks(k_valores)
plt.tight_layout()
plt.savefig('images/dist_binomial.png', dpi=150)
plt.show()

# ── ESTADÍSTICOS ──
media_b = n * p
varianza_b = n * p * (1 - p)
std_b = np.sqrt(varianza_b)

print(f"\nMedia (n×p):   {media_b:.2f} fraudes esperados")
print(f"Std:           {std_b:.2f}")
print(f"Rango normal:  {media_b-2*std_b:.1f} a {media_b+2*std_b:.1f}")
```

### Cuándo aparece

| Área | Ejemplo | n | p |
|---|---|---|---|
| **Fraude** | ¿Cuántas de n transacciones son fraude? | n trans | p_fraude |
| **Salud** | ¿Cuántos pacientes responden al tratamiento? | n pacientes | p_respuesta |
| **Marketing** | ¿Cuántos emails generan clic? | n emails | p_clic |
| **Ambiente** | ¿Cuántas empresas cumplen la norma? | n empresas | p_cumplimiento |
| **Calidad** | ¿Cuántos productos salen defectuosos? | n productos | p_defecto |

### Relación con la Normal

```python
# Con n grande y p no extrema,
# Binomial ≈ Normal (Teorema del Límite Central)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
configs = [(10, 0.3, 'n=10, p=0.3\nAsimétrica'),
           (50, 0.3, 'n=50, p=0.3\nMás simétrica'),
           (200, 0.3, 'n=200, p=0.3\nCasi Normal')]

for ax, (n_val, p_val, titulo) in zip(axes, configs):
    k = np.arange(0, n_val + 1)
    probs = binom.pmf(k, n_val, p_val)
    ax.bar(k, probs, color='steelblue', edgecolor='white', alpha=0.8)

    # Curva normal equivalente
    mu = n_val * p_val
    sig = np.sqrt(n_val * p_val * (1 - p_val))
    x = np.linspace(mu - 4*sig, mu + 4*sig, 200)
    from scipy.stats import norm
    ax.plot(x, norm.pdf(x, mu, sig), color='red', linewidth=2)
    ax.set_title(titulo)

plt.suptitle('Binomial → Normal con n grande (TLC)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('images/dist_binomial_a_normal.png', dpi=150)
plt.show()
```

---

## 📦 3. Poisson: Conteos de eventos en un período

### Qué es

Modela el **número de eventos** que ocurren en un período de tiempo o espacio fijo,
cuando los eventos son independientes y ocurren a una tasa constante λ (lambda).

```
P(X = k) = (λ^k × e^(-λ)) / k!

λ = tasa promedio de eventos por período
k = número de eventos que queremos calcular
```

### Características clave

```python
from scipy.stats import poisson
import matplotlib.pyplot as plt
import numpy as np

# ── EJEMPLOS PRÁCTICOS ──

# Escenario 1: Fraudes por hora en plataforma eBay
lambda_fraudes = 3   # promedio 3 intentos de fraude/hora

print("=== Fraudes por hora (λ=3) ===")
for k in range(8):
    p = poisson.pmf(k, lambda_fraudes)
    print(f"P({k} fraudes/hora) = {p:.4f} ({p*100:.2f}%)")

# Probabilidad de más de 7 (alerta de ataque)
p_alerta = 1 - poisson.cdf(7, lambda_fraudes)
print(f"P(>7 fraudes) = {p_alerta:.6f} ← ✇ ataque coordinado")

# Escenario 2: Consultas médicas por día
lambda_consultas = 25   # promedio 25 consultas/día

# ¿Qué tan probable es tener más de 35?
p_saturacion = 1 - poisson.cdf(35, lambda_consultas)
print(f"\nP(>35 consultas) = {p_saturacion:.4f}")
print(f"→ Sistema saturado {p_saturacion*100:.2f}% de los días")

# ── VISUALIZAR DISTINTOS LAMBDAS ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
lambdas = [(1, 'λ=1\n(eventos raros)'),
           (5, 'λ=5\n(eventos moderados)'),
           (20, 'λ=20\n(muchos eventos → normal)')]

for ax, (lam, titulo) in zip(axes, lambdas):
    k = np.arange(0, int(lam * 3) + 1)
    probs = poisson.pmf(k, lam)
    ax.bar(k, probs, color='teal', edgecolor='white', alpha=0.8)
    ax.axvline(lam, color='red', linestyle='--', linewidth=2, label=f'λ={lam}')
    ax.set_title(titulo)
    ax.set_xlabel('Número de eventos (k)')
    ax.set_ylabel('Probabilidad')
    ax.legend()

plt.suptitle('Distribución Poisson — Distintos valores de λ', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('images/dist_poisson.png', dpi=150)
plt.show()

# ── PROPIEDADES ESPECIALES ──
# En Poisson: media = varianza = λ
# Si varianza >> media → usar Binomial Negativa
lam = 5
datos_poisson = np.random.poisson(lam, 10000)
print(f"\nMedia:    {datos_poisson.mean():.3f}")
print(f"Varianza: {datos_poisson.var():.3f}")
print(f"λ:        {lam}")
print("✓ Media ≈ Varianza ≈ λ")
```

### Cuándo aparece

| Área | Variable | λ típico |
|---|---|---|
| **Fraude** | Intentos de acceso no autorizado por hora | 2-10/hora |
| **Call center** | Llamadas entrantes por minuto | 5-20/min |
| **Salud** | Casos de enfermedad por semana | Var. |
| **Ambiente** | Derrames industriales por año | 0.5-5/año |
| **Retail** | Pedidos por minuto en hora pico | 10-50/min |
| **Software** | Bugs reportados por día | 3-15/día |

### Diferencia con Binomial

```
Binomial: n intentos FINITOS, p probabilidad
→ "De 100 transacciones, ¿cuántas son fraude?"

Poisson: tiempo/espacio CONTINUO, λ tasa
→ "¿Cuántos fraudes ocurren en 1 hora?"

Cuando n→∞ y p→0 manteniendo λ=np constante:
Binomial(n,p) ≈ Poisson(λ)
```

---

## 📦 4. Exponencial: Tiempo ENTRE eventos

### Qué es

Modela el **tiempo de espera hasta el próximo evento** cuando los eventos siguen una distribución Poisson.
Si los fraudes ocurren a tasa λ (Poisson), el tiempo entre fraudes sigue una Exponencial.

```
F(x) = 1 - e^(-λx)     (distribución acumulada)
f(x) = λ × e^(-λx)     (densidad de probabilidad)

λ = tasa de eventos (mismo que Poisson)
x = tiempo hasta el próximo evento
```

### Características y código

```python
from scipy.stats import expon
import matplotlib.pyplot as plt
import numpy as np

# ── ESCENARIO: tiempo entre fraudes ──
lambda_fraudes = 3   # 3 fraudes por hora
escala = 1 / lambda_fraudes  # tiempo promedio entre fraudes

# 1/λ = tiempo promedio entre eventos
tiempo_promedio = 1 / lambda_fraudes
print(f"λ = {lambda_fraudes} fraudes/hora")
print(f"Tiempo promedio entre fraudes: {tiempo_promedio*60:.1f} minutos")

# Probabilidades
# P(próximo fraude en menos de 10 min)
p_10min = expon.cdf(10/60, scale=escala)
print(f"\nP(fraude en < 10 min): {p_10min:.4f} ({p_10min*100:.1f}%)")

# P(más de 30 min sin fraude)
p_30min = 1 - expon.cdf(30/60, scale=escala)
print(f"P(> 30 min sin fraude): {p_30min:.4f} ({p_30min*100:.1f}%)")

# ── SIN MEMORIA — propiedad clave ──
# La exponencial no tiene memoria:
# P(X > s+t | X > s) = P(X > t)
# "Haber esperado s minutos no aumenta la probabilidad de que llegue pronto"

# ── VISUALIZAR ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.linspace(0, 3, 500)

# Diferentes tasas
for lam, color, label in [(0.5, 'steelblue', 'λ=0.5 (eventos raros)'),
                            (1, 'teal', 'λ=1'),
                            (3, 'coral', 'λ=3 (eventos frecuentes)')]:
    y = expon.pdf(x, scale=1/lam)
    axes[0].plot(x, y, color=color, linewidth=2, label=label)

axes[0].set_title('PDF Exponencial — distintas tasas')
axes[0].set_xlabel('Tiempo hasta el próximo evento')
axes[0].set_ylabel('Densidad')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Distribución acumulada
for lam, color, label in [(0.5, 'steelblue', 'λ=0.5'),
                            (1, 'teal', 'λ=1'),
                            (3, 'coral', 'λ=3')]:
    y_cdf = expon.cdf(x, scale=1/lam)
    axes[1].plot(x, y_cdf, color=color, linewidth=2, label=label)

axes[1].set_title('CDF Exponencial — P(X ≤ x)')
axes[1].set_xlabel('Tiempo')
axes[1].set_ylabel('Probabilidad acumulada')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50%')

plt.suptitle('Distribución Exponencial', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('images/dist_exponencial.png', dpi=150)
plt.show()
```

### Cuándo aparece

| Área | Variable |
|---|---|
| **Fraude** | Tiempo entre intentos de fraude |
| **Salud** | Tiempo entre llegadas a urgencias |
| **Retail** | Tiempo entre compras del mismo cliente |
| **Tecnología** | Tiempo entre fallas de sistema |
| **Call center** | Tiempo entre llamadas entrantes |

---

## 📦 5. Uniforme: Todos los valores igualmente probables

### Qué es

Todos los valores en el rango [a, b] tienen exactamente la misma probabilidad.
Sin picos, sin sesgo, completamente plana.

```python
from scipy.stats import uniform
import matplotlib.pyplot as plt
import numpy as np

# ── DISTRIBUCIÓN UNIFORME ──
a, b = 0, 100   # rango

# Probabilidad de cualquier valor específico = 0
# Probabilidad de un rango = (b2 - b1) / (b - a)

# P(valor entre 20 y 40)
p_rango = (40 - 20) / (b - a)
print(f"P(20 ≤ X ≤ 40) = {p_rango:.2f} = 20%")

# ── VISUALIZAR ──
fig, ax = plt.subplots(figsize=(10, 5))
x = np.linspace(-10, 110, 500)
y = uniform.pdf(x, loc=a, scale=b-a)
ax.plot(x, y, color='steelblue', linewidth=3)
ax.fill_between(x, y, where=(x >= a) & (x <= b), alpha=0.3, color='steelblue')
ax.set_title('Distribución Uniforme U(0, 100)')
ax.set_xlabel('Valor')
ax.set_ylabel('Densidad (1/100 = 0.01)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/dist_uniforme.png', dpi=150)
plt.show()
```

### Cuándo aparece en Ciencia de Datos

```python
# USO PRINCIPAL: generación de números aleatorios y muestreo, no tanto para modelar datos reales

# 1. Dividir datos en train/test aleatoriamente from sklearn.model_selection import train_test_split
# → Internamente usa distribución uniforme para seleccionar

# 2. Inicializar pesos en redes neuronales
pesos_iniciales = np.random.uniform(-0.1, 0.1, size=(100, 50))

# 3. Generar datos sintéticos para pruebas
datos_prueba = np.random.uniform(low=0, high=1000, size=1000)

# 4. Detectar si tus datos SON uniformes (señal de posible error en datos o variable categórica)
_, p_uniform = stats.kstest(datos_prueba, 'uniform', args=(0, 1000))
print(f"¿Datos uniformes? p={p_uniform:.4f}", "✓" if p_uniform > 0.05 else "✗")
```

---

## 📦 6. Binomial Negativa: Conteos con alta variabilidad

### Qué es

Una extensión de la Poisson para cuando la varianza es **mucho mayor** que la media (sobredispersión). 
Muy común en datos reales de salud, biología y comportamiento humano.

```
Poisson: media = varianza (ideal)
Binomial Negativa: varianza > media (realista)
```

```python
from scipy.stats import nbinom
import matplotlib.pyplot as plt
import numpy as np

# ── CUÁNDO USARLA ──
# Señal de alerta: si varianza >> media en tus conteos

def elegir_distribucion_conteos(datos):
    media = datos.mean()
    varianza = datos.var()
    ratio = varianza / media

    print(f"Media:    {media:.3f}")
    print(f"Varianza: {varianza:.3f}")
    print(f"Ratio V/M: {ratio:.3f}")

    if ratio < 1.5:
        print("→ ✓ Poisson es apropiada")
    else:
        print(f"→ ✇ Sobredispersión (ratio={ratio:.1f})")
        print("   Usar Binomial Negativa")

# Ejemplo: consultas médicas por paciente por año
consultas_normal = np.random.poisson(3, 1000)
consultas_variables = np.random.negative_binomial(n=2, p=0.4, size=1000)  # alta variabilidad

print("=== Consultas estándar (Poisson) ===")
elegir_distribucion_conteos(consultas_normal)

print("\n=== Consultas con alta variabilidad (Neg.Bin.) ===")
elegir_distribucion_conteos(consultas_variables)
```

### Cuándo aparece

| Área | Variable | Por qué hay sobredispersión |
|---|---|---|
| **Salud** | Consultas por paciente | Enfermos crónicos vs sanos |
| **Retail** | Compras por cliente | Compradores frecuentes vs ocasionales |
| **Software** | Bugs por módulo | Algunos módulos muy complejos |
| **Digital** | Posts por usuario | Influencers vs usuarios normales |
| **Ambiente** | Especies por zona | Zonas con alta vs baja biodiversidad |

---

## 📦 7. Beta: Datos que son proporciones

### Qué es

Modela datos que son **proporciones o probabilidades**, valores entre 0 y 1. 
Es la distribución natural para tasas, porcentajes y probabilidades.

```python
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np

# ── VISUALIZAR DIFERENTES FORMAS ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

configs = [
    (1, 1, 'Beta(1,1)\nUniforme — sin info previa'),
    (2, 5, 'Beta(2,5)\nSesgo hacia valores bajos'),
    (5, 2, 'Beta(5,2)\nSesgo hacia valores altos'),
    (5, 5, 'Beta(5,5)\nSimétrica, centrada en 0.5'),
    (0.5, 0.5, 'Beta(0.5,0.5)\nBimodal — extremos'),
    (10, 3, 'Beta(10,3)\nMuy concentrada alta'),
]

x = np.linspace(0.001, 0.999, 500)
colores = ['steelblue', 'teal', 'coral', 'gold', 'purple', 'green']

for ax, (a, b, titulo), color in zip(axes.flatten(), configs, colores):
    y = beta.pdf(x, a, b)
    ax.plot(x, y, color=color, linewidth=2.5)
    ax.fill_between(x, y, alpha=0.2, color=color)
    ax.set_title(titulo, fontsize=9)
    ax.set_xlabel('Proporción (0 a 1)')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

plt.suptitle('Distribución Beta — Flexibilidad de formas', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('images/dist_beta.png', dpi=150)
plt.show()
```

### Cuándo aparece

| Variable | Ejemplo | α, β típicos |
|---|---|---|
| Tasa de conversión | % clics que compran | Beta(ventas+1, no_ventas+1) |
| Tasa de fraude | % transacciones fraudulentas | Beta(fraudes+1, normales+1) |
| Tasa de respuesta | % pacientes que mejoran | Beta(mejoran+1, no_mejoran+1) |
| Participación de mercado | % del mercado | Beta(ventas_marca, ventas_total) |
| Precisión de modelo | Accuracy de clasificador | Beta(correctas+1, incorrectas+1) |

---

## 📦 8. Chi-Cuadrada: Para pruebas de hipótesis

### Qué es

No modela datos directamente, es una distribución derivada que se usa para **pruebas estadísticas**, especialmente:
- Prueba de bondad de ajuste
- Prueba de independencia entre variables
- Prueba de homogeneidad

```python
from scipy.stats import chi2, chi2_contingency
import pandas as pd
import numpy as np

# ── PRUEBA DE INDEPENDENCIA ──
# ¿Hay relación entre región y segmento?
# H0: región y segmento son INDEPENDIENTES
# H1: hay relación entre región y segmento

tabla_contingencia = pd.crosstab(df['region'], df['segmento'])

chi2_stat, p_value, dof, expected = chi2_contingency(tabla_contingencia)

print(f"Chi² estadístico: {chi2_stat:.4f}")
print(f"Grados de libertad: {dof}")
print(f"p-value: {p_value:.6f}")

if p_value < 0.05:
    print("✓ Hay relación estadísticamente significativa entre región y segmento")
else:
    print("✇ No hay evidencia de relación entre región y segmento")

# ── PRUEBA DE BONDAD DE AJUSTE ──
# ¿Las ventas por segmento son uniformes?
# H0: ventas distribuidas uniformemente entre segmentos

ventas_obs = df.groupby('segmento')['ventas_total'].count()
esperado_uniforme = np.full(len(ventas_obs), ventas_obs.mean())

chi2_stat, p_value = stats.chisquare(ventas_obs, f_exp=esperado_uniforme)

print(f"\nBondad de ajuste — ¿distribución uniforme?")
print(f"Chi² = {chi2_stat:.2f}, p = {p_value:.6f}")
print("✗ No es uniforme, hay segmentos dominantes"
      if p_value < 0.05 else "✓ Distribución uniforme")
```

---

## 🗺️ Guía de selección rápida

```python
def seleccionar_distribucion(datos, tipo='continuo'):
    """
    Guía para elegir la distribución correcta.
    """
    if tipo == 'continuo':
        media = datos.mean()
        mediana = datos.median() if hasattr(datos, 'median') else np.median(datos)
        std = datos.std()
        minimo = datos.min()

        if minimo < 0:
            print("→ Normal (puede tener valores negativos)")

        elif abs(media - mediana) / std < 0.1:
            print("→ Normal (simétrica, media ≈ mediana)")

        elif media > mediana * 1.5:
            log_datos = np.log(datos[datos > 0])
            _, p_log = stats.normaltest(log_datos)
            if p_log > 0.05:
                print("→ Log-Normal (sesgo positivo extremo)")
            else:
                print("→ Pareto o distribución de cola pesada")

        elif 0 <= datos.min() and datos.max() <= 1:
            print("→ Beta (proporción entre 0 y 1)")

    elif tipo == 'discreto':
        media = datos.mean()
        varianza = datos.var()
        ratio = varianza / media

        if datos.nunique() == 2:
            print("→ Bernoulli (solo 0 y 1)")
        elif ratio < 1.5:
            print(f"→ Poisson (λ≈{media:.1f})")
        else:
            print(f"→ Binomial Negativa (sobredispersión {ratio:.1f}x)")

    elif tipo == 'tiempo':
        print("→ Exponencial (tiempo entre eventos)")
        print("   O Weibull si la tasa cambia con el tiempo")

# Usar en tu proyecto:
seleccionar_distribucion(df['ventas_total'], 'continuo')
seleccionar_distribucion(df['unidades_vendidas'], 'discreto')
```

---

## 📊 Resumen visual: todas juntas

```python
# Visualizar las 6 distribuciones principales
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

np.random.seed(42)

distribuciones = [
    ('Log-Normal\n(ventas, salarios)', 'continua', np.random.lognormal(0, 1, 5000), 'steelblue'),

    ('Binomial\n(éxito/fracaso)', 'discreta', np.random.binomial(50, 0.3, 5000), 'teal'),

    ('Poisson\n(conteos por período)', 'discreta', np.random.poisson(5, 5000), 'coral'),

    ('Exponencial\n(tiempo entre eventos)', 'continua', np.random.exponential(2, 5000), 'gold'),

    ('Beta\n(proporciones 0-1)', 'continua', np.random.beta(3, 7, 5000), 'purple'),

    ('Binomial Negativa\n(conteos sobredispersos)', 'discreta', np.random.negative_binomial(3, 0.5, 5000), 'green'),
]

for ax, (nombre, tipo, datos, color) in zip(axes.flatten(), distribuciones):
    ax.hist(datos, bins=40, color=color, edgecolor='white', alpha=0.8, density=True)
    ax.set_title(nombre, fontsize=10, fontweight='bold')
    ax.set_xlabel('Valor')
    ax.set_ylabel('Densidad')
    media = np.mean(datos)
    ax.axvline(media, color='red', linestyle='--', linewidth=1.5, label=f'Media={media:.2f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Las 6 Distribuciones Más Importantes\n'
             'en Ciencia de Datos', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/todas_distribuciones.png', dpi=150)
plt.show()
```

---

## <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzMxY200dzhyc2RhbG5zbnk3MWZoZHE2ODZnM3I3Y2lseHNtYXI4diZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/WmvhTzEl9ihy9TqPme/giphy.gif" width="40"> Analogía con Medicina Tradicional China

Cada distribución corresponde a un patrón de presentación diferente en MTC:

| Distribución | Analogía MTC |
|---|---|
| **Normal** | Patrón bien equilibrado → el estado ideal de salud |
| **Log-Normal** | Exceso crónico → la mayoría tiene síntomas leves, pocos tienen exceso extremo |
| **Binomial** | Diagnóstico dicotómico → ¿hay desequilibrio de Yin-Yang o no? |
| **Poisson** | Frecuencia de episodios agudos por mes → cuántas crisis |
| **Exponencial** | Tiempo entre recaídas → cuánto tarda en volver el síntoma |
| **Beta** | Proporción de pacientes que responden → efectividad del tratamiento |
| **Binomial Negativa** | Alta variabilidad en respuesta → algunos pacientes muy diferentes a otros |

---

## ⚡ Cheat Sheet rápido

```
CONTINUAS:
→ Normal            = campana simétrica, fenómenos naturales
→ Log-Normal        = sesgo positivo extremo: ventas, salarios → transformar con log() antes de ML
→ Exponencial       = tiempo ENTRE eventos Poisson
→ Uniforme          = todos los valores igualmente probables
→ Beta              = proporciones y probabilidades (0 a 1)

DISCRETAS:
→ Binomial          = éxito/fracaso en n intentos finitos
→ Poisson           = conteos en período, media ≈ varianza
→ Binomial Negativa = conteos con varianza >> media

SEÑALES EN TUS DATOS:
→ media >> mediana            → Log-Normal o Pareto
→ media ≈ varianza (conteos)  → Poisson
→ varianza >> media (conteos) → Binomial Negativa
→ datos entre 0 y 1           → Beta
→ datos binarios (0/1)        → Bernoulli/Binomial
→ tiempo hasta evento         → Exponencial

TRANSFORMACIONES:
→ Log-Normal → log(x) o log1p(x) para normalizar
→ Beta       → logit(p) = log(p/(1-p)) para normalizar
→ Poisson    → sqrt(x) para estabilizar varianza

PRUEBAS ESTADÍSTICAS:
→ Chi-cuadrada → independencia entre categóricas
→ t-test       → comparar medias (asume normalidad)
→ Mann-Whitney → comparar medianas (no paramétrico)
→ KS test      → verificar si sigue una distribución
```

---

## 🔗 Aplicado a mis proyectos

### Proyecto EDA: Ventas Retail

```python
# Distribuciones identificadas:

# ventas_total → LOG-NORMAL
# → Media $90.51 >> Mediana $16.81
# → log(ventas) sigue distribución normal
# → Por eso los gráficos necesitaban escala log
# → Para ML: transformar con log1p antes de entrenar

# unidades_vendidas → LOG-NORMAL o POISSON
# → Verificar ratio varianza/media
ratio = df['unidades_vendidas'].var() / df['unidades_vendidas'].mean()
print(f"Ratio V/M unidades: {ratio:.2f}")
# Si ratio >> 1 → Binomial Negativa
# Si ratio ≈ 1 → Poisson

# por_region (conteo transacciones) → POISSON
# → ~17,000 transacciones por región
# → Si ratio V/M ≈ 1 → Poisson apropiada

# fraude (binario sí/no) → BINOMIAL
# → Cada transacción = intento
# → p = tasa de fraude observada
```

### Posibles proyectos futuros

| Dataset | Variable | Distribución esperada |
|---|---|---|
| SEMARNAT | Emisiones CO₂ por empresa | Log-Normal |
| OMS | Esperanza de vida | Normal o sesgada negativa |
| INEGI | Ingresos por hogar | Log-Normal |
| Kaggle fraude | Fraude sí/no | Binomial |
| Kaggle fraude | Monto transacción | Log-Normal |
| IMSS | Consultas por paciente/año | Poisson o Binomial Negativa |
| Dataset clínico | Tasa de respuesta tratamiento | Beta |

---

*Nota creada: 2026 · Serie: Estadística para Ciencia de Datos*
*Parte del certificado Científico de Datos — EBAC*
*🪷 github.com/ReginaPema*
