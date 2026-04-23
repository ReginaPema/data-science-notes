# 📊 03 · Scatter Plot
## Gráfica de Dispersión: relación entre dos variables numéricas

> El scatter plot es la única gráfica que te muestra directamente si dos variables están relacionadas,
> cómo se relacionan, y qué tan fuerte es esa relación. Es la gráfica de la correlación hecha visible.

> **Regla de oro:**
> Antes de calcular cualquier correlación numérica, grafica el scatter primero.
> Los números pueden mentirte, la gráfica no.

---

## 🧱 Anatomía de un scatter plot

```
Eje Y │                        •
(var  │                   • •
 2)   │              • • •  •
      │         • • • • •
      │    • • • • •
      │• • • •
      └──────────────────────────→ Eje X (variable 1)

Cada punto = una observación / fila del dataset
Posición X = valor de la variable 1
Posición Y = valor de la variable 2
Patrón de puntos = tipo de relación entre ambas
```

| Elemento | Qué representa |
|---|---|
| Cada punto | Una fila de tu dataset (una transacción, un paciente, un país) |
| Posición en X | El valor de la primera variable |
| Posición en Y | El valor de la segunda variable |
| Color del punto | Una tercera variable categórica (opcional) |
| Tamaño del punto | Una cuarta variable numérica (opcional) |
| Patrón general | El tipo de relación entre las variables |
| Línea de tendencia | La dirección promedio de la relación |

---

## 🐍 Cómo hacerlo en Python

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ── SCATTER BÁSICO CON MATPLOTLIB ──
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(df['precio_unitario'],
           df['unidades_vendidas'],
           alpha=0.4,          # transparencia, crucial con muchos puntos
           s=20,               # tamaño de cada punto
           color='steelblue',
           edgecolors='none')  # sin borde, más limpio

ax.set_title('Precio Unitario vs Unidades Vendidas', fontsize=14)
ax.set_xlabel('Precio Unitario ($)')
ax.set_ylabel('Unidades Vendidas')
plt.tight_layout()
plt.savefig('images/scatter_basico.png', dpi=150)
plt.show()

# ── CON LÍNEA DE TENDENCIA ──
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(df['precio_unitario'], df['unidades_vendidas'],
           alpha=0.3, s=15, color='steelblue', edgecolors='none')

# Línea de tendencia con numpy
z = np.polyfit(df['precio_unitario'],
               df['unidades_vendidas'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['precio_unitario'].min(),
                     df['precio_unitario'].max(), 100)
ax.plot(x_line, p(x_line),
        color='red', linewidth=2,
        label=f'Tendencia lineal')

# Correlación en el título
r = df['precio_unitario'].corr(df['unidades_vendidas'])
ax.set_title(f'Precio vs Unidades  |  r = {r:.3f}', fontsize=14)
ax.set_xlabel('Precio Unitario ($)')
ax.set_ylabel('Unidades Vendidas')
ax.legend()
plt.tight_layout()
plt.savefig('images/scatter_tendencia.png', dpi=150)
plt.show()

# ── CON SEABORN: regplot (scatter + regresión) ──
fig, ax = plt.subplots(figsize=(10, 6))

sns.regplot(data=df,
            x='precio_unitario',
            y='unidades_vendidas',
            scatter_kws={'alpha': 0.3, 's': 15,
                         'color': 'steelblue'},
            line_kws={'color': 'red', 'linewidth': 2},
            ax=ax)

ax.set_title('Precio vs Unidades (con banda de confianza)')
plt.tight_layout()
plt.show()

# ── CON TERCERA VARIABLE: color por categoría ──
fig, ax = plt.subplots(figsize=(12, 7))

sns.scatterplot(data=df,
                x='precio_unitario',
                y='unidades_vendidas',
                hue='segmento',      # color por segmento
                alpha=0.5,
                s=30,
                ax=ax)

ax.set_title('Precio vs Unidades por Segmento')
ax.set_xlabel('Precio Unitario ($)')
ax.set_ylabel('Unidades Vendidas')
ax.legend(title='Segmento', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('images/scatter_por_segmento.png', dpi=150)
plt.show()

# ── CON CUARTA VARIABLE — tamaño por magnitud ──
fig, ax = plt.subplots(figsize=(12, 7))

sns.scatterplot(data=df.sample(1000),  # muestra si hay muchos puntos
                x='precio_unitario',
                y='unidades_vendidas',
                hue='segmento',
                size='ventas_total',    # tamaño = magnitud de venta
                sizes=(20, 300),        # rango de tamaños
                alpha=0.6,
                ax=ax)

ax.set_title('Precio vs Unidades\n(color=segmento, tamaño=venta total)')
plt.tight_layout()
plt.show()

# ── ESCALA LOGARÍTMICA EN AMBOS EJES ──
# Útil cuando hay valores extremos en ambas variables
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Normal: difícil de leer
axes[0].scatter(df['precio_unitario'], df['ventas_total'],
                alpha=0.2, s=10, color='steelblue')
axes[0].set_title('Escala normal — aplastado')
axes[0].set_xlabel('Precio ($)')
axes[0].set_ylabel('Venta Total ($)')

# Log-log: mucho más informativo
axes[1].scatter(df['precio_unitario'], df['ventas_total'],
                alpha=0.2, s=10, color='teal')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_title('Escala log-log — patrón visible')
axes[1].set_xlabel('Precio ($) — escala log')
axes[1].set_ylabel('Venta Total ($) — escala log')

plt.tight_layout()
plt.savefig('images/scatter_log.png', dpi=150)
plt.show()

# ── MATRIZ DE SCATTER: todas vs todas ──
# Para explorar relaciones entre múltiples variables a la vez
numericas = df[['precio_unitario', 'unidades_vendidas',
                'ventas_total', 'precio_promedio']].sample(2000)

fig = sns.pairplot(numericas,
                   diag_kind='kde',    # histograma en diagonal
                   plot_kws={'alpha': 0.3, 's': 10},
                   height=2.5)
fig.figure.suptitle('Matriz de Scatter — Variables Numéricas',
                    y=1.02, fontsize=14)
plt.savefig('images/scatter_matrix.png', dpi=150,
            bbox_inches='tight')
plt.show()
```

---

## 🧭 Cómo leer un scatter plot: los 7 patrones

### Patrón 1: Correlación positiva fuerte
```
Y │              • •
  │          • • •
  │       • • •
  │    • • •
  │• • •
  └──────────────→ X
```
**Qué significa:**
Cuando X sube, Y también sube.
Los puntos forman una línea diagonal ascendente clara.

**Coeficiente:** r cercano a +1.0

**Ejemplos reales:**
- Unidades vendidas vs ingreso total *(r = 0.92 en mi proyecto EDA)*
- Altura vs peso en personas adultas
- Temperatura vs consumo de electricidad en verano
- Inversión en marketing vs ventas generadas

**Insight de negocio:**
Las variables se mueven juntas predeciblemente.
Puedes usar una para predecir la otra.
Base para modelos de regresión lineal.

---

### Patrón 2: Correlación negativa fuerte
```
Y │• •
  │    • •
  │        • •
  │            • •
  │                • •
  └──────────────────→ X
```
**Qué significa:**
Cuando X sube, Y baja.
Los puntos forman una línea diagonal descendente clara.

**Coeficiente:** r cercano a -1.0

**Ejemplos reales:**
- Dosis de medicamento vs intensidad del síntoma
- Horas de ejercicio semanal vs presión arterial
- Precio de producto vs demanda (elasticidad)
- Cobertura forestal vs temperatura promedio

**Insight de negocio:**
Relación inversa predecible.
Aumentar X tiene costo en Y, o reducir X tiene beneficio en Y.

---

### Patrón 3: Sin correlación (nube de puntos)
```
Y │  •  •    •
  │    •  • •  •
  │  •    •  •
  │    •  •    •
  │  •    •  •
  └──────────────→ X
```
**Qué significa:**
No hay relación lineal entre las variables.
Los puntos están distribuidos aleatoriamente.
Conocer X no te dice nada sobre Y.

**Coeficiente:** r cercano a 0

**Ejemplos reales:**
- Talla de zapatos vs calificación en matemáticas
- Edad vs color favorito
- Día del mes vs ventas (en negocios estables)

**⚠️ Importante:**
Sin correlación LINEAL no significa sin relación. Puede haber una relación no lineal 
que el scatter revela pero el coeficiente r no captura.

---

### Patrón 4: Correlación curvilínea
```
Y │         • •
  │      • •   • •
  │   • •         • •
  │• •               • •
  └──────────────────────→ X
```
**Qué significa:**
Hay una relación, pero no es lineal. La correlación de Pearson (r) puede ser
cercana a 0 aunque la relación sea muy fuerte. Esto es exactamente por qué siempre
debes graficar antes de calcular r.

**Ejemplos reales:**
- Dosis vs efecto (muy poca dosis: sin efecto, dosis óptima: máximo efecto, sobredosis: efecto negativo)
- Temperatura vs productividad (muy frío: baja, óptimo: alta, muy caliente: baja)
- Horas de sueño vs rendimiento cognitivo

**Insight:**
Necesitas modelos no lineales (regresión polinomial, árbol de decisión) para capturar esta relación.

---

### Patrón 5: Correlación con outliers que distorsionan
```
Y │
  │                              • ← outlier
  │
  │  • • • • • • • •
  │ • • • • • • •
  └──────────────────────────────→ X
```
**Qué significa:**
La nube principal no muestra correlación, pero uno o pocos outliers crean una
correlación artificial en el cálculo de r.
O viceversa: una correlación real es distorsionada por outliers extremos.

**Por qué es peligroso:**
Puedes concluir que hay correlación cuando no la hay, o que no hay cuando sí la hay.

```python
# Siempre compara con y sin outliers
r_completo = df['col1'].corr(df['col2'])
r_sin_outliers = (df[df['col2'] < limite_sup]
                  ['col1'].corr(
                   df[df['col2'] < limite_sup]['col2']))

print(f"r con outliers: {r_completo:.3f}")
print(f"r sin outliers: {r_sin_outliers:.3f}")
if abs(r_completo - r_sin_outliers) > 0.1:
    print("⚠️ Los outliers están distorsionando la correlación")
```

---

### Patrón 6: Correlación elástica (forma de L)
```
Y │•
  │• •
  │• • •
  │• • • •  •
  │• • • • • • • •    •        •
  └──────────────────────────────→ X
```
**Qué significa:**
Volúmenes altos solo ocurren a precios bajos.
A precios altos, el volumen cae drásticamente.
Forma característica de mercados sensibles al precio (commodities).

**Este es exactamente el patrón que encontré en mi proyecto EDA:**
Unidades >100 solo ocurren a precios <$50.

**Insight de negocio:**
Mercado elástico al precio. Bajar el precio puede aumentar volumen, 
pero necesitas analizar si el margen total mejora o empeora.

---

### Patrón 7: Múltiples nubes separadas
```
Y │              • •
  │           • • •        Grupo B
  │          • •
  │
  │   • •
  │• • •           Grupo A
  │  • •
  └──────────────────────→ X
```
**Qué significa:**
Hay subgrupos distintos en los datos, cada uno con su propia correlación interna.
Mezclarlos puede ocultar o crear correlaciones falsas.

**Ejemplos reales:**
- Clientes minoristas vs mayoristas
- Pacientes con tratamiento A vs tratamiento B
- Ventas en temporada alta vs baja

**Insight:**
Segmenta y analiza cada grupo por separado. La correlación del dataset 
completo puede ser completamente diferente a la de cada subgrupo.

---

## 🔢 La correlación de Pearson: cómo interpretarla

```python
# Calcular
r = df['col1'].corr(df['col2'])
print(f"r = {r:.3f}")

# Con p-value (significancia estadística)
from scipy import stats
r_valor, p_valor = stats.pearsonr(df['col1'], df['col2'])
print(f"r = {r_valor:.3f}")
print(f"p = {p_valor:.4f}")
print(f"¿Significativo? {'Sí' if p_valor < 0.05 else 'No'} (p < 0.05)")
```

### Escala de interpretación

```
r = +1.00  →  Correlación positiva perfecta
r = +0.90  →  Muy fuerte positiva
r = +0.70  →  Fuerte positiva
r = +0.50  →  Moderada positiva
r = +0.30  →  Débil positiva
r =  0.00  →  Sin correlación lineal
r = -0.30  →  Débil negativa
r = -0.50  →  Moderada negativa
r = -0.70  →  Fuerte negativa
r = -0.90  →  Muy fuerte negativa
r = -1.00  →  Correlación negativa perfecta
```

### La advertencia más importante

> **Correlación ≠ Causalidad**
> Que dos variables se muevan juntas NO significa que una cause la otra.

**Ejemplos de correlaciones sin causalidad:**
- Ventas de helado correlacionan con muertes por ahogamiento
- *(ambas suben en verano, el calor es la causa real)*
- Número de televisores per cápita correlaciona con esperanza de vida
  *(ambos indican nivel de desarrollo, no es causal)*
- Producción de pollo en EE.UU. correlaciona con importaciones de petróleo
  *(correlación espuria, coincidencia estadística)*

```python
# Siempre pregúntate:
# ¿Hay una tercera variable que explique ambas?
# ¿La relación tiene sentido lógico/causal?
# ¿Es solo una coincidencia estadística?
```

---

## 🎯 Cuándo usar scatter plot vs otras gráficas

| Necesito... | Usa |
|---|---|
| Ver relación entre 2 numéricas | Scatter |
| Ver si la relación es lineal o curva | Scatter |
| Ver correlación entre muchas variables | Heatmap de correlación |
| Comparar grupos en una variable | Boxplot |
| Ver tendencia en el tiempo | Line chart |
| Ver distribución de una variable | Histograma |
| Ver relación + distribución de ambas | Scatter + histogramas marginales |

```python
# Scatter con distribuciones marginales (la versión más completa)
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2,
                       width_ratios=[4, 1],
                       height_ratios=[1, 4])

ax_scatter = fig.add_subplot(gs[1, 0])
ax_hist_x  = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
ax_hist_y  = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

# Scatter central
ax_scatter.scatter(df['precio_unitario'],
                   df['unidades_vendidas'],
                   alpha=0.3, s=10, color='steelblue')
ax_scatter.set_xlabel('Precio Unitario ($)')
ax_scatter.set_ylabel('Unidades Vendidas')

# Histograma superior (X)
ax_hist_x.hist(df['precio_unitario'],
               bins=40, color='steelblue', alpha=0.7)
ax_hist_x.set_ylabel('Frecuencia')

# Histograma derecho (Y)
ax_hist_y.hist(df['unidades_vendidas'],
               bins=40, orientation='horizontal',
               color='teal', alpha=0.7)
ax_hist_y.set_xlabel('Frecuencia')

r = df['precio_unitario'].corr(df['unidades_vendidas'])
fig.suptitle(f'Precio vs Unidades  |  r = {r:.3f}',
             fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig('images/scatter_con_marginales.png',
            dpi=150, bbox_inches='tight')
plt.show()
```

---

## ❌ Errores comunes

**Error 1: No usar alpha (transparencia)**
Con miles de puntos superpuestos, el scatter se convierte en una mancha sólida.
Siempre usa `alpha` entre 0.1 y 0.5 con datasets grandes.

```python
# ❌ Con 100,000 puntos: mancha ilegible
ax.scatter(df['x'], df['y'])

# ✅ Con alpha: puedes ver la densidad
ax.scatter(df['x'], df['y'], alpha=0.1, s=5)
```

**Error 2: No graficar antes de calcular r**
El coeficiente de correlación puede ser el mismo para distribuciones
completamente diferentes (Cuarteto de Anscombe).

```
Dataset A: r = 0.816  → correlación lineal real
Dataset B: r = 0.816  → correlación curvilínea
Dataset C: r = 0.816  → correlación perfecta con 1 outlier
Dataset D: r = 0.816  → sin correlación + 1 outlier extremo
```

**Siempre: gráfica primero, calcula después.**

**Error 3: Asumir causalidad**
La correlación más fuerte del mundo no prueba que A cause B.
Siempre busca la explicación causal fuera de los datos.

**Error 4: No muestrear con datasets grandes**
Con más de 50,000 puntos, el scatter tarda mucho y se vuelve ilegible.
Usa una muestra representativa.

```python
# Con datasets grandes
df_muestra = df.sample(n=5000, random_state=42)
ax.scatter(df_muestra['x'], df_muestra['y'], alpha=0.3)

# O usa hexbin, muestra densidad de puntos
ax.hexbin(df['x'], df['y'],
          gridsize=30, cmap='YlOrRd')
plt.colorbar(ax.collections[0], label='Frecuencia')
```

**Error 5: Variables en el eje equivocado**
Si quieres predecir Y a partir de X, la variable que "causa" o predice va siempre en el eje X.
La variable resultado va en el eje Y.

---

## <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzMxY200dzhyc2RhbG5zbnk3MWZoZHE2ODZnM3I3Y2lseHNtYXI4diZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/WmvhTzEl9ihy9TqPme/giphy.gif" width="40"> Analogía con Medicina Tradicional China

El scatter plot es como el diagnóstico de correlación entre síntomas en MTC:

| Scatter | Diagnóstico MTC |
|---|---|
| Cada punto | Un paciente con sus dos síntomas medidos |
| Correlación positiva fuerte | Síntomas que siempre aparecen juntos → mismo síndrome |
| Sin correlación | Síntomas independientes → síndrome diferente |
| Correlación negativa | A más de un síntoma, menos del otro → balance Yin-Yang |
| Outlier alejado | El paciente atípico → diagnóstico diferencial |
| Dos nubes separadas | Dos síndromes distintos mezclados en la muestra |
| Correlación ≠ causalidad | Un síntoma no causa al otro → ambos son manifestación del mismo desequilibrio |

> En MTC, que fiebre y sed aparezcan juntas no significa que la fiebre cause la sed,
> ambas son manifestaciones del mismo patrón de calor interno.
> La correlación muestra el patrón, no la causa.

---

## ⚡ Cheat Sheet rápido

```
PATRÓN DEL SCATTER → LO QUE SIGNIFICA

Diagonal ascendente  →  Correlación positiva   →  r positivo
Diagonal descendente →  Correlación negativa   →  r negativo
Nube aleatoria       →  Sin correlación lineal →  r ≈ 0
Forma de U o curva   →  Relación no lineal     →  r puede engañar
Forma de L           →  Elasticidad            →  mercado sensible al precio
Dos nubes separadas  →  Subgrupos distintos    →  segmenta antes de analizar
Outlier aislado      →  Distorsiona el r       →  analiza con y sin él

SIEMPRE:
→ alpha entre 0.1 y 0.5 con datos grandes
→ grafica ANTES de calcular r
→ añade línea de tendencia
→ escribe r en el título
→ usa escala log si hay outliers extremos
→ NO concluyas causalidad solo por correlación

CON MÁS VARIABLES:
→ hue = tercera variable categórica (color)
→ size = cuarta variable numérica (tamaño)
→ pairplot = todas las combinaciones posibles
```

---

## 🔗 Aplicado a mis proyectos

### Proyecto EDA: Ventas Retail

```python
# Los 3 scatter plots que hice y qué encontré:

# SCATTER 1: Precio vs Unidades Vendidas
# → Forma de L característica
# → Volúmenes >100 unidades solo a precios <$50
# → r = -0.12 (correlación débil negativa)
# → Insight: mercado elástico al precio (commodities de limpieza)

# SCATTER 2: Promedio Semanal vs Valor de Venta
# → Correlación positiva moderada
# → Patrón de abanico: más dispersión a volúmenes altos
# → Confirma: las compras grandes son más variables en precio

# SCATTER 3: Tamaño del Producto vs Precio
# → Correlación positiva débil
# → Muchos clusters verticales: indica que el tamaño tiene rangos discretos (presentaciones fijas)
# → Productos de mismo tamaño tienen precios muy variables según marca
```

### Posibles proyectos futuros
| Dataset | Variables a cruzar | Patrón esperado |
|---|---|---|
| SEMARNAT | PIB estatal vs emisiones CO₂ | Positivo → más industria más emisiones |
| OMS | PIB per cápita vs esperanza de vida | Positivo curvilíneo → efecto techo |
| INEGI | Educación vs ingreso familiar | Positivo moderado a fuerte |
| Kaggle fraude | Monto vs frecuencia transacciones | Nube + outliers = fraudes |
| IMSS | Edad vs número de diagnósticos | Positivo → más edad más condiciones |

---

*Nota creada: 2026 · Serie: Visualizaciones para Ciencia de Datos*
*Parte del certificado Científico de Datos — EBAC*
*🪷 github.com/ReginaPema*
