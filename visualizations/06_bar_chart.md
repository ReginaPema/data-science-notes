#  <img src="https://img.icons8.com/?size=50&id=80255&format=png&color=000000" align="center"/> Bar Chart
## Gráfica de Barras — Comparar categorías

> El bar chart es la gráfica de la comparación: cuando quieres saber cuál categoría es mayor,
> cuál es menor, y qué tan grande es la diferencia entre ellas.

> **Regla de oro:**
> Si tu eje X tiene categorías (no tiempo, no números), usa bar chart.
> La altura de la barra es una métrica que el ojo humano compara con precisión.
> Todo lo demás (ángulos, áreas, colores) es menos preciso visualmente.

---

## 🧱 Anatomía de un bar chart

```
Valor │
      │  ████
      │  ████
      │  ████  ████
      │  ████  ████
      │  ████  ████  ████
      │  ████  ████  ████  ████
      │  ████  ████  ████  ████  ████
      └──────────────────────────────→ Categorías
        Cat A   Cat B  Cat C  Cat D  Cat E

Elementos clave:
* Cada barra   = una categoría
* Altura/largo = el valor de la métrica
* Baseline (0) = empieza en cero
* Orden        = de mayor a menor (recomendado)
* Espacio      = separación entre barras
```

| Elemento | Qué representa |
|---|---|
| Cada barra | Una categoría (región, segmento, marca, mes...) |
| Altura de la barra | El valor de la métrica para esa categoría |
| Eje X | Las categorías a comparar |
| Eje Y | La escala de la métrica |
| Baseline | Siempre en 0 (preferible no cambiarlo) |
| Orden de barras | De mayor a menor (facilita la lectura) |
| Color | Puede indicar una tercera variable o énfasis |
| Etiqueta de valor | El número exacto encima o dentro de cada barra |

---

## 🐍 Cómo hacerlo en Python

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ── BAR CHART VERTICAL BÁSICO ──
ventas_segmento = (df.groupby('segmento')['ventas_total']
                     .sum()
                     .sort_values(ascending=False)
                     .reset_index())

fig, ax = plt.subplots(figsize=(10, 6))

barras = ax.bar(ventas_segmento['segmento'],
                ventas_segmento['ventas_total'] / 1e6,
                color='steelblue',
                edgecolor='white',
                linewidth=0.5)

ax.set_title('Ventas Totales por Segmento', fontsize=14)
ax.set_xlabel('Segmento')
ax.set_ylabel('Ventas ($M)')
ax.set_ylim(0, ventas_segmento['ventas_total'].max() / 1e6 * 1.15)

# Etiquetas de valor encima de cada barra
for barra in barras:
    altura = barra.get_height()
    ax.text(barra.get_x() + barra.get_width() / 2,
            altura + 0.05,
            f'${altura:.1f}M',
            ha='center', va='bottom',
            fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('images/barchart_vertical.png', dpi=150)
plt.show()

# ── BAR CHART HORIZONTAL ──
# Mejor cuando los nombres de categorías son largos
ventas_marca = (df.groupby('marca')['ventas_total']
                  .sum()
                  .sort_values(ascending=True)  # ascendente para horizontal
                  .tail(10)                     # top 10
                  .reset_index())

fig, ax = plt.subplots(figsize=(10, 7))

barras = ax.barh(ventas_marca['marca'],
                 ventas_marca['ventas_total'] / 1e6,
                 color='teal', edgecolor='white')

ax.set_title('Top 10 Marcas por Ventas Totales', fontsize=14)
ax.set_xlabel('Ventas ($M)')
ax.set_xlim(0, ventas_marca['ventas_total'].max() / 1e6 * 1.15)

# Etiquetas dentro de la barra
for barra in barras:
    largo = barra.get_width()
    ax.text(largo + 0.05, barra.get_y() + barra.get_height() / 2,
            f'${largo:.2f}M', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('images/barchart_horizontal.png', dpi=150)
plt.show()

# ── CON COLOR DESTACANDO UNA BARRA ──
# Para enfatizar una categoría específica
colores = ['#E63946' if seg == 'Bleach'
           else '#ADB5BD' for seg in ventas_segmento['segmento']]

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(ventas_segmento['segmento'],
       ventas_segmento['ventas_total'] / 1e6,
       color=colores, edgecolor='white')

ax.set_title('Ventas por Segmento — Bleach Domina el Mercado', fontsize=14)
ax.set_xlabel('Segmento')
ax.set_ylabel('Ventas ($M)')

# Anotación explicativa
ax.annotate('68.7% del\ntotal de ventas',
            xy=(0, 7.6),
            xytext=(1.5, 6),
            fontsize=10,
            color='#E63946',
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#E63946'))

plt.tight_layout()
plt.savefig('images/barchart_destacado.png', dpi=150)
plt.show()

# ── BARRAS AGRUPADAS (grouped bars) ──
# Para comparar dos métricas por categoría
resumen = df.groupby('region').agg(
    ventas_total=('ventas_total', 'sum'),
    ticket_promedio=('ventas_total', 'mean')
).reset_index()

x = np.arange(len(resumen['region']))
ancho = 0.35

fig, ax1 = plt.subplots(figsize=(12, 6))

# Barras de ventas totales
barras1 = ax1.bar(x - ancho/2,
                  resumen['ventas_total'] / 1e6,
                  ancho,
                  label='Ventas totales ($M)',
                  color='steelblue',
                  edgecolor='white')

ax1.set_ylabel('Ventas Totales ($M)', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')

# Segundo eje — ticket promedio
ax2 = ax1.twinx()
barras2 = ax2.bar(x + ancho/2,
                  resumen['ticket_promedio'],
                  ancho,
                  label='Ticket promedio ($)',
                  color='coral',
                  edgecolor='white')

ax2.set_ylabel('Ticket Promedio ($)', color='coral')
ax2.tick_params(axis='y', labelcolor='coral')

ax1.set_xticks(x)
ax1.set_xticklabels(resumen['region'], rotation=45)
ax1.set_title('Ventas Totales vs Ticket Promedio por Región', fontsize=14)

# Leyenda combinada
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('images/barchart_agrupado.png', dpi=150)
plt.show()

# ── BARRAS APILADAS (stacked bars) ──
# Para mostrar composición dentro de cada categoría
ventas_pivot = df.pivot_table(
    values='ventas_total',
    index='region',
    columns='segmento',
    aggfunc='sum'
) / 1e6

ventas_pivot = ventas_pivot.fillna(0)

fig, ax = plt.subplots(figsize=(12, 6))

colores_segmento = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51', '#A8DADC', '#457B9D']

bottom = np.zeros(len(ventas_pivot))

for i, segmento in enumerate(ventas_pivot.columns):
    ax.bar(ventas_pivot.index,
           ventas_pivot[segmento],
           bottom=bottom,
           label=segmento,
           color=colores_segmento[i % len(colores_segmento)],
           edgecolor='white',
           linewidth=0.5)
    bottom += ventas_pivot[segmento].values

ax.set_title('Composición de Ventas por Región y Segmento', fontsize=14)
ax.set_xlabel('Región')
ax.set_ylabel('Ventas ($M)')
ax.legend(title='Segmento', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('images/barchart_apilado.png', dpi=150)
plt.show()

# ── BARRAS APILADAS 100% ──
# Para comparar proporciones ignorando magnitud total
ventas_pct = ventas_pivot.div(ventas_pivot.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(12, 6))
bottom = np.zeros(len(ventas_pct))

for i, segmento in enumerate(ventas_pct.columns):
    ax.bar(ventas_pct.index,
           ventas_pct[segmento],
           bottom=bottom,
           label=segmento,
           color=colores_segmento[i % len(colores_segmento)],
           edgecolor='white',
           linewidth=0.5)
    bottom += ventas_pct[segmento].values

ax.set_title('Composición % de Ventas por Región y Segmento\n(Barras al 100%)', fontsize=14)
ax.set_xlabel('Región')
ax.set_ylabel('Participación (%)')
ax.set_ylim(0, 100)
ax.legend(title='Segmento', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('images/barchart_apilado_100.png', dpi=150)
plt.show()

# ── CON SEABORN ──
# Más sencillo para barras con error estándar
fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(data=df,
            x='segmento',
            y='ventas_total',
            estimator='sum',      # suma, mean, median, etc.
            errorbar=None,        # sin barras de error
            palette='Blues_d',
            order=df.groupby('segmento')['ventas_total']
                    .sum()
                    .sort_values(ascending=False)
                    .index,
            ax=ax)

ax.set_title('Ventas por Segmento (Seaborn)', fontsize=14)
ax.set_xlabel('Segmento')
ax.set_ylabel('Ventas Totales ($)')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()
```

---

## 🧭 Los 5 tipos de bar chart y cuándo usar cada uno

### Tipo 1: Vertical: comparar magnitudes
```
█
█  █
█  █  █
█  █  █  █
─────────────
A  B  C  D
```
**Cuándo:** Pocas categorías (2-8), nombres cortos, quieres comparar magnitudes.

**Ejemplos:** Ventas por segmento, ingresos por región, usuarios por país.

---

### Tipo 2: Horizontal: nombres largos
```
Categoria larga A │████████████
Categoria larga B │██████████
Categoria larga C │████████
Categoria larga D │██████
──────────────────────────────→
```
**Cuándo:** Nombres de categorías largos, muchas categorías (8+), rankings.

**Ejemplos:** Top 10 marcas, top 20 países, lista de productos por nombre completo.

---

### Tipo 3: Agrupadas: comparar dos métricas
```
█ ░   █ ░   █ ░   █ ░
█ ░   █ ░   █ ░   █ ░
─────────────────────
  A     B     C     D
█ = métrica 1  ░ = métrica 2
```
**Cuándo:** Quieres comparar dos métricas para las mismas categorías simultáneamente.

**Ejemplos:** Ventas vs meta por región, precio promedio vs margen por producto,
2022 vs 2023 por trimestre.

---

### Tipo 4: Apiladas: composición absoluta
```
█░▓   █░▓   █░▓
█░▓   █░▓   █░▓
█░▓   █░▓   █░▓
─────────────────
 A     B     C
```
**Cuándo:** Quieres ver el total de cada categoría Y su composición interna simultáneamente.

**Ejemplos:** Ventas por región desglosadas por segmento, ingresos por año desglosados
por producto, población por país desglosada por grupo etario.

---

### Tipo 5: Apiladas 100%: composición relativa
```
██░░▓▓▓   ██░░░▓▓▓   ███░▓▓
██░░▓▓▓   ██░░░▓▓▓   ███░▓▓
──────────────────────────────
    A          B          C
(todas al 100%)
```
**Cuándo:** Quieres comparar la proporción interna de cada categoría ignorando sus magnitudes absolutas.

**Ejemplos:** ¿En qué región Bleach tiene mayor dominancia relativa?
¿En qué año la mezcla de productos fue más diversificada?

---

## 🎨 Principios de diseño para bar charts

### 1. Siempre ordena las barras
```python
# ✗ Sin orden → más difícil de leer
ax.bar(df['categoria'], df['valor'])

# ✓ Ordenado de mayor a menor
df_ordenado = df.sort_values('valor', ascending=False)
ax.bar(df_ordenado['categoria'], df_ordenado['valor'])

# Excepción: cuando el orden tiene significado
# (meses del año, días de la semana, etapas de un proceso)
```

### 2. Siempre empieza en 0
```python
# ✗ Engañoso → exagera las diferencias
ax.set_ylim(5_000_000, 8_000_000)

# ✓ Honesto → perspectiva correcta
ax.set_ylim(0, df['ventas'].max() * 1.15)
```

### 3. Agrega etiquetas de valor
```python
# Los números exactos eliminan ambigüedad visual
for barra in barras:
    altura = barra.get_height()
    ax.text(barra.get_x() + barra.get_width() / 2,
            altura + ax.get_ylim()[1] * 0.01,
            f'${altura/1e6:.1f}M',
            ha='center', va='bottom',
            fontsize=9, fontweight='bold')
```

### 4. Usa color con propósito
```python
# ✗ Color diferente por estética → distrae
colores_random = ['red', 'blue', 'green', 'yellow', 'purple']
ax.bar(categorias, valores, color=colores_random)

# ✓ Un solo color para comparación simple
ax.bar(categorias, valores, color='steelblue')

# ✓ Color diferente para destacar una categoría
colores = ['#E63946' if c == 'Bleach' else '#ADB5BD'
           for c in categorias]
ax.bar(categorias, valores, color=colores)

# ✓ Gradiente para mostrar ranking
palette = sns.color_palette('Blues_d', n_colors=len(categorias))
ax.bar(categorias, valores, color=palette)
```

### 5. Limita el número de categorías
```
Hasta 8 categorías → Bar chart vertical
8-15 categorías    → Bar chart horizontal
Más de 15          → Agrupa las menores en "Otros"
                       o usa un top N

# Agrupar categorías menores
umbral = df['ventas'].quantile(0.2)  # bottom 20%
df['categoria_plot'] = df['categoria'].where(
    df['ventas'] > umbral, other='Otros')
```

---

## 🔍 Bar chart vs otras gráficas

| Necesito... | Usa |
|---|---|
| Comparar magnitudes entre categorías | Bar chart |
| Ver tendencia en el tiempo | Line chart |
| Ver composición de un todo | Bar apilado o Pie chart |
| Ver distribución de valores | Histograma |
| Ver relación entre dos numéricas | Scatter |
| Comparar distribuciones entre grupos | Boxplot |
| Ranking de muchos elementos | Bar horizontal |

**La confusión más común:**
```
Bar chart vs Line chart con tiempo:

Datos mensuales de CATEGORÍAS distintas:
→ Bar chart (cada mes es una categoría)

Datos mensuales de UNA MÉTRICA en el tiempo:
→ Line chart (hay continuidad entre meses)

¿Cuál usar para ventas mensuales?
→ Si quieres mostrar COMPARACIÓN entre meses: bar
→ Si quieres mostrar TENDENCIA temporal: line
→ En la práctica: muchos proyectos usan ambos
```

---

## ❌ Errores comunes

**Error 1: Cortar el eje Y en 0**
Es el error más grave en visualización.
Hace que diferencias pequeñas parezcan enormes.

```
✗ Engañoso:                    ✓ Honesto:
Ventas ($M)                     Ventas ($M)
7.6 │ █                         8 │ █
7.4 │    █                      6 │    █
7.2 │       █                   4 │       █
7.0 │          █                2 │          █
    │ A  B  C  D                0 │ A  B  C  D

La diferencia parece enorme    La diferencia real es pequeña
```

**Error 2: No ordenar las barras**
Sin orden, el ojo tiene que escanear toda la gráfica para encontrar el mayor y el menor.
Siempre ordena excepto cuando el orden tiene significado propio.

**Error 3: Demasiadas categorías**
Con 20+ barras la gráfica se vuelve ilegible.
Agrupa las menores en "Otros" o muestra solo el top N.

**Error 4: Barras en 3D**
Las barras 3D distorsionan la percepción de altura → las barras del fondo parecen más cortas de lo que son.
Nunca uses 3D en visualización de datos.

**Error 5: Usar pie chart cuando deberías usar bar**
El ojo humano compara longitudes con mucha más precisión que ángulos.
Para la mayoría de comparaciones, el bar chart es más preciso que el pie.

**Error 6: No poner etiquetas de valor**
Si la audiencia no puede leer los valores exactos, no puede tomar decisiones precisas.
Siempre agrega los números.

---

## <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzMxY200dzhyc2RhbG5zbnk3MWZoZHE2ODZnM3I3Y2lseHNtYXI4diZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/WmvhTzEl9ihy9TqPme/giphy.gif" width="40" Analogía con Medicina Tradicional China

El bar chart es como el diagnóstico comparativo entre pacientes o tratamientos:

| Bar chart | Comparación clínica MTC |
|---|---|
| Cada barra | Un paciente, tratamiento o condición |
| Altura de barra | Intensidad del síntoma o efectividad |
| Orden de barras | Ranking de efectividad de tratamientos |
| Barra destacada en rojo | El caso más severo o el más efectivo |
| Barras agrupadas | Mismo paciente antes y después del tratamiento |
| Barras apiladas | Composición del patrón → cuánto de cada elemento |
| Barras 100% | Proporción relativa → qué elemento domina el patrón |

> En una revisión clínica, comparar la intensidad de síntomas entre
> 10 pacientes con el mismo diagnóstico es exactamente un bar chart mental.
> El más alto necesita tratamiento más urgente o intensivo.

---

## ⚡ Cheat Sheet rápido

```
TIPOS Y CUÁNDO USAR:
→ Vertical      = pocas categorías, nombres cortos
→ Horizontal    = muchas categorías, nombres largos, rankings
→ Agrupadas     = comparar dos métricas por categoría
→ Apiladas      = total + composición simultáneos
→ Apiladas 100% = solo composición relativa

DISEÑO:
→ Siempre ordena de mayor a menor (excepto cuando el orden tiene significado)
→ Siempre empieza el eje en 0
→ Agrega etiquetas de valor en cada barra
→ Un color para comparación simple
→ Color diferente solo para destacar una barra
→ Máximo 8-10 barras por gráfica

NUNCA:
→ Cortar el eje Y (empieza siempre en 0)
→ Usar 3D
→ Usar colores diferentes sin propósito
→ Mostrar 20+ barras sin agrupar
→ Usar pie chart cuando bar es más claro

BAR vs LINE:
→ Comparación entre categorías = bar
→ Tendencia en el tiempo = line
→ Meses como categorías = bar o line (ambos válidos)
```

---

## 🔗 Aplicado a mis proyectos

### Proyecto EDA: Ventas Retail

```python
# Los bar charts que hice y qué encontré:

# BAR CHART 1: Ventas totales por segmento
# → Bleach: $7.6M — barra enormemente más alta
# → Liquid & Gel: $1.5M — segunda
# → Visualizar esto fue el momento donde la concentración de mercado se hizo obvia
# → Antes del gráfico era un número ($7.6M) después del gráfico era una realidad visual

# BAR CHART 2: Top 10 marcas (horizontal)
# → Cloralex: $5.39M — dominancia aplastante
# → Vanish: $2.17M — segunda muy lejana
# → Usé horizontal porque los nombres de marca son largos y difíciles de rotar

# BAR CHART 3: Ventas por región
# → México: $5.5M (barra enorme)
# → Áreas 1-6: $0.7M-$1.2M cada una
# → La anomalía de Área 2 fue visible inmediatamente en el bar chart:
#   barra de ventas parecida a las otras Áreas pero ticket promedio muy diferente

# COMBINACIÓN QUE FUNCIONÓ BIEN:
# Bar apilado (ventas absolutas por región)
# + Bar 100% (composición por segmento en cada región)
# → Primera: ver magnitudes totales
# → Segunda: ver si la mezcla es similar o diferente  entre regiones — fue donde encontré que México
#   tiene mayor concentración en Bleach que las Áreas
```

### Posibles proyectos futuros
| Dataset | Bar chart útil | Insight esperado |
|---|---|---|
| SEMARNAT | Emisiones por estado (horizontal) | Qué estados contaminan más |
| OMS | Mortalidad por causa (horizontal) | Principales causas de muerte |
| INEGI | PIB por sector económico (apilado) | Composición de la economía |
| Kaggle fraude | Fraudes por categoría de comercio | Dónde ocurre más fraude |
| IMSS | Consultas por especialidad (horizontal) | Demanda por tipo de atención |

---

*Nota creada: 2026 · Serie: Visualizaciones para Ciencia de Datos*
*Parte del certificado Científico de Datos — EBAC*
*🪷 github.com/ReginaPema* 
