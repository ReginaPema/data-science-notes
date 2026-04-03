# EDA Process
## Exploratory Data Analysis · Análisis Exploratorio de Datos

> EDA es la conversación que tienes con tus datos antes de sacar cualquier conclusión.
> Es el paso que separa el análisis real de las suposiciones disfrazadas de insights.

> **Regla de oro:**
> Nunca empieces a modelar o a reportar sin haber hecho un EDA completo primero.
> Los datos siempre tienen sorpresas.

---

## 🗺️ El flujo completo del EDA
```
       Dataset limpio (output del ETL)
                     ↓
   ┌─────────────────────────────────────┐
   │  FASE 1: Entender el dataset        │
   │  ¿Qué tengo? ¿Cuánto? ¿De qué tipo? │
   └─────────────────┬───────────────────┘
                     ↓
   ┌─────────────────────────────────────┐
   │  FASE 2: Análisis univariado        │
   │  Una variable a la vez              │
   └─────────────────┬───────────────────┘
                     ↓
   ┌─────────────────────────────────────┐
   │  FASE 3: Análisis bivariado         │
   │  Relación entre dos variables       │
   └─────────────────┬───────────────────┘
                     ↓
   ┌─────────────────────────────────────┐
   │  FASE 4: Análisis multivariado      │
   │  Múltiples variables simultáneas    │
   └─────────────────┬───────────────────┘
                     ↓
   ┌─────────────────────────────────────┐
   │  FASE 5: Outliers y anomalías       │
   │  ¿Qué se sale de lo normal?         │
   └─────────────────┬───────────────────┘
                     ↓
   ┌─────────────────────────────────────┐
   │  FASE 6: Insights y recomendaciones │
   │  ¿Qué me dicen los datos?           │
   └─────────────────────────────────────┘
```

---

## FASE 1: Entender el dataset

### Para qué sirve
Antes de analizar cualquier cosa necesitas saber exactamente con que estás trabajando.

### Las preguntas que debes responder siempre
```python
import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/dataset_clean.csv')

# ── DIMENSIONES ──
print(f"Filas:    {df.shape[0]:,}")
print(f"Columnas: {df.shape[1]}")
print(f"Celdas:   {df.shape[0] * df.shape[1]:,}")

# ── TIPOS DE DATOS ──
print(df.dtypes)
print(f"\nNuméricas:  {df.select_dtypes(include='number').columns.tolist()}")
print(f"Categóricas:  {df.select_dtypes(include='object').columns.tolist()}")
print(f"Fechas:       {df.select_dtypes(include='datetime').columns.tolist()}")

# ── CALIDAD DE DATOS ──
# Nulos
nulos = df.isnull().sum()
pct_nulos = (nulos / len(df) * 100).round(2)
reporte_nulos = pd.DataFrame({
    'nulos': nulos,
    'porcentaje': pct_nulos
}).query('nulos > 0').sort_values('porcentaje', ascending=False)
print(reporte_nulos)

# Duplicados
print(f"\nDuplicados: {df.duplicated().sum()}")

# ── RANGO TEMPORAL (si hay fechas) ──
if 'fecha' in df.columns:
    print(f"\nFecha inicio: {df['fecha'].min()}")
    print(f"Fecha fin:      {df['fecha'].max()}")
    print(f"Periodo:        {(df['fecha'].max()-df['fecha'].min()).days} días")

# ── ESTADÍSTICAS GENERALES ──
print(df.describe())
# Media, std, min, Q1, mediana, Q3, max por columna numérica

# ── PRIMERAS FILAS ──
print(df.head())
print(df.sample(5))  # 5 filas aleatorias, más representativo que head()
```

### Lo que documentas al final de la Fase 1
```markdown
## Dataset Overview
- Filas: 122,002 transacciones
- Columnas: 18 variables
- Periodo: Enero 2022 – Julio 2023 (18 meses)
- Nulos: 0 (dataset ya limpio del ETL)
- Duplicados: 0
- Variables numéricas: precio, unidades, ventas_total
- Variables categóricas: región, segmento, marca, formato
- Variables temporales: fecha, año, mes, trimestre
```

---

## FASE 2: Análisis Univariado

### Para qué sirve
Entender cada variable por separado antes de buscar relaciones entre ellas.

### Variables numéricas
```python
import matplotlib.pyplot as plt
import seaborn as sns

def analizar_numerica(df, columna, titulo=None):
    """
    Análisis completo de una variable numérica.
    Combina estadísticas + visualización.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titulo = titulo or columna

    # Estadísticas
    media    = df[columna].mean()
    mediana  = df[columna].median()
    std      = df[columna].std()
    Q1       = df[columna].quantile(0.25)
    Q3       = df[columna].quantile(0.75)
    IQR      = Q3 - Q1
    lim_sup  = Q3 + 1.5 * IQR
    outliers = (df[columna] > lim_sup).sum()

    # Gráfica 1: Histograma
    axes[0].hist(df[columna], bins=50,
                 color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(media,   color='red',    ls='--', label=f'Media: {media:,.2f}')
    axes[0].axvline(mediana, color='orange', ls='--', label=f'Mediana: {mediana:,.2f}')
    axes[0].set_title(f'Distribución — {titulo}')
    axes[0].legend()

    # Gráfica 2: Boxplot
    axes[1].boxplot(df[columna], vert=True)
    axes[1].set_title(f'Boxplot — {titulo}')
    axes[1].set_ylabel(columna)

    # Gráfica 3: Histograma escala log (si hay outliers)
    axes[2].hist(df[columna].clip(lower=0.01),
                 bins=50, color='teal', edgecolor='white', alpha=0.8,
                 log=True)
    axes[2].set_title(f'Distribución (escala log) — {titulo}')

    plt.tight_layout()
    plt.savefig(f'images/univariado_{columna}.png', dpi=150)
    plt.show()

    # Reporte de texto
    print(f"\n{'='*40}")
    print(f"ANÁLISIS: {titulo}")
    print(f"{'='*40}")
    print(f"Media:      {media:>12,.2f}")
    print(f"Mediana:    {mediana:>12,.2f}")
    print(f"Std Dev:    {std:>12,.2f}")
    print(f"Mínimo:     {df[columna].min():>12,.2f}")
    print(f"Q1 (25%):   {Q1:>12,.2f}")
    print(f"Q3 (75%):   {Q3:>12,.2f}")
    print(f"Máximo:     {df[columna].max():>12,.2f}")
    print(f"IQR:        {IQR:>12,.2f}")
    print(f"Outliers:   {outliers:>12,} ({outliers/len(df)*100:.1f}%)")

    # Interpretar sesgo
    if media > mediana * 1.1:
        print(f"\n  Sesgo POSITIVO: media > mediana")
        print(f"   El promedio sobreestima el caso típico")
    elif mediana > media * 1.1:
        print(f"\n  Sesgo NEGATIVO: mediana > media")
        print(f"   El promedio subestima el caso típico")
    else:
        print(f"\n Distribución aproximadamente simétrica")

# Usar la función:
analizar_numerica(df, 'ventas_total', 'Ventas Totales ($)')
analizar_numerica(df, 'precio_unitario', 'Precio Unitario ($)')
analizar_numerica(df, 'unidades_vendidas', 'Unidades Vendidas')
```

### Variables categóricas
```python
def analizar_categorica(df, columna, top_n=10, titulo=None):
    """
    Análisis completo de una variable categórica.
    """
    titulo = titulo or columna
    conteo = df[columna].value_counts()
    pct    = df[columna].value_counts(normalize=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfica 1: Barras horizontales (top N)
    top = conteo.head(top_n)
    axes[0].barh(top.index[::-1], top.values[::-1],
                 color='steelblue')
    axes[0].set_title(f'Top {top_n} — {titulo} (frecuencia)')
    axes[0].set_xlabel('Transacciones')

    # Gráfica 2: Pie chart (participación %)
    top_pct = pct.head(top_n)
    axes[1].pie(top_pct.values,
                labels=top_pct.index,
                autopct='%1.1f%%',
                startangle=90)
    axes[1].set_title(f'Participación % — {titulo}')

    plt.tight_layout()
    plt.savefig(f'images/univariado_{columna}.png', dpi=150)
    plt.show()

    # Reporte
    print(f"\n{'='*40}")
    print(f"ANÁLISIS CATEGÓRICO: {titulo}")
    print(f"{'='*40}")
    print(f"Categorías únicas: {df[columna].nunique()}")
    print(f"\nTop {top_n}:")
    for cat, cnt, p in zip(conteo.index[:top_n],
                            conteo.values[:top_n],
                            pct.values[:top_n]):
        print(f"  {cat:<20} {cnt:>8,}  ({p:.1f}%)")

# Usar la función:
analizar_categorica(df, 'segmento', 'Segmento de Producto')
analizar_categorica(df, 'region',   'Región Geográfica')
analizar_categorica(df, 'marca',    'Marca', top_n=10)
```

---

## FASE 3: Análisis Bivariado

### Para qué sirve
Entender cómo se relacionan dos variables entre sí. ¿Cuándo una sube, sube la otra? 
¿Hay diferencias entre grupos? ¿Qué categoría vende más?

### Numérica vs Numérica → Correlación
```python
# Scatter plot — relación entre dos numéricas
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df['precio_unitario'], df['unidades_vendidas'],
           alpha=0.3, s=10, color='steelblue')
ax.set_xlabel('Precio Unitario ($)')
ax.set_ylabel('Unidades Vendidas')
ax.set_title('Precio vs Unidades Vendidas')

# Línea de tendencia
z = np.polyfit(df['precio_unitario'], df['unidades_vendidas'], 1)
p = np.poly1d(z)
ax.plot(sorted(df['precio_unitario']),
        p(sorted(df['precio_unitario'])),
        color='red', linewidth=2, label='Tendencia')
ax.legend()
plt.savefig('images/scatter_precio_unidades.png', dpi=150)
plt.show()

# Correlación numérica
correlacion = df['precio_unitario'].corr(df['unidades_vendidas'])
print(f"Correlación de Pearson: {correlacion:.3f}")

# Interpretar
if abs(correlacion) >= 0.7:
    fuerza = "FUERTE"
elif abs(correlacion) >= 0.4:
    fuerza = "MODERADA"
else:
    fuerza = "DÉBIL"

direccion = "POSITIVA" if correlacion > 0 else "NEGATIVA"
print(f"→ Correlación {fuerza} {direccion}")
```

### Categórica vs Numérica → Comparación entre grupos
```python
# Ventas totales por categoría
ventas_por_grupo = (df.groupby('segmento')['ventas_total']
                      .sum()
                      .sort_values(ascending=True))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Barras — totales
axes[0].barh(ventas_por_grupo.index,
             ventas_por_grupo.values / 1e6,
             color='steelblue')
axes[0].set_title('Ventas Totales por Segmento ($M)')
axes[0].set_xlabel('Ventas ($M)')

# Boxplot — distribución dentro de cada grupo
sns.boxplot(data=df, x='segmento', y='ventas_total',
            palette='pastel', ax=axes[1])
axes[1].set_yscale('log')
axes[1].set_title('Distribución de Ventas por Segmento')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('images/bivariado_segmento_ventas.png', dpi=150)
plt.show()

# Métricas por grupo
resumen = df.groupby('segmento')['ventas_total'].agg([
    ('total',      'sum'),
    ('media',      'mean'),
    ('mediana',    'median'),
    ('transacc',   'count'),
    ('pct_total',  lambda x: x.sum() / df['ventas_total'].sum() * 100)
]).sort_values('total', ascending=False).round(2)
print(resumen)
```

### Temporal → evolución en el tiempo
```python
# Tendencia mensual
ventas_mensual = (df.groupby(['año', 'mes'])['ventas_total']
                    .sum()
                    .reset_index())
ventas_mensual['periodo'] = (ventas_mensual['año'].astype(str) + '-' +
                              ventas_mensual['mes'].astype(str).str.zfill(2))
ventas_mensual = ventas_mensual.sort_values('periodo')

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(ventas_mensual['periodo'],
        ventas_mensual['ventas_total'] / 1000,
        marker='o', linewidth=2, color='steelblue',
        markersize=6, label='Ventas mensuales')

# Promedio móvil de 3 meses
ventas_mensual['rolling_3m'] = (ventas_mensual['ventas_total']
                                  .rolling(window=3, center=True)
                                  .mean())
ax.plot(ventas_mensual['periodo'],
        ventas_mensual['rolling_3m'] / 1000,
        linewidth=2, color='red', ls='--',
        label='Promedio móvil 3 meses')

ax.set_title('Tendencia Mensual de Ventas')
ax.set_ylabel('Ventas ($K)')
ax.tick_params(axis='x', rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/tendencia_temporal.png', dpi=150)
plt.show()
```

---

## FASE 4: Análisis Multivariado

### Para qué sirve
Cuando ya entiendes las variables individualmente y en pares, el análisis multivariado te permite
ver patrones que solo emergen cuando combinas tres o más variables simultáneamente.

### Mapa de calor de correlaciones
```python
# Seleccionar solo numéricas
numericas = df.select_dtypes(include='number')
correlaciones = numericas.corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(correlaciones, dtype=bool))  # triángulo superior

sns.heatmap(correlaciones,
            annot=True,          # mostrar números
            fmt='.2f',           # 2 decimales
            cmap='RdBu_r',       # rojo=positivo, azul=negativo
            center=0,
            mask=mask,           # solo triángulo inferior
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlación de Pearson'},
            ax=ax)

ax.set_title('Matriz de Correlación entre Variables Numéricas',
             fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('images/heatmap_correlacion.png', dpi=150)
plt.show()

# Identificar correlaciones fuertes automáticamente
print("\n Correlaciones fuertes (|r| > 0.5):")
for col1 in correlaciones.columns:
    for col2 in correlaciones.columns:
        if col1 < col2:  # evitar duplicados y diagonal
            r = correlaciones.loc[col1, col2]
            if abs(r) > 0.5:
                direccion = "positiva ↑↑" if r > 0 else "negativa ↑↓"
                print(f"  {col1} ↔ {col2}: r={r:.3f} ({direccion})")
```

### Segmentación cruzada
```python
# Ventas por región Y segmento simultáneamente
tabla_pivot = df.pivot_table(
    values='ventas_total',
    index='region',
    columns='segmento',
    aggfunc='sum'
) / 1e6  # en millones

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(tabla_pivot,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            linewidths=0.5,
            ax=ax)
ax.set_title('Ventas ($M) por Región y Segmento')
plt.tight_layout()
plt.savefig('images/heatmap_region_segmento.png', dpi=150)
plt.show()
```

---

## FASE 5: Outliers y Anomalías

### Para qué sirve
Identificar valores que se salen del patrón general.
No para eliminarlos automáticamente, sino para
entender qué representan y decidir qué hacer con ellos.
```python
# ── DETECCIÓN ──
def detectar_outliers_iqr(df, columna):
    Q1  = df[columna].quantile(0.25)
    Q3  = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR

    outliers = df[(df[columna] < lim_inf) |
                  (df[columna] > lim_sup)]

    print(f"Variable: {columna}")
    print(f"  Límite inferior: {lim_inf:,.2f}")
    print(f"  Límite superior: {lim_sup:,.2f}")
    print(f"  Outliers: {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")
    print(f"  Valor máximo: {df[columna].max():,.2f}")

    return outliers

outliers_ventas = detectar_outliers_iqr(df, 'ventas_total')

# ── ANÁLISIS DE OUTLIERS ──
# ¿De dónde vienen? ¿Son todos del mismo grupo?
print("\n¿De qué segmento son los outliers?")
print(outliers_ventas['segmento'].value_counts())

print("\n¿De qué región?")
print(outliers_ventas['region'].value_counts())

print("\n¿Qué marca?")
print(outliers_ventas['marca'].value_counts().head(5))

# ── VISUALIZACIÓN ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Boxplot general
axes[0,0].boxplot(df['ventas_total'])
axes[0,0].set_yscale('log')
axes[0,0].set_title('Outliers — General')

# Por segmento
sns.boxplot(data=df, x='segmento', y='ventas_total',
            palette='pastel', ax=axes[0,1])
axes[0,1].set_yscale('log')
axes[0,1].set_title('Outliers por Segmento')
axes[0,1].tick_params(axis='x', rotation=45)

# Por región
sns.boxplot(data=df, x='region', y='ventas_total',
            palette='pastel', ax=axes[1,0])
axes[1,0].set_yscale('log')
axes[1,0].set_title('Outliers por Región')
axes[1,0].tick_params(axis='x', rotation=45)

# Por marca (top 10)
top_marcas = df['marca'].value_counts().head(10).index
sns.boxplot(data=df[df['marca'].isin(top_marcas)],
            x='marca', y='ventas_total',
            palette='pastel', ax=axes[1,1])
axes[1,1].set_yscale('log')
axes[1,1].set_title('Outliers por Marca (Top 10)')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('images/outliers_analisis.png', dpi=150)
plt.show()

# ── DECISIÓN: qué hacer con ellos ──
# Analizar con y sin outliers para entender su impacto
media_con    = df['ventas_total'].mean()
media_sin    = df[df['ventas_total'] <= lim_sup]['ventas_total'].mean()
print(f"\nMedia CON outliers:  ${media_con:,.2f}")
print(f"Media SIN outliers:  ${media_sin:,.2f}")
print(f"Diferencia:          ${media_con - media_sin:,.2f} ({(media_con-media_sin)/media_sin*100:.1f}%)")
```

---

## FASE 6: Insights y Recomendaciones

### Para qué sirve
El EDA termina con conclusiones accionables. Esta es la fase donde traduces los números en lenguaje de negocio.

### Estructura de un insight bien formado
```
❌ Insight malo:
"Bleach tiene ventas altas"

✅ Insight bien formado:
OBSERVACIÓN:   Bleach representa el 68.7% del ingreso total ($7.6M)
CONTEXTO:      Los 3 segmentos principales concentran el 91.2% de ventas
IMPLICACIÓN:   Alta dependencia de un solo segmento = riesgo de concentración
RECOMENDACIÓN: Diversificar con campañas para Powder y Liquid & Gel
```

### Plantilla para documentar insights
```python
insights = {
    'hallazgo_1': {
        'titulo':          'Concentración extrema en Bleach',
        'observacion':     'Bleach representa 68.7% del ingreso ($7.6M)',
        'evidencia':       'Top 3 segmentos = 91.2% de ventas totales',
        'riesgo':          'Alta dependencia de un solo segmento y marca',
        'recomendacion':   'Campañas para diversificar hacia Powder y Liquid & Gel'
    },
    'hallazgo_2': {
        'titulo':          'Anomalía geográfica en Área 2',
        'observacion':     'Área 2 tiene transacciones similares a México (~18K vs ~21K)',
        'evidencia':       'Pero genera 4.6x menos ingresos ($1.2M vs $5.5M)',
        'riesgo':          'Ticket promedio muy bajo — oportunidad de crecimiento',
        'recomendacion':   'Estrategias de cross-selling y bundles en Áreas 1-6'
    },
    'hallazgo_3': {
        'titulo':          'Elasticidad precio-volumen',
        'observacion':     'Correlación de 0.92 entre unidades y valor de venta',
        'evidencia':       'Volúmenes >100 unidades solo ocurren a precios <$50',
        'riesgo':          'Mercado sensible al precio — márgenes bajo presión',
        'recomendacion':   'Explorar bundling para aumentar ticket sin bajar precio unitario'
    }
}

# Imprimir reporte de insights
print("=" * 60)
print("INSIGHTS DEL EDA")
print("=" * 60)
for key, insight in insights.items():
    print(f"\n {insight['titulo']}")
    print(f"   Observación:      {insight['observacion']}")
    print(f"   Evidencia:        {insight['evidencia']}")
    print(f"   Riesgo/Oportunidad: {insight['riesgo']}")
    print(f"   Recomendación:    {insight['recomendacion']}")
```

---

## Checklist completo del EDA

Antes de dar por terminado un EDA, verifica:
```
FASE 1: Entender el dataset
☐ Dimensiones conocidas (filas, columnas)
☐ Tipos de datos verificados y correctos
☐ Rango temporal identificado (si aplica)
☐ Nulos cuantificados por columna
☐ Duplicados verificados

FASE 2: Univariado
☐ Histograma + boxplot para cada numérica
☐ Frecuencia + % para cada categórica
☐ Media vs mediana comparadas
☐ Sesgo identificado

FASE 3: Bivariado
☐ Correlaciones entre numéricas calculadas
☐ Distribución por cada grupo categórico
☐ Tendencia temporal graficada

FASE 4: Multivariado
☐ Heatmap de correlaciones completo
☐ Al menos una tabla pivot o segmentación cruzada

FASE 5: Outliers
☐ Outliers detectados con IQR
☐ Origen de outliers investigado
☐ Decisión documentada (conservar/eliminar/separar)

FASE 6: Insights
☐ Mínimo 3 insights bien formados
☐ Cada insight con observación + evidencia + recomendación
☐ Preguntas sin responder documentadas para análisis futuro
```

---

## <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzMxY200dzhyc2RhbG5zbnk3MWZoZHE2ODZnM3I3Y2lseHNtYXI4diZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/WmvhTzEl9ihy9TqPme/giphy.gif" width="40"> Analogía con Medicina Tradicional China

El EDA es el equivalente al proceso de diagnóstico en MTC antes de cualquier tratamiento:

| EDA | Diagnóstico MTC |
|---|---|
| Fase 1: Entender el dataset | Primera consulta: historia clínica completa |
| Fase 2: Univariado | Evaluar cada síntoma por separado |
| Fase 3: Bivariado | Buscar relaciones entre síntomas |
| Fase 4: Multivariado | Ver el patrón completo del paciente |
| Fase 5: Outliers | Identificar síntomas que no encajan |
| Fase 6: Insights | Diagnóstico + protocolo de tratamiento |
| Modelar sin EDA | Tratar sin diagnosticar: siempre un error |

> Un buen diagnóstico toma tiempo.
> Un buen EDA también.
> Los atajos en ambos casos cuestan caro.

---

## ⚡ Cheat Sheet rápido
```
SIEMPRE al empezar:
→ df.shape, df.info(), df.describe(), df.isnull().sum()

FASE 2: por tipo de variable:
→ Numérica:    histograma + boxplot + media vs mediana
→ Categórica:  value_counts() + barplot + % acumulado

FASE 3: por combinación:
→ Num vs Num:  scatter + correlación de Pearson
→ Cat vs Num:  boxplot por grupos + groupby().agg()
→ Temporal:    lineplot + rolling mean

FASE 4:
→ Siempre: heatmap de correlaciones
→ Si hay grupos: pivot_table con heatmap

FASE 5:
→ IQR para detectar
→ Siempre preguntar: ¿por qué existe este outlier?
→ Analizar con y sin outliers

FASE 6: cada insight debe tener:
→ Observación (qué veo)
→ Evidencia (número que lo soporta)
→ Implicación (qué significa para el negocio)
→ Recomendación (qué hacer)
```

---

## 🔗 Aplicado a mis proyectos

### Proyecto 2: EDA Ventas Retail
```python
# Mi EDA siguió exactamente este flujo:

# FASE 1: Dataset de 122,002 transacciones
#         18 meses (ene 2022 - jul 2023)
#         0 nulos, 0 duplicados (ETL previo limpio)

# FASE 2: Univariado
#         → Media $90.51 vs Mediana $16.81 = sesgo positivo extremo
#         → Bleach: 68.7% del ingreso total
#         → México: 50% del ingreso total

# FASE 3: Bivariado
#         → Correlación ventas-unidades: 0.92 (muy fuerte positiva)
#         → Área 2 vs México: mismas transacciones, 4.6x menos ingreso
#         → Picos de ventas en jul 2022 y ene 2023

# FASE 4: Multivariado
#         → Bleach + México = combinación dominante
#         → Precio correlaciona negativamente con volumen (-0.12)

# FASE 5: Outliers
#         → 14,241 outliers (11.67%)
#         → Origen: Bleach + México (transacciones mayoristas)
#         → Decisión: conservar — representan negocio real

# FASE 6: 3 insights principales
#         → Riesgo de concentración (Cloralex/Bleach)
#         → Anomalía geográfica (Área 2)
#         → Elasticidad precio-volumen
```

### Posibles proyectos futuros de EDA

| Proyecto | Fase más interesante | Por qué |
|---|---|---|
| SEMARNAT emisiones | Fase 5: Outliers | Industrias gigantes vs típicas |
| OMS mortalidad | Fase 3: Bivariado | Correlación PIB vs salud |
| INEGI ingresos | Fase 2: Univariado | Sesgo extremo por desigualdad |
| Kaggle fraude | Fase 5: Outliers | Los outliers SON el fraude |
| IMSS consultas | Fase 4: Multivariado | Patrón edad + enfermedad + región |

---

*Nota creada: 2026 · Parte del certificado Científico de Datos — EBAC*
*🪷 github.com/ReginaPema*``
