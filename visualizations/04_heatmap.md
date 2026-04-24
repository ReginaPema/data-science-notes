# Heatmap de Correlación
## Mapa de Calor — Todas las relaciones de un vistazo

> El heatmap de correlación es el scatter plot de todas las variables al mismo tiempo.
> En lugar de hacer n×n gráficas individuales, ves todas las relaciones en una sola imagen codificada en color.

> **Regla de oro:**
> El heatmap es siempre la última gráfica del análisis bivariado,
> se hace después de entender cada variable por separado.
> Es el resumen visual de todas las correlaciones.

---

## 🧱 Anatomía de un heatmap de correlación

```
         Var1   Var2   Var3   Var4   Var5
        ┌──────┬──────┬──────┬──────┬──────┐
  Var1  │ 1.00 │ 0.92 │-0.12 │ 0.37 │ 0.65 │
        ├──────┼──────┼──────┼──────┼──────┤
  Var2  │ 0.92 │ 1.00 │-0.04 │ 0.43 │ 0.61 │
        ├──────┼──────┼──────┼──────┼──────┤
  Var3  │-0.12 │-0.04 │ 1.00 │-0.22 │-0.07 │
        ├──────┼──────┼──────┼──────┼──────┤
  Var4  │ 0.37 │ 0.43 │-0.22 │ 1.00 │ 0.16 │
        ├──────┼──────┼──────┼──────┼──────┤
  Var5  │ 0.65 │ 0.61 │-0.07 │ 0.16 │ 1.00 │
        └──────┴──────┴──────┴──────┴──────┘

  Cada celda = correlación entre dos variables
  Color rojo intenso  = correlación positiva fuerte (+1)
  Color azul intenso  = correlación negativa fuerte (-1)
  Color blanco/neutro = sin correlación (0)
  Diagonal principal  = siempre 1.0 (variable consigo misma)
```

| Elemento | Qué representa |
|---|---|
| Filas y columnas | Las variables numéricas del dataset |
| Cada celda | El coeficiente de correlación entre dos variables |
| Color de la celda | La fuerza y dirección de la correlación |
| Número dentro | El valor exacto del coeficiente r |
| Diagonal principal | Siempre 1.0 → cada variable correlaciona perfectamente consigo misma |
| Triángulo superior = inferior | El heatmap es simétrico → se puede mostrar solo la mitad |

---

## 🐍 Cómo hacerlo en Python

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ── PASO 1: CALCULAR LA MATRIZ DE CORRELACIÓN ──
# Solo con columnas numéricas
numericas = df.select_dtypes(include='number')
correlaciones = numericas.corr()

print("Matriz de correlación:")
print(correlaciones.round(2))

# ── HEATMAP BÁSICO ──
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(correlaciones,
            annot=True,          # mostrar números en celdas
            fmt='.2f',           # formato: 2 decimales
            cmap='RdBu_r',       # rojo=positivo, azul=negativo
            center=0,            # blanco en 0
            square=True,         # celdas cuadradas
            linewidths=0.5,      # líneas entre celdas
            ax=ax)

ax.set_title('Matriz de Correlación', fontsize=14)
plt.tight_layout()
plt.savefig('images/heatmap_basico.png', dpi=150)
plt.show()

# ── HEATMAP PROFESIONAL: solo triángulo inferior ──
fig, ax = plt.subplots(figsize=(12, 10))

# Máscara para ocultar triángulo superior (redundante)
mask = np.triu(np.ones_like(correlaciones, dtype=bool))

sns.heatmap(correlaciones,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            mask=mask,                     # ocultar triángulo superior
            square=True,
            linewidths=0.5,
            linecolor='white',
            cbar_kws={
                'label': 'Correlación de Pearson',
                'shrink': 0.8
            },
            annot_kws={'size': 10},        # tamaño del número
            vmin=-1, vmax=1,               # escala fija
            ax=ax)

ax.set_title('Matriz de Correlación entre Variables Numéricas',
             fontsize=14, pad=20)
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
plt.savefig('images/heatmap_profesional.png', dpi=150)
plt.show()

# ── IDENTIFICAR CORRELACIONES FUERTES ──
print("\nCorrelaciones fuertes (|r| > 0.5):")
print("-" * 50)

pares_vistos = set()

for col1 in correlaciones.columns:
    for col2 in correlaciones.columns:
        if col1 != col2:
            par = tuple(sorted([col1, col2]))
            if par not in pares_vistos:
                r = correlaciones.loc[col1, col2]
                if abs(r) > 0.5:
                    pares_vistos.add(par)
                    if r > 0:
                        direccion = "↑↑ positiva"
                        emoji = "🔴"
                    else:
                        direccion = "↑↓ negativa"
                        emoji = "🔵"
                    print(f"{emoji} {col1} ↔ {col2}")
                    print(f"   r = {r:.3f} | {direccion}")
                    print()

# ── HEATMAP DE TABLA PIVOT ──
# No solo para correlaciones — también para
# cruzar dos variables categóricas
tabla = df.pivot_table(
    values='ventas_total',
    index='region',
    columns='segmento',
    aggfunc='sum'
) / 1e6  # en millones

fig, ax = plt.subplots(figsize=(12, 6))

sns.heatmap(tabla,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',       # amarillo=bajo, rojo=alto
            linewidths=0.5,
            linecolor='white',
            cbar_kws={'label': 'Ventas ($M)'},
            ax=ax)

ax.set_title('Ventas ($M) por Región y Segmento', fontsize=14)
ax.set_xlabel('Segmento')
ax.set_ylabel('Región')
plt.tight_layout()
plt.savefig('images/heatmap_pivot.png', dpi=150)
plt.show()

# ── HEATMAP TEMPORAL ──
# Ventas por mes y año — patrón estacional
tabla_tiempo = df.pivot_table(
    values='ventas_total',
    index='mes',
    columns='año',
    aggfunc='sum'
) / 1e6

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(tabla_tiempo,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            linewidths=0.5,
            linecolor='white',
            cbar_kws={'label': 'Ventas ($M)'},
            ax=ax)

# Nombres de meses en español
meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
         'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
ax.set_yticklabels(meses, rotation=0)
ax.set_title('Ventas ($M) por Mes y Año\n(Detección de Estacionalidad)',
             fontsize=14)
ax.set_xlabel('Año')
ax.set_ylabel('Mes')
plt.tight_layout()
plt.savefig('images/heatmap_temporal.png', dpi=150)
plt.show()
```

---

## 🎨 Paletas de colores: cuándo usar cada una

```python
# La elección del colormap cambia completamente
# cómo se lee el heatmap

# ── PARA CORRELACIONES (divergente: tiene centro) ────
cmap='RdBu_r'      # rojo=positivo, azul=negativo (✓ más usado)
cmap='coolwarm'    # rojo=positivo, azul=negativo (más suave)
cmap='PiYG'        # verde=positivo, rosa=negativo
cmap='RdYlGn'      # verde=positivo, rojo=negativo

# ── PARA MAGNITUDES (secuencial: sin centro) ──
cmap='YlOrRd'      # amarillo→naranja→rojo (valores altos=rojo) ✓
cmap='Blues'       # blanco→azul (valores altos=azul oscuro)
cmap='Greens'      # blanco→verde
cmap='viridis'     # morado→verde→amarillo (daltonismo-friendly) ✓

# ── REGLA PRÁCTICA ──
# Correlaciones (van de -1 a +1, tienen cero como neutro)
#   → Usa colormap DIVERGENTE (RdBu_r, coolwarm)
#   → Siempre pon center=0

# Magnitudes (ventas, temperatura, frecuencia)
#   → Usa colormap SECUENCIAL (YlOrRd, Blues, viridis)
#   → No uses center

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titulos  = ['RdBu_r (correlaciones)', 'YlOrRd (magnitudes)', 'viridis (universal)']
colormaps = ['RdBu_r', 'YlOrRd', 'viridis']

for ax, cmap_name, titulo in zip(axes, colormaps, titulos):
    sns.heatmap(correlaciones,
                annot=True, fmt='.2f',
                cmap=cmap_name,
                center=0 if cmap_name == 'RdBu_r' else None,
                square=True,
                linewidths=0.5,
                ax=ax)
    ax.set_title(titulo)

plt.tight_layout()
plt.show()
```

---

## 🧭 Cómo leer un heatmap: los patrones clave

### Leer los colores primero

```
Color intenso ROJO  →  r cercano a +1      →  correlación positiva fuerte
Color suave ROJO    →  r entre 0.3-0.7     →  correlación positiva moderada
Color BLANCO/NEUTRO →  r cercano a 0       →  sin correlación lineal
Color suave AZUL    →  r entre -0.3 y -0.7 → correlación negativa moderada
Color intenso AZUL  →  r cercano a -1      →  correlación negativa fuerte
```

### Identificar grupos de variables relacionadas

```
Si ves un bloque de celdas rojas juntas:
→ Ese grupo de variables se mueve junto
→ Puede indicar multicolinealidad (problema en ML: las variables dicen lo mismo)
→ O puede indicar un factor común que las explica a todas

Si ves una variable con toda su fila/columna azul:
→ Esa variable tiene relación INVERSA con todas las demás
→ Muy relevante para análisis de riesgo

Si ves una variable con toda su fila/columna blanca:
→ Esa variable es INDEPENDIENTE de todas las demás
→ Aporta información única al análisis
```

### Los 5 patrones más importantes

```python
# PATRÓN 1: Bloque diagonal rojo
# → Grupo de variables muy correlacionadas entre sí
# → En ML: posible multicolinealidad, considera eliminar algunas

# PATRÓN 2: Una variable con columna/fila toda azul
# → Variable que va en dirección opuesta a las demás
# → Muy útil para análisis de riesgo o diversificación

# PATRÓN 3: Una variable con columna/fila toda blanca
# → Variable independiente, aporta información única
# → Valiosa para modelos de ML, reduce redundancia

# PATRÓN 4: Correlaciones simétricas evidentes
# → El heatmap es un cuadrado simétrico por definición
# → Si r(A,B) = 0.92, entonces r(B,A) = 0.92
# → Por eso se muestra solo el triángulo inferior

# PATRÓN 5: Diagonal siempre = 1.0
# → Cada variable correlaciona perfectamente consigo misma
# → Se puede ocultar para no distraer
```

---

## 🔍 Heatmap vs Scatter: cuándo usar cada uno

| Situación | Mejor opción |
|---|---|
| Ver relación entre 2 variables en detalle | Scatter Plot |
| Ver si la relación es lineal o curva | Scatter Plot |
| Ver relación entre MUCHAS variables a la vez | Heatmap |
| Comunicar resultados a audiencia no técnica | Heatmap |
| Identificar qué variables analizar primero | Heatmap |
| Ver outliers que distorsionan correlación | Scatter Plot |
| Cruzar dos variables categóricas por magnitud | Heatmap Pivot |
| Detectar estacionalidad temporal | Heatmap Temporal |

**El flujo recomendado:**
```
Heatmap primero  →  identifica qué pares son interesantes
Scatter después  →  investiga cada par importante en detalle
```

---

## 🎯 Los 3 tipos de heatmap en Ciencia de Datos

### Tipo 1: Correlación (el más común en EDA)
Para ver relaciones entre variables numéricas.
```python
correlaciones = df.select_dtypes(include='number').corr()
sns.heatmap(correlaciones, cmap='RdBu_r', center=0,
            annot=True, fmt='.2f')
```

### Tipo 2: Pivot Table (para cruzar categorías)
Para ver magnitudes cruzando dos categóricas.
```python
tabla = df.pivot_table(values='ventas',
                       index='region',
                       columns='segmento',
                       aggfunc='sum')
sns.heatmap(tabla, cmap='YlOrRd', annot=True, fmt='.0f')
```

### Tipo 3: Temporal (para detectar estacionalidad)
Para ver patrones cíclicos en el tiempo.
```python
tabla = df.pivot_table(values='ventas',
                       index='mes',
                       columns='año',
                       aggfunc='sum')
sns.heatmap(tabla, cmap='Blues', annot=True, fmt='.2f')
```

---

## ❌ Errores comunes

**Error 1: Usar colormap secuencial para correlaciones**
Si usas `Blues` o `YlOrRd` para correlaciones, no puedes distinguir correlaciones positivas
de negativas, todo se ve como magnitud. Siempre usa colormap divergente con `center=0`.

```python
# ✗ Incorrecto para correlaciones
sns.heatmap(correlaciones, cmap='YlOrRd')

# ✓ Correcto para correlaciones
sns.heatmap(correlaciones, cmap='RdBu_r', center=0)
```

**Error 2: No poner `annot=True`**
Un heatmap sin números es bonito pero poco útil para análisis.
El color te dice la dirección, el número te dice la magnitud exacta.

**Error 3: No enmascarar el triángulo superior**
El heatmap es simétrico. Mostrar ambos triángulos dobla la información sin agregar valor.
Usa `mask=np.triu(...)` para mostrar solo el triángulo inferior.

**Error 4: Incluir variables no numéricas**
La correlación de Pearson solo funciona con variables numéricas continuas.
Variables binarias (0/1) pueden incluirse, pero variables categóricas de texto
deben excluirse o codificarse primero.

```python
# ✗ Error común
correlaciones = df.corr()  # falla si hay columnas de texto

# ✓ Correcto
numericas = df.select_dtypes(include='number')
correlaciones = numericas.corr()
```

**Error 5: Confundir correlación con causalidad**
El heatmap te muestra que dos variables se mueven juntas, no por qué.
La interpretación causal siempre viene del conocimiento del dominio, no del coeficiente.

**Error 6: Heatmap con demasiadas variables**
Con más de 15-20 variables, el heatmap se vuelve ilegible. 
Selecciona las más relevantes o agrúpalas.

```python
# Con muchas variables, selecciona las más relevantes
# Opción 1: selección manual
cols_relevantes = ['ventas_total', 'precio_unitario',
                   'unidades_vendidas', 'var_pct']
correlaciones = df[cols_relevantes].corr()

# Opción 2: filtrar solo correlaciones fuertes con target
target = 'ventas_total'
correlaciones_con_target = (df.select_dtypes(include='number')
                              .corr()[target]
                              .abs()
                              .sort_values(ascending=False))
top_vars = correlaciones_con_target.head(10).index
correlaciones = df[top_vars].corr()
```

---

## <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzMxY200dzhyc2RhbG5zbnk3MWZoZHE2ODZnM3I3Y2lseHNtYXI4diZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/WmvhTzEl9ihy9TqPme/giphy.gif" width="40"> Analogía con Medicina Tradicional China

El heatmap de correlación es como el mapa de meridianos en MTC:

| Heatmap | Sistema de Meridianos MTC |
|---|---|
| Cada celda | Punto de intersección entre dos meridianos |
| Color rojo intenso | Conexión energética fuerte → los órganos se afectan mutuamente |
| Color azul intenso | Relación inversa → cuando uno sube el otro baja (Yin-Yang) |
| Color blanco | Sin conexión directa → sistemas independientes |
| Bloque de rojos | Grupo de meridianos del mismo elemento (Fuego, Tierra, Metal...) |
| Variable independiente (blanco) | Meridiano con función autónoma |
| Diagonal = 1.0 | Cada meridiano se conecta perfectamente consigo mismo |

En MTC, un diagnóstico complejo requiere ver cómo todos los sistemas se relacionan
simultáneamente, exactamente lo que hace el heatmap con las variables.

> El mapa de meridianos es el heatmap del cuerpo humano.

---

## ⚡ Cheat Sheet rápido

```
PARA CORRELACIONES:
→ cmap='RdBu_r' + center=0
→ annot=True, fmt='.2f'
→ mask triángulo superior
→ square=True

PARA MAGNITUDES (pivot table):
→ cmap='YlOrRd' o 'Blues'
→ Sin center
→ fmt depende de la escala (.2f, .0f, '.1%')

LEER EL HEATMAP:
→ Rojo intenso    = correlación positiva fuerte (r > 0.7)
→ Azul intenso    = correlación negativa fuerte (r < -0.7)
→ Blanco / neutro = sin correlación (r ≈ 0)

BUSCAR SIEMPRE:
→ ¿Hay bloques de rojo? → multicolinealidad o factor común
→ ¿Hay filas/columnas blancas? → variable independiente — valiosa
→ ¿Hay filas/columnas azules? → variable inversa — útil para riesgo
→ ¿Cuáles variables correlacionan con mi variable objetivo?

FLUJO RECOMENDADO:
1. Heatmap → identificar pares interesantes
2. Scatter → investigar cada par en detalle
3. Conclusión → correlación + dirección + posible explicación

NUNCA:
→ Asumir causalidad por correlación alta
→ Incluir variables categóricas de texto
→ Usar colormap secuencial para correlaciones
→ Omitir los números (annot=True siempre)
```

---

## 🔗 Aplicado a mis proyectos

### Proyecto EDA: Ventas Retail

```python
# Lo que encontré en mi heatmap:

# CORRELACIONES FUERTES:
# r = 0.92  TOTAL_VALUE_SALES ↔ TOTAL_UNIT_SALES
#   → A más unidades, más ingreso total
#   → Relación directa y predecible
#   → Base para modelo de regresión lineal futuro

# r = 0.83  VAR_PCT ↔ ABOVE_AVG
#   → Las transacciones de alto porcentaje son casi siempre las de valor superior al promedio
#   → Variables redundantes: dicen lo mismo

# r = -0.78 VAR_WEEKLY_AVG ↔ TOTAL_UNIT_AVG_WEEKLY_SALES
#   → Correlación negativa fuerte
#   → A mayor promedio semanal de unidades, menor variación relativa,
      los grandes compradores son más consistentes

# r = 0.68  PRICE ↔ HIGH_VALUE
#   → Los productos de precio alto son casi siempre los de alto valor total
#   → Confirma que el precio unitario es driver del valor de la transacción

# VARIABLES INDEPENDIENTES (r ≈ 0 con todo):
# SIZE_NUM: el tamaño del producto no correlaciona fuertemente con nada
# → Aporta información única al análisis
# → Útil como feature independiente en ML

# HEATMAP PIVOT que haría en próximo análisis:
tabla_region_segmento = df.pivot_table(
    values='ventas_total',
    index='region',
    columns='segmento',
    aggfunc='sum'
)
# → Ver qué segmento domina en cada región
# → Identificar oportunidades de crecimiento en combinaciones región-segmento débiles
```

### Posibles proyectos futuros
| Dataset | Tipo de heatmap | Insight esperado |
|---|---|---|
| SEMARNAT | Correlación + Temporal | Emisiones vs PIB; estacionalidad industrial |
| OMS | Correlación | PIB vs salud vs educación; factores de desarrollo |
| INEGI | Pivot región×ingreso | Desigualdad geográfica en México |
| Kaggle fraude | Correlación | Qué variables predicen mejor el fraude |
| IMSS | Temporal mes×año | Estacionalidad de enfermedades |

---

*Nota creada: 2026 · Serie: Visualizaciones para Ciencia de Datos*
*Parte del certificado Científico de Datos — EBAC*
*🪷 github.com/ReginaPema*
