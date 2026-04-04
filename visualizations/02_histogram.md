# Histogram
## Histograma

> El histograma es la radiografía de tus datos.
> Te muestra exactamente cómo se distribuyen los valores de una variable:
> dónde se concentran, dónde son escasos, y qué forma tiene esa distribución.

---

## 🧱 Anatomía de un histograma
```
Frecuencia
    │
    │     ██
    │    ████
    │   ██████
    │  ████████
    │ ██████████
    │████████████░░
    │████████████░░░░░░
    └─────────────────────→ Valor de la variable
    ↑           ↑      ↑
  Mínimo     Moda     Máximo
  
  Cada barra = un rango de valores (bin)
  Altura de la barra = cuántos datos caen en ese rango
```

| Elemento | Nombre técnico | Qué significa |
|---|---|---|
| Eje X | Variable analizada | Los valores posibles de tu variable |
| Eje Y | Frecuencia / Conteo | Cuántos datos tienen ese valor |
| Cada barra | Bin (intervalo) | Un rango de valores agrupados |
| Ancho de barra | Tamaño del bin | Qué tan granular es el análisis |
| Altura de barra | Frecuencia | Cuántos datos caen en ese intervalo |
| Pico más alto | Moda | El valor más común en los datos |

---

## 🐍 Cómo hacerlo en Python
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ── HISTOGRAMA BÁSICO CON MATPLOTLIB ──
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(df['ventas_total'],
        bins=50,                    # número de barras
        color='steelblue',
        edgecolor='white',          # borde entre barras
        alpha=0.8)                  # transparencia

ax.set_title('Distribución de Ventas Totales', fontsize=14)
ax.set_xlabel('Ventas ($)')
ax.set_ylabel('Frecuencia (número de transacciones)')
plt.tight_layout()
plt.show()

# ── CON LÍNEAS DE MEDIA Y MEDIANA ──
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(df['ventas_total'], bins=50,
        color='steelblue', edgecolor='white', alpha=0.8)

media   = df['ventas_total'].mean()
mediana = df['ventas_total'].median()

ax.axvline(media, color='red', linewidth=2,
           linestyle='--', label=f'Media: ${media:,.2f}')
ax.axvline(mediana, color='orange', linewidth=2,
           linestyle='--', label=f'Mediana: ${mediana:,.2f}')

ax.set_title('Distribución de Ventas con Media y Mediana')
ax.set_xlabel('Ventas ($)')
ax.set_ylabel('Frecuencia')
ax.legend()
plt.tight_layout()
plt.savefig('images/histograma_ventas.png', dpi=150)
plt.show()

# ── CON SEABORN (agrega curva de densidad KDE) ──
fig, ax = plt.subplots(figsize=(10, 6))

sns.histplot(data=df,
             x='ventas_total',
             bins=50,
             kde=True,              # ← curva de densidad
             color='steelblue',
             ax=ax)

ax.set_title('Distribución de Ventas con Curva de Densidad')
ax.set_xlabel('Ventas ($)')
plt.tight_layout()
plt.show()

# ── ESCALA LOGARÍTMICA (para datos muy sesgados) ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Normal
axes[0].hist(df['ventas_total'], bins=50,
             color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_title('Escala normal — difícil de leer')
axes[0].set_xlabel('Ventas ($)')

# Log
axes[1].hist(df['ventas_total'], bins=50,
             color='teal', edgecolor='white', alpha=0.8,
             log=True)              # ← eje Y en log
axes[1].set_title('Escala log: mucho más informativa')
axes[1].set_xlabel('Ventas ($)')

plt.tight_layout()
plt.savefig('images/histograma_escala_log.png', dpi=150)
plt.show()

# ── MÚLTIPLES VARIABLES JUNTAS ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

variables = ['ventas_total', 'precio_unitario',
             'unidades_vendidas', 'ingreso_promedio']

for ax, var in zip(axes.flatten(), variables):
    sns.histplot(data=df, x=var, bins=40, kde=True,
                 color='steelblue', ax=ax)
    media_var   = df[var].mean()
    mediana_var = df[var].median()
    ax.axvline(media_var, color='red', ls='--',
               label=f'Media: {media_var:,.1f}')
    ax.axvline(mediana_var, color='orange', ls='--',
               label=f'Mediana: {mediana_var:,.1f}')
    ax.set_title(f'Distribución — {var}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('images/histogramas_multiples.png', dpi=150)
plt.show()

# ── HISTOGRAMA POR GRUPOS (superpuestos) ──
fig, ax = plt.subplots(figsize=(12, 6))

grupos = df['segmento'].unique()
colores = ['steelblue', 'teal', 'coral',
           'gold', 'purple', 'green', 'orange']

for grupo, color in zip(grupos, colores):
    datos_grupo = df[df['segmento'] == grupo]['ventas_total']
    ax.hist(datos_grupo, bins=40, alpha=0.5,
            label=grupo, color=color, edgecolor='none')

ax.set_title('Distribución de Ventas por Segmento')
ax.set_xlabel('Ventas ($)')
ax.set_ylabel('Frecuencia')
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.savefig('images/histograma_por_grupo.png', dpi=150)
plt.show()
```

---

## 🎛️ El parámetro más importante: 'bins'

El número de bins (barras) cambia completamente lo que ves. No hay un número perfecto, depende de tus datos.
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

bins_opciones = [10, 50, 200]
titulos = ['Pocos bins (10) → muy general',
           'Bins moderados (50) → equilibrado',
           'Muchos bins (200) → muy detallado']

for ax, bins, titulo in zip(axes, bins_opciones, titulos):
    ax.hist(df['ventas_total'], bins=bins,
            color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_title(titulo)
    ax.set_xlabel('Ventas ($)')

plt.tight_layout()
plt.show()
```
```
Pocos bins:      Muchos bins:     Equilibrado:
████             █ █ █ █ █        ██
████████         ████ ██ █        ████
████████████     ████████████     ██████
                                  ████████

Pierdes detalle  Demasiado ruido  Puedes leer la forma
```

**Regla práctica para elegir bins:**
- Menos de 1,000 filas → 20–30 bins
- 1,000 a 10,000 filas → 30–50 bins
- Más de 10,000 filas → 50–100 bins
- Siempre prueba 2-3 valores y quédate con el más claro

---

## 🧭 Cómo leer un histograma: las 6 formas

### Forma 1: Distribución Normal (campana)
```
        ████
      ████████
    ████████████
  ████████████████
████████████████████
```
**Qué significa:**
Los datos se distribuyen simétricamente alrededor del centro. La mayoría está cerca de la media. 
Media ≈ Mediana ≈ Moda

**Ejemplos reales:**
- Altura de personas adultas
- Error de medición en instrumentos calibrados
- Notas de un examen bien diseñado
- Temperatura diaria en un mes estable

**Insight:** El promedio es confiable y representativo. Puedes usarlo en reportes con confianza.

---

### Forma 2: Sesgo positivo (cola a la derecha)
```
████
████░░
████░░░░░
████░░░░░░░░░░
```
**Qué significa:**
La mayoría de los valores son bajos, pero hay una cola larga hacia la derecha con valores altos poco frecuentes.
Media > Mediana

**Ejemplos reales:**
- Ingresos y salarios
- Ventas por transacción *(mi proyecto EDA)*
- Precios de vivienda
- Tiempo de espera en servicios
- Emisiones de CO₂ por empresa

**Insight:**
La media sobreestima al caso típico. Usa la mediana para describir al cliente o caso "normal".
Investiga la cola derecha, puede ser tu segmento más valioso.

---

### Forma 3: Sesgo negativo (cola a la izquierda)
```
              ████
          ░░░░████
      ░░░░░░░░████
░░░░░░░░░░░░░░████
```
**Qué significa:**
La mayoría de los valores son altos, pero hay una cola hacia la izquierda con valores bajos poco frecuentes.
Media < Mediana

**Ejemplos reales:**
- Satisfacción del cliente en servicio excelente
- Edad de jubilación (mayoría llega a 65+)
- Calificaciones en app con buenas reseñas
- Tiempo de vida de productos de alta calidad

**Insight:**
La mayoría está satisfecha o tiene buen desempeño. Los valores bajos son excepciones que merecen atención especial.

---

### Forma 4: Distribución uniforme (plana)
```
████████████████████
████████████████████
████████████████████
```
**Qué significa:**
Todos los valores tienen aproximadamente la misma frecuencia. No hay un valor más común que otro.

**Ejemplos reales:**
- Resultado de un dado justo
- Día del mes en que nace la gente
- Últimos dígitos de números de teléfono

**Insight:**
No hay patrón dominante. Si esperas una distribución normal y obtienes una uniforme, puede indicar datos aleatorios o sin estructura.

---

### Forma 5: Bimodal (dos picos)
```
████        ████
████████  ████████
████████████████████
```
**Qué significa:**
Hay dos grupos distintos mezclados en el mismo análisis. Dos modas, dos comportamientos diferentes.

**Ejemplos reales:**
- Tickets de compra: minorista + mayorista
- Edades: empleados jóvenes + senior
- Tiempo de ejercicio: sedentarios + deportistas
- Dosis de medicamento: dos protocolos

**Insight:**
Señal de alerta: tienes dos poblaciones mezcladas. Segmenta antes de analizar. 
La media no representa a ninguno de los dos grupos.
```python
# Detectar bimodalidad
from scipy.signal import find_peaks

counts, bins = np.histogram(df['columna'], bins=50)
peaks, _ = find_peaks(counts, height=counts.max()*0.3)
print(f"Picos detectados: {len(peaks)}")
if len(peaks) > 1:
    print("Posible distribución bimodal. Considera segmentar")
```

---

### Forma 6: Con pico muy alto y cola muy larga
```
██
██░
██░░
██░░░
██░░░░░░░░░░░░░░░░░
```
**Qué significa:**
Distribución con curtosis alta. La inmensa mayoría está concentrada en un rango muy pequeño, pero hay valores extremadamente altos.
Típico cuando hay outliers muy extremos.

**Ejemplos reales:**
- Tiempo entre terremotos (mayoría: calma, pocos: catástrofe)
- Ventas virales en redes sociales
- Retornos financieros extremos

**Insight:**
La escala normal hace que todo se vea aplastado. Usa escala logarítmica para leer la forma real.

---

## 📊 Histograma vs Boxplot — cuándo usar cada uno

| Necesito saber | Usa |
|---|---|
| La forma exacta de la distribución | Histograma |
| Si hay outliers específicos | Boxplot |
| Comparar distribución entre muchos grupos | Boxplot |
| Ver si hay dos picos (bimodal) | Histograma |
| Los percentiles exactos (Q1, Q3) | Boxplot |
| Dónde se concentran la mayoría de valores | Histograma |
| Resumen rápido en poco espacio | Boxplot |
| Detalle completo de la distribución | Histograma |

**La regla práctica:**
Empieza siempre con el histograma para ver la forma completa.
Luego usa el boxplot para comparar grupos.
```python
# El combo perfecto: úsalos juntos
fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                          gridspec_kw={'height_ratios': [3, 1]})

# Histograma arriba
sns.histplot(data=df, x='ventas_total',
             bins=50, kde=True,
             color='steelblue', ax=axes[0])
axes[0].axvline(df['ventas_total'].mean(),
                color='red', ls='--', label='Media')
axes[0].axvline(df['ventas_total'].median(),
                color='orange', ls='--', label='Mediana')
axes[0].set_title('Distribución de Ventas')
axes[0].legend()

# Boxplot abajo (mismo eje X)
axes[1].boxplot(df['ventas_total'],
                vert=False,           # horizontal
                widths=0.5)
axes[1].set_xlabel('Ventas ($)')
axes[1].set_yticks([])

plt.tight_layout()
plt.savefig('images/histograma_boxplot_combo.png', dpi=150)
plt.show()
```

---

## ❌ Errores comunes

**Error 1: No ajustar los bins**
El número de bins por default rara vez es el mejor. Siempre prueba varios valores antes de decidir.

**Error 2: No usar escala log con datos sesgados**
Si tus datos tienen outliers extremos, el histograma normal va a mostrar una sola barra alta y todo lo demás aplastado. Cambia a escala log.

**Error 3: Confundir frecuencia con porcentaje**
El eje Y puede mostrar conteo absoluto o porcentaje (densidad). Asegúrate de saber cuál es.
```python
# Frecuencia absoluta (default)
ax.hist(df['ventas'], bins=50)

# Densidad (porcentaje) → suma a 1
ax.hist(df['ventas'], bins=50, density=True)
```

**Error 4: Superponer muchos grupos**
Con más de 3-4 grupos, el histograma superpuesto
se vuelve ilegible. Usa FacetGrid o boxplot mejor.
```python
# ❌ Ilegible con muchos grupos
for grupo in df['segmento'].unique():  # 7 grupos = caos visual
    ax.hist(df[df['segmento']==grupo]['ventas'], alpha=0.5)

# ✅ Mejor: FacetGrid de Seaborn
g = sns.FacetGrid(df, col='segmento', col_wrap=3, height=3)
g.map(sns.histplot, 'ventas_total', bins=30)
g.set_titles('{col_name}')
plt.tight_layout()
```

**Error 5: No poner etiquetas**
Un histograma sin título, sin unidades en el eje X y sin indicar qué representa el eje Y es inútil para cualquier audiencia.

---

## <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzMxY200dzhyc2RhbG5zbnk3MWZoZHE2ODZnM3I3Y2lseHNtYXI4diZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/WmvhTzEl9ihy9TqPme/giphy.gif" width="40"> Analogía con Medicina Tradicional China

El histograma es como el mapa de distribución de síntomas en una población de pacientes:

| Histograma | Epidemiología / MTC |
|---|---|
| Eje X: valores de la variable | Intensidad del síntoma (leve → severo) |
| Eje Y: frecuencia | Cuántos pacientes tienen esa intensidad |
| Pico alto (moda) | Presentación más común del patrón |
| Cola larga a la derecha | Casos severos, menos frecuentes pero críticos |
| Distribución bimodal | Dos síndromes distintos mezclados en el diagnóstico |
| Distribución normal | Patrón clásico bien definido |

Un epidemiólogo mirando la distribución de intensidad de dolor en 1,000 pacientes está leyendo exactamente un histograma mental.

---

## ⚡ Cheat Sheet rápido
```
FORMA DEL HISTOGRAMA → LO QUE SIGNIFICA

Campana simétrica    →  Distribución normal  → Media confiable
Cola derecha larga   →  Sesgo positivo       → Usa mediana, investiga outliers altos
Cola izquierda larga →  Sesgo negativo       → Investiga valores bajos excepcionales
Plano / uniforme     →  Sin patrón dominante → Datos aleatorios o sin estructura
Dos picos            →  Bimodal              → Segmenta: tienes dos poblaciones mezcladas
Pico alto + cola     →  Outliers extremos    → Usa escala logarítmica

BINS:
Muy pocos  →  Pierdes detalle de la forma
Muy muchos →  Demasiado ruido, difícil de leer
Regla práctica: empieza con 50 y ajusta

CUÁNDO USAR ESCALA LOG:
→ Cuando media >> mediana (sesgo extremo)
→ Cuando hay outliers 10x o más grandes que la mediana
→ Cuando la mayoría de las barras se ven aplastadas

SIEMPRE INCLUIR:
→ Línea de media (roja)
→ Línea de mediana (naranja)
→ Título descriptivo
→ Etiquetas en ambos ejes
→ Unidades en el eje X
```

---

## 🔗 Aplicado a mis proyectos

### Proyecto EDA: Ventas Retail
```python
# Lo que encontré con el histograma:

# Histograma de ventas_total:
# → Barra altísima cerca de $0-100
# → Cola larguísima hasta $12,236
# → Media $90.51 vs Mediana $16.81
# → Forma: sesgo positivo extremo
# → Necesitó escala log para ser legible
# → Confirmó: mercado dominado por transacciones pequeñas frecuentes + pocas transacciones mayoristas enormes

# Histograma de unidades_vendidas:
# → Forma muy similar al de ventas
# → Media 3.21 uds vs Mediana 0.37 uds
# → Mayoría de transacciones: menos de 1 unidad (fracciones, productos vendidos por peso o volumen)
# → Cola hasta 500 unidades por transacción
```

### Posibles proyectos futuros
| Dataset | Forma esperada | Por qué |
|---|---|---|
| INEGI ingresos hogar | Sesgo positivo extremo | Desigualdad económica México |
| OMS esperanza de vida | Sesgo negativo leve | Mayoría países con expectativa alta |
| SEMARNAT emisiones | Sesgo positivo + bimodal posible | Industrias pequeñas vs gigantes |
| Kaggle fraude | Pico enorme + cola mínima | 99%+ son transacciones normales |
| Calificaciones PISA | Aproximadamente normal | Diseñada así intencionalmente |

---

*Nota creada: 2026 · Serie: Visualizaciones para Ciencia de Datos*
*Parte del certificado Científico de Datos — EBAC*
*🪷 github.com/ReginaPema*
