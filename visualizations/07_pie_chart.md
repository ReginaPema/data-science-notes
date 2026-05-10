# <img src="https://img.icons8.com/?size=50&id=P6lFKH1LrQnG&format=png&color=000000" align="center"/> Pie Chart
## Gráfica de Pay → Participación porcentual

> **La gráfica más polémica en Ciencia de Datos.**
> Los expertos en visualización casi universalmente recomiendan usarla poco, o no usarla.
> Pero existe, se usa en presentaciones ejecutivas, y necesitas saber cuándo es válida 
> y cuándo debes sustituirla por un bar chart.

> **Regla de oro:**
> El pie chart solo tiene sentido cuando quieres mostrar que algo es "la mitad", "un tercio" o "casi todo".
> Para cualquier otra comparación, el bar chart es más preciso.

---

## <img src="https://img.icons8.com/?size=40&id=81170&format=png&color=000000" align="center"/> Anatomía de un pie chart

```
            Categoría A
               30%
          ___________
        /      A      \
       /   ___________.\
      | B /            |\ C
      |  /      🥧     | \
      | /               |  \
       \        D       /
        \______________/
              D
             25%

Elementos:
* Círculo completo  = 100% del total
* Cada sector       = una categoría
* Ángulo del sector = proporción de esa categoría
* Suma de todos     = siempre 100%
```

| Elemento | Qué representa |
|---|---|
| Círculo completo | El 100%, el todo que se está dividiendo |
| Cada sector (slice) | Una categoría y su proporción del total |
| Ángulo del sector | La participación porcentual |
| Etiqueta | El nombre y/o porcentaje de cada sector |
| Separación (explode) | Énfasis en un sector específico |

---

## <img src="https://img.icons8.com/?size=40&id=121464&format=png&color=000000" align="center"/> Cómo hacerlo en Python

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ── PIE CHART BÁSICO ──
ventas_segmento = (df.groupby('segmento')['ventas_total']
                     .sum()
                     .sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(9, 7))

wedges, texts, autotexts = ax.pie(
    ventas_segmento.values,
    labels=ventas_segmento.index,
    autopct='%1.1f%%',          # formato del porcentaje
    startangle=90,              # empieza desde arriba
    colors=sns.color_palette('Blues_d', n_colors=len(ventas_segmento)))

# Mejorar legibilidad de los textos
for text in autotexts:
    text.set_fontsize(9)
    text.set_fontweight('bold')

ax.set_title('Participación de Ventas por Segmento', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('images/piechart_basico.png', dpi=150)
plt.show()

# ── CON SECTOR DESTACADO (explode) ──
# Para enfatizar una categoría específica
explode = [0.05 if seg == 'Bleach'
           else 0
           for seg in ventas_segmento.index]

fig, ax = plt.subplots(figsize=(9, 7))

wedges, texts, autotexts = ax.pie(
    ventas_segmento.values,
    labels=ventas_segmento.index,
    autopct='%1.1f%%',
    startangle=90,
    explode=explode,            # separa el sector destacado
    colors=sns.color_palette('Blues_d', n_colors=len(ventas_segmento)),
           shadow=True)         # sombra para profundidad

for text in autotexts:
    text.set_fontsize(9)
    text.set_fontweight('bold')

ax.set_title('Bleach domina con el 68.7% de las ventas', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('images/piechart_explode.png', dpi=150)
plt.show()

# ── DONUT CHART (versión moderna del pie) ──
# Más limpio visualmente → deja espacio para un número central
fig, ax = plt.subplots(figsize=(9, 7))

wedges, texts, autotexts = ax.pie(
    ventas_segmento.values,
    labels=ventas_segmento.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette('Blues_d', n_colors=len(ventas_segmento)),
           wedgeprops=dict(width=0.5))  # ← esto lo convierte en donut

# Número central
total = ventas_segmento.sum()
ax.text(0, 0, f'${total/1e6:.1f}M\nTotal',
        ha='center', va='center',
        fontsize=14, fontweight='bold',
        color='#333333')

for text in autotexts:
    text.set_fontsize(8)

ax.set_title('Participación de Ventas por Segmento', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('images/donut_chart.png', dpi=150)
plt.show()

# ── PIE + BAR JUNTOS (lo más profesional) ──
# Muestra participación Y magnitud simultáneamente
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Pie: participación relativa
axes[0].pie(
    ventas_segmento.values,
    labels=[f'{s}\n{v/ventas_segmento.sum()*100:.1f}%'
            for s, v in zip(ventas_segmento.index, ventas_segmento.values)],
                            startangle=90, colors=sns.color_palette('Blues_d',
                            n_colors=len(ventas_segmento)),
    wedgeprops=dict(width=0.6))
axes[0].set_title('Participación Relativa (%)', fontsize=12)

# Bar → magnitud absoluta
colores = sns.color_palette('Blues_d', n_colors=len(ventas_segmento))
barras = axes[1].bar(
    ventas_segmento.index,
    ventas_segmento.values / 1e6,
    color=colores,
    edgecolor='white')

for barra in barras:
    altura = barra.get_height()
    axes[1].text(
        barra.get_x() + barra.get_width() / 2,
        altura + 0.05,
        f'${altura:.1f}M',
        ha='center', va='bottom',
        fontsize=8, fontweight='bold')

axes[1].set_title('Ventas Absolutas ($M)', fontsize=12)
axes[1].set_ylabel('Ventas ($M)')
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_ylim(0, ventas_segmento.max() / 1e6 * 1.15)

fig.suptitle('Análisis de Participación de Mercado por Segmento',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/pie_y_bar_combinado.png', dpi=150)
plt.show()
```

---

## <img src="https://img.icons8.com/?size=40&id=80406&format=png&color=000000" align="center"/> Cuándo SÍ usar pie chart

El pie chart es válido en casos muy específicos.
La pregunta clave es: **¿qué quiero que el lector vea?**

### <img src="https://img.icons8.com/?size=30&id=81146&format=png&color=000000" align="center"/> Caso 1: Mostrar que algo domina claramente
```
Bleach: 68.7%  ← más de 2/3 del total
Resto:  31.3%
```
Cuando un sector es tan grande que la dominancia es obvia visualmente, el pie funciona. 
El ojo capta fácilmente "más de la mitad" o "casi todo".

```python
# Simplificar para que el mensaje sea claro
ventas_simplificado = pd.Series({
    'Bleach': 7_581_283,
    'Resto':  3_461_576
})

fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(ventas_simplificado,
       labels=['Bleach\n68.7%', 'Resto\n31.3%'],
       colors=['#1D6FA4', '#ADB5BD'],
       startangle=90,
       wedgeprops=dict(width=0.6))
ax.set_title('Bleach representa más de 2/3\nde las ventas totales')
plt.show()
```

### <img src="https://img.icons8.com/?size=30&id=81146&format=png&color=000000" align="center"/> Caso 2: Pocas categorías (máximo 5)
Con 2-4 categorías los sectores son legibles y comparables. 
Con más de 5 los sectores pequeños son imposibles de distinguir.

### <img src="https://img.icons8.com/?size=30&id=81146&format=png&color=000000" align="center"/> Caso 3: Audiencia ejecutiva no técnica
En presentaciones a directivos o stakeholders que no tienen background técnico, el pie chart es familiar e intuitivo. 
Comunica "qué parte del pastel tiene cada uno" sin necesitar explicación.

### <img src="https://img.icons8.com/?size=30&id=81146&format=png&color=000000" align="center"/> Caso 4: Los porcentajes son el mensaje
Cuando lo que importa es la proporción (no la magnitud absoluta), el pie lo comunica directamente.

```
"México representa el 50% de las ventas"
→ Pie chart lo muestra inmediatamente
→ Bar chart requiere leer el eje Y y calcular mentalmente el porcentaje
```

---

## <img src="https://img.icons8.com/?size=30&id=80339&format=png&color=000000" align="center"/> Cuándo NO usar pie chart

### <img src="https://img.icons8.com/?size=30&id=80339&format=png&color=000000" align="center"/> Caso 1: Muchas categorías (más de 5)
```python
# Con 7 segmentos como en mi EDA:
# Bleach: 68.7%, Liquid&Gel: 13.2%, Powder: 9.4%,
# Pretreat: 5.7%, Bar: 2.2%, Sanitizer: 0.7%, Others: 0.1%

# <img src="https://img.icons8.com/?size=30&id=80339&format=png&color=000000" align="center"/> Pie chart → los últimos 4 son indiferenciables
# <img src="https://img.icons8.com/?size=30&id=81146&format=png&color=000000" align="center"/> Bar chart horizontal ordenado → mucho más claro
```

### <img src="https://img.icons8.com/?size=30&id=80339&format=png&color=000000" align="center"/> Caso 2: Diferencias pequeñas entre categorías
```
Categoría A: 26%
Categoría B: 24%
Categoría C: 25%
Categoría D: 25%

# El ojo NO puede distinguir estos ángulos
# Un bar chart muestra la diferencia claramente
```

### <img src="https://img.icons8.com/?size=30&id=80339&format=png&color=000000" align="center"/> Caso 3: Quieres comparar cambios en el tiempo
```
# <img src="https://img.icons8.com/?size=30&id=80339&format=png&color=000000" align="center"/> Dos pie charts → difícil comparar
2022: [pie chart]    2023: [pie chart]

# <img src="https://img.icons8.com/?size=30&id=81146&format=png&color=000000" align="center"/> Bar chart agrupado o apilado 100%
# muestra el cambio de proporciones claramente
```

### <img src="https://img.icons8.com/?size=30&id=80339&format=png&color=000000" align="center"/> Caso 4: Valores muy similares o muy pequeños
Los sectores menores al 5% son prácticamente invisibles en un pie.
Sus etiquetas se superponen y el gráfico se vuelve ilegible.

### <img src="https://img.icons8.com/?size=30&id=80339&format=png&color=000000" align="center"/> Caso 5: Necesitas valores exactos
El ojo humano no puede leer ángulos con precisión.
Si necesitas que alguien compare 26% vs 24%, usa bar chart.

---

## <img src="https://img.icons8.com/?size=40&id=80435&format=png&color=000000" align="center"/> Pie Chart vs Bar Chart

```python
# El mismo dato → dos gráficas muy distintas
datos = pd.Series({
    'Bleach':      68.7,
    'Liquid & Gel': 13.2,
    'Powder':       9.4,
    'Pretreat':     5.7,
    'Bar':          2.2,
    'Sanitizer':    0.7,
    'Others':       0.1
})

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# PIE → difícil comparar los segmentos pequeños
axes[0].pie(datos.values,
            labels=datos.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette('Blues_d', n_colors=7))
axes[0].set_title('Pie Chart\n✗ Difícil comparar segmentos pequeños', fontsize=11)

# BAR → clara y precisa
colores = sns.color_palette('Blues_d', n_colors=7)
barras = axes[1].barh(datos.index[::-1],
                      datos.values[::-1],
                      color=colores[::-1],
                      edgecolor='white')

for barra in barras:
    largo = barra.get_width()
    axes[1].text(largo + 0.3,
                barra.get_y() + barra.get_height() / 2,
                f'{largo:.1f}%',
                ha='left', va='center', fontsize=9)

axes[1].set_title('Bar Chart\n✓ Mucho más fácil de comparar', fontsize=11)
axes[1].set_xlabel('Participación (%)')
axes[1].set_xlim(0, 80)

plt.suptitle('El mismo dato: ¿cuál comunica mejor?', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/pie_vs_bar.png', dpi=150)
plt.show()
```

**La conclusión visual siempre es la misma:**
El bar chart muestra diferencias con más precisión.
El pie chart solo gana cuando la proporción
es el mensaje principal y las categorías son pocas.

---

## <img src="https://img.icons8.com/?size=40&id=81279&format=png&color=000000" align="center"/> Donut Chart → la versión mejorada

El donut chart es un pie chart con el centro vacío. Es preferible al pie porque:

**Ventajas del donut sobre el pie:**
- El espacio central puede mostrar un número resumen
- Visualmente más moderno y limpio
- El ojo lo lee ligeramente mejor que el pie completo
- Se usa mucho en dashboards ejecutivos

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

datos_simple = pd.Series({'México': 50.0, 'Otras regiones': 50.0})

# Pie tradicional
axes[0].pie(datos_simple,
            labels=datos_simple.index,
            autopct='%1.0f%%',
            colors=['#1D6FA4', '#ADB5BD'],
            startangle=90)
axes[0].set_title('Pie Chart')

# Donut
axes[1].pie(datos_simple,
            labels=datos_simple.index,
            autopct='%1.0f%%',
            colors=['#1D6FA4', '#ADB5BD'],
            startangle=90,
            wedgeprops=dict(width=0.5))

# Número central → el valor más importante
axes[1].text(0, 0, '50%\nMéxico',
             ha='center', va='center',
             fontsize=14, fontweight='bold')
axes[1].set_title('Donut Chart ✓ (versión mejorada)')

plt.tight_layout()
plt.savefig('images/pie_vs_donut.png', dpi=150)
plt.show()
```

---

## <img src="https://img.icons8.com/?size=40&id=7MuYE737H6Vg&format=png&color=000000" align="center"/> Regla de los ángulos: por qué el pie engaña

El problema fundamental del pie chart es que el cerebro humano es malo comparando ángulos:

```
¿Cuál sector es más grande?

Pie chart:           Bar chart:
    /\               A │████████████  45%
   /  \              B │█████████████ 40%
  / A? \             C │████          15%
 /______\
(        )
(   B?   )
(________)

Con el pie: difícil saberlo sin etiquetas
Con el bar: obvio de inmediato
```

Esta limitación cognitiva es la razón principal por la que los expertos en visualización prefieren el bar chart.
El pie chart requiere que leas las etiquetas para entenderlo. Si necesitas leer números, mejor usar un bar chart directamente.

---

## <img src="https://img.icons8.com/?size=40&id=80923&format=png&color=000000" align="center"/> Errores comunes

**Error 1: Más de 5-6 categorías**
Con muchas categorías los sectores pequeños son invisibles y sus etiquetas se superponen creando caos visual.
Agrupa las menores en "Otros".

```python
# Simplificar antes de graficar
umbral_pct = 5  # categorías < 5% → "Otros"
datos_filtrado = datos[datos >= umbral_pct]
otros = datos[datos < umbral_pct].sum()
if otros > 0:
    datos_filtrado['Otros'] = otros
```

**Error 2: Pie chart 3D**
El 3D distorsiona los ángulos, los sectores del frente parecen más grandes de lo que son.
Nunca uses pie 3D.

**Error 3: Comparar dos pie charts**
Es casi imposible comparar la misma categoría en dos pies distintos.
Usa bar chart agrupado o apilado 100%.

**Error 4: Sin etiquetas de porcentaje**
Un pie sin porcentajes es inútil.
El ojo no puede leer ángulos.
Siempre pon `autopct='%1.1f%%'`.

**Error 5: Colores sin contraste**
Sectores adyacentes con colores similares se fusionan visualmente.
Asegúrate de que cada sectorsea claramente distinguible del siguiente.

**Error 6: Usar pie para mostrar cambio**
Si quieres mostrar que la proporción de Bleach bajó del 70% al 60%, dos pie charts juntos no lo comunican bien.
Usa una línea temporal o bar apilado 100%.

---

## <img src="https://img.icons8.com/?size=40&id=hKNUXmKDeI2V&format=png&color=000000" align="center"/> El árbol de decisión: ¿pie o bar?

```
¿Quiero mostrar proporciones/participación?
            ↓
          ¿Sí?
            ↓
¿Tengo menos de 5 categorías?
    ├── Sí → ¿La dominancia es obvia (>50%)?
    │              ├── Sí → Pie o Donut ✓
    │              └── No → Bar chart es más claro
    └── No → Bar chart horizontal ✓
             (agrupa menores en "Otros" si son muchas)

¿Quiero comparar proporciones entre períodos?
    → Bar chart apilado 100% ✓

¿La audiencia es ejecutiva y el mensaje es simple?
    → Pie o Donut puede funcionar ✓

¿Necesito precisión en los valores?
    → Bar chart siempre ✓
```

---

## <img src="https://img.icons8.com/?size=45&id=NGU2KSe3KO8r&format=png&color=000000" align="center"/> Analogía con Medicina Tradicional China

El pie chart en MTC sería como el diagnóstico de los cinco elementos:

| Pie chart | Cinco Elementos MTC |
|---|---|
| Círculo completo | El organismo como sistema total |
| Cada sector | Participación de cada elemento (Madera, Fuego, Tierra, Metal, Agua) |
| Sector dominante | El elemento en exceso → el que desequilibra |
| Sectores iguales | Balance perfecto entre elementos |
| Sector muy pequeño | El elemento deficiente → el que necesita refuerzo |
| Donut con número central | El patrón principal con su diagnóstico |

> En MTC, el diagnóstico de los cinco elementos es como un pie chart del organismo:
> ¿qué proporción de la energía total está en cada sistema?
> Un elemento que ocupa el 70% del espacio es Bleach en mis datos de ventas
> → está desequilibrado y genera dependencia.

---

## <img src="https://img.icons8.com/?size=40&id=9nyiB6VEUhJy&format=png&color=000000" align="center"/> Cheat Sheet rápido

```
CUÁNDO SÍ USAR PIE:
→ Pocas categorías (2-5 máximo)
→ La dominancia es obvia (>50% en un sector)
→ El porcentaje es el mensaje principal
→ Audiencia ejecutiva no técnica
→ Dashboard donde el donut con número central comunica mejor que un bar

CUÁNDO NO USAR PIE:
→ Más de 5 categorías
→ Diferencias pequeñas entre sectores
→ Comparar cambios entre períodos
→ Necesitas valores exactos
→ Cualquier análisis técnico serio

DONUT > PIE:
→ Siempre que uses pie considera el donut
→ Permite número resumen en el centro
→ Más moderno y limpio visualmente

TIPOS DE PIE:
→ Pie simple      = proporciones básicas
→ Pie con explode = énfasis en un sector
→ Donut           = versión moderna con espacio central
→ Pie + Bar       = combo ideal para presentaciones

NUNCA:
→ Pie en 3D
→ Más de 5-6 sectores
→ Dos pies para comparar el mismo dato
→ Pie sin etiquetas de porcentaje
```

---

## <img src="https://img.icons8.com/?size=40&id=80358&format=png&color=000000" align="center"/> Aplicado a mis proyectos

### Proyecto EDA: Ventas Retail

```python
# Cómo usé y debí usar el pie en mi EDA:

# DONDE SÍ FUNCIONÓ:
# Pie de participación geográfica simplificado
datos_geo = pd.Series({
    'México': 50.0,
    'Otras regiones': 50.0
})
# → 2 categorías, mensaje claro: México = mitad exacta
# → El pie lo comunica perfectamente

# DONDE EL BAR FUE MEJOR:
# Los 7 segmentos de producto
# → Bleach (68.7%) vs 6 segmentos más
# → Usé pie en el EDA pero los sectores pequeños (Sanitizer 0.7%, Others 0.1%) eran prácticamente invisibles
# → Un bar horizontal habría sido más claro para mostrar las diferencias exactas

# LA COMBINACIÓN IDEAL para presentación:
# Donut con "68.7%" en el centro para Bleach + Bar horizontal para todos los segmentos
# → Donut: impacto visual inmediato
# → Bar: precisión para el análisis
```

### <img src="https://img.icons8.com/?size=40&id=axCN1QLCYi4j&format=png&color=000000" align="center"/> Posibles proyectos futuros
| Dataset | Pie útil | Por qué |
|---|---|---|
| SEMARNAT | Emisiones por tipo de industria (3 grupos) | Pocas categorías, dominancia clara |
| OMS | Causas de mortalidad (simplificado) | Comunicar a audiencia general |
| INEGI | PIB por sector (3-4 sectores) | Proporción relativa clara |
| Kaggle fraude | Fraude vs no fraude | Solo 2 categorías → donut perfecto |
| IMSS | Tipo de consulta (simplificado) | Mensaje de participación |

---

*Nota creada: 2026 · Serie: Visualizaciones para Ciencia de Datos* 

*Parte del certificado Científico de Datos — EBAC* <img src="https://img.icons8.com/?size=25&id=42810&format=png&color=000000" align="center"/>

*<img src="https://img.icons8.com/?size=25&id=Gh99xffskT40&format=png&color=000000" align="center"/> github.com/ReginaPema* 
