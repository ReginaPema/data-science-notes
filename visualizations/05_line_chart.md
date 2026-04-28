# Line Chart
## Gráfica de Línea — El tiempo hecho visible

> El line chart es la única gráfica diseñada específicamente para mostrar cómo cambia algo a lo largo del tiempo.
> Convierte una secuencia de números en una historia visual con principio, desarrollo y tendencia.

> **Regla de oro:**
> Si tu eje X es tiempo, usa line chart.
> Si tu eje X es una categoría, usa bar chart.
> La línea implica continuidad, solo úsala cuando los puntos realmente están conectados en el tiempo.

---

## 🧱 Anatomía de un line chart

```
Valor │
      │         *peak*
      │        /      \
      │       /        \         *peak*
      │      /          \       /      \
      │-----/------------\-----/        \----
      │    /              \   /          \
      │   /  tendencia →   \ /            \
      │  /               *valley*
      └──────────────────────────────────────→ Tiempo

Elementos clave:
* Cada punto     = valor en un momento específico
* La línea       = conexión entre momentos consecutivos
* Pendiente ↑    = crecimiento
* Pendiente ↓    = caída
* Pico (peak)    = máximo local
* Valle          = mínimo local
* Tendencia      = dirección general a largo plazo
* Estacionalidad = patrón que se repite periódicamente
```

| Elemento | Qué representa |
|---|---|
| Eje X | Tiempo (días, semanas, meses, años, trimestres) |
| Eje Y | La métrica que estás midiendo |
| Cada punto | El valor de la métrica en ese momento |
| La línea | La continuidad entre momentos |
| Pendiente | La velocidad y dirección del cambio |
| Pico | Máximo local: momento de mayor valor |
| Valle | Mínimo local: momento de menor valor |
| Línea de tendencia | Dirección general ignorando fluctuaciones |
| Promedio móvil | Suavizado de la señal para ver tendencia |

---

## 🐍 Cómo hacerlo en Python

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Preparar datos temporales
ventas_mensual = (df.groupby(['año', 'mes'])['ventas_total']
                    .sum()
                    .reset_index())
ventas_mensual['periodo'] = (
    ventas_mensual['año'].astype(str) + '-' +
    ventas_mensual['mes'].astype(str).str.zfill(2))
ventas_mensual = ventas_mensual.sort_values('periodo')

# ── LINE CHART BÁSICO ──
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(ventas_mensual['periodo'],
        ventas_mensual['ventas_total'] / 1000,
        color='steelblue',
        linewidth=2,
        marker='o',          # punto en cada valor
        markersize=6,
        label='Ventas mensuales ($K)')

ax.set_title('Tendencia Mensual de Ventas', fontsize=14)
ax.set_xlabel('Mes')
ax.set_ylabel('Ventas ($K)')
ax.tick_params(axis='x', rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/linechart_basico.png', dpi=150)
plt.show()

# ── CON PROMEDIO MÓVIL ──
# Suaviza la señal para ver la tendencia real
# sin el ruido de fluctuaciones puntuales
fig, ax = plt.subplots(figsize=(14, 6))

# Línea original
ax.plot(ventas_mensual['periodo'],
        ventas_mensual['ventas_total'] / 1000,
        color='steelblue', linewidth=1.5,
        marker='o', markersize=5, alpha=0.7,
        label='Ventas mensuales')

# Promedio móvil de 3 meses
ventas_mensual['rolling_3m'] = (
    ventas_mensual['ventas_total']
    .rolling(window=3, center=True)
    .mean())

ax.plot(ventas_mensual['periodo'],
        ventas_mensual['rolling_3m'] / 1000,
        color='red', linewidth=2.5,
        linestyle='--',
        label='Promedio móvil 3 meses')

ax.set_title('Ventas con Promedio Móvil (Tendencia Suavizada)',
             fontsize=14)
ax.set_xlabel('Mes')
ax.set_ylabel('Ventas ($K)')
ax.tick_params(axis='x', rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/linechart_rolling.png', dpi=150)
plt.show()

# ── MÚLTIPLES LÍNEAS: por grupo ──
ventas_segmento = (df.groupby(['año', 'mes', 'segmento'])
                     ['ventas_total']
                     .sum()
                     .reset_index())
ventas_segmento['periodo'] = (
    ventas_segmento['año'].astype(str) + '-' +
    ventas_segmento['mes'].astype(str).str.zfill(2))

fig, ax = plt.subplots(figsize=(14, 7))

sns.lineplot(data=ventas_segmento,
             x='periodo',
             y='ventas_total',
             hue='segmento',      # una línea por segmento
             marker='o',
             markersize=5,
             linewidth=1.5,
             ax=ax)

ax.set_title('Tendencia Mensual por Segmento', fontsize=14)
ax.set_xlabel('Mes')
ax.set_ylabel('Ventas ($)')
ax.tick_params(axis='x', rotation=45)
ax.legend(title='Segmento',
          bbox_to_anchor=(1.05, 1),
          loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/linechart_por_segmento.png', dpi=150)
plt.show()

# ── ESCALA LOGARÍTMICA ──
# Cuando hay líneas con magnitudes muy distintas
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Escala normal — Bleach aplasta a todos
sns.lineplot(data=ventas_segmento,
             x='periodo', y='ventas_total',
             hue='segmento', marker='o',
             markersize=4, linewidth=1.5,
             ax=axes[0])
axes[0].set_title('Escala normal — segmentos pequeños invisibles')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3)

# Escala log — todos los segmentos legibles
sns.lineplot(data=ventas_segmento,
             x='periodo', y='ventas_total',
             hue='segmento', marker='o',
             markersize=4, linewidth=1.5,
             ax=axes[1])
axes[1].set_yscale('log')
axes[1].set_title('Escala logarítmica — todos los segmentos visibles ✅')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/linechart_log_comparacion.png', dpi=150)
plt.show()

# ── ANOTACIONES EN PICOS Y VALLES ──
fig, ax = plt.subplots(figsize=(14, 6))

ventas = ventas_mensual['ventas_total'] / 1000
periodos = ventas_mensual['periodo']

ax.plot(periodos, ventas,
        color='steelblue', linewidth=2,
        marker='o', markersize=6)

# Encontrar y anotar máximo y mínimo
idx_max = ventas.idxmax()
idx_min = ventas.idxmin()

ax.annotate(f'Máximo\n${ventas[idx_max]:,.0f}K',
            xy=(periodos[idx_max], ventas[idx_max]),
            xytext=(0, 15), textcoords='offset points',
            ha='center', fontsize=9,
            color='green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green'))

ax.annotate(f'Mínimo\n${ventas[idx_min]:,.0f}K',
            xy=(periodos[idx_min], ventas[idx_min]),
            xytext=(0, -25), textcoords='offset points',
            ha='center', fontsize=9,
            color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red'))

ax.set_title('Tendencia Mensual: Picos y Valles Identificados',
             fontsize=14)
ax.set_xlabel('Mes')
ax.set_ylabel('Ventas ($K)')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/linechart_anotado.png', dpi=150)
plt.show()

# ── DOS MÉTRICAS EN EJE DOBLE ──
# Cuando quieres comparar dos variables
# con escalas completamente distintas
fig, ax1 = plt.subplots(figsize=(14, 6))

# Eje izquierdo: ventas en $
color1 = 'steelblue'
ax1.plot(ventas_mensual['periodo'],
         ventas_mensual['ventas_total'] / 1000,
         color=color1, linewidth=2,
         marker='o', markersize=5,
         label='Ventas ($K)')
ax1.set_xlabel('Mes')
ax1.set_ylabel('Ventas ($K)', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.tick_params(axis='x', rotation=45)

# Eje derecho: unidades vendidas
ax2 = ax1.twinx()
color2 = 'coral'

unidades_mensual = (df.groupby(['año', 'mes'])['unidades_vendidas']
                      .sum()
                      .reset_index())
unidades_mensual['periodo'] = (
    unidades_mensual['año'].astype(str) + '-' +
    unidades_mensual['mes'].astype(str).str.zfill(2))
unidades_mensual = unidades_mensual.sort_values('periodo')

ax2.plot(unidades_mensual['periodo'],
         unidades_mensual['unidades_vendidas'] / 1000,
         color=color2, linewidth=2,
         marker='s', markersize=5, linestyle='--',
         label='Unidades (K)')
ax2.set_ylabel('Unidades Vendidas (K)', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# Leyenda combinada
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='upper left')

ax1.set_title('Ventas y Unidades — Eje Doble', fontsize=14)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/linechart_doble_eje.png', dpi=150)
plt.show()
```

---

## 🧭 Cómo leer un line chart: los 8 patrones

### Patrón 1: Tendencia alcista (uptrend)
```
     │                    /
     │                   /
     │               ___/
     │           ___/
     │       ___/
     │   ___/
     └──────────────────────→ tiempo
```
**Qué significa:**
La métrica crece consistentemente lo largo del tiempo.
La pendiente general es positiva.

**Ejemplos reales:**
- Ventas de empresa en crecimiento
- Usuarios activos de app nueva
- Temperatura global a largo plazo
- Valor de inversión bien gestionada

**Insight de negocio:**
La tendencia actual es favorable.
Proyecta el crecimiento para planear capacidad, inventario o inversión futura.

---

### Patrón 2: Tendencia bajista (downtrend)
```
     │\
     │  \
     │    \___
     │        \___
     │             \___
     │                  \___
     └──────────────────────→ tiempo
```
**Qué significa:**
La métrica cae consistentemente.
Señal de alerta que requiere investigación inmediata.

**Ejemplos reales:**
- Ventas de producto obsoleto
- Cobertura forestal con deforestación
- Efectividad de tratamiento sin refuerzo
- Retención de clientes cayendo

**Insight de negocio:**
⚠️ Acción requerida.
¿Cuándo empezó la caída?
¿Coincide con algún cambio interno o externo en ese momento?

---

### Patrón 3: Estacionalidad
```
     │  /\      /\      /\
     │ /  \    /  \    /  \
     │/    \  /    \  /    \
     │      \/      \/      \
     └──────────────────────→ tiempo
     Q1 Q2 Q3 Q4 Q1 Q2 Q3 Q4
```
**Qué significa:**
El patrón se repite periódicamente.
Hay ciclos regulares de subida y bajada que ocurren en los mismos momentos del año, mes o semana.

**Ejemplos reales:**
- Ventas retail: pico en noviembre-diciembre
- Temperatura: ciclo anual verano-invierno
- Consultas médicas: pico en temporada de frío
- Emisiones CO₂: ciclo estacional por calefacción

**Cómo detectarla:**
```python
# Descomposición de serie de tiempo
from statsmodels.tsa.seasonal import seasonal_decompose

resultado = seasonal_decompose(
    ventas_mensual['ventas_total'],
    model='additive',
    period=12)           # período anual

fig, axes = plt.subplots(4, 1, figsize=(14, 10))
resultado.observed.plot(ax=axes[0], title='Original')
resultado.trend.plot(ax=axes[1], title='Tendencia')
resultado.seasonal.plot(ax=axes[2], title='Estacionalidad')
resultado.resid.plot(ax=axes[3], title='Residuo (ruido)')
plt.tight_layout()
plt.show()
```

---

### Patrón 4: Tendencia con estacionalidad
```
     │         /\              /\
     │        /  \            /  \
     │    /\ /    \       /\ /    \
     │   /  V      \     /  V
     │  /            \  /
     │ /              \/
     └──────────────────────────→ tiempo
```
**Qué significa:**
Hay crecimiento general (tendencia) pero también fluctuaciones cíclicas (estacionalidad). 
Los picos y valles se vuelven progresivamente más altos.

**Ejemplos reales:**
- Empresa en crecimiento con ventas estacionales
- Temperatura global con ciclo anual

**Insight:**
Para proyecciones futuras necesitas modelar ambos componentes por separado:
la tendencia y la estacionalidad.

---

### Patrón 5: Ruptura estructural (structural break)
```
     │           │
     │___________|
     │           |            ___
     │           |        ___/
     │           |    ___/
     │           |___/
     └───────────┼───────────────→ tiempo
               evento
```
**Qué significa:**
En un momento específico, el comportamiento cambia drásticamente y no vuelve a ser igual.
Generalmente coincide con un evento externo.

**Ejemplos reales:**
- Ventas de mascarillas en marzo 2020
- Tráfico web tras viralización en redes sociales
- Producción industrial tras cambio regulatorio
- Adopción de medicamento tras aprobación

**Cómo identificar el punto de cambio:**
```python
# Busca el punto donde la pendiente cambia más
ventas_mensual['cambio'] = (
    ventas_mensual['ventas_total'].diff().abs())
punto_cambio = ventas_mensual.loc[
    ventas_mensual['cambio'].idxmax(), 'periodo']
print(f"Mayor cambio en: {punto_cambio}")
```

---

### Patrón 6: Volatilidad alta (ruido)
```
     │  /\  /\    /\
     │ /  \/  \  /  \/\
     │/        \/      \  /\
     │                  \/  \/\
     └──────────────────────────→ tiempo
```
**Qué significa:**
Los valores fluctúan mucho de un período al siguiente sin patrón claro.
La señal tiene mucho "ruido".

**Cómo manejarlo:**
Usa promedio móvil para suavizar y ver la tendencia subyacente.
La volatilidad en sí misma es información, indica inestabilidad o alta variabilidad.

```python
# Suavizar con promedio móvil
for ventana in [3, 6, 12]:
    ventas_mensual[f'rolling_{ventana}m'] = (
        ventas_mensual['ventas_total']
        .rolling(window=ventana, center=True)
        .mean())
```

---

### Patrón 7: Plateau (estancamiento)
```
     │          _____________
     │      ___/
     │  ___/
     │_/
     └──────────────────────→ tiempo
```
**Qué significa:**
Hubo crecimiento pero se estabilizó en un nivel. 
La métrica ya no crece ni cae significativamente.

**Ejemplos reales:**
- Adopción de tecnología tras saturar el mercado
- Efectividad de tratamiento que llega a su límite
- Usuarios de plataforma madura

**Insight de negocio:**
¿El plateau es el nivel máximo natural o hay algo bloqueando el crecimiento?
Necesitas investigar para diferenciarlo.

---

### Patrón 8: Dato atípico temporal (spike)
```
     │          │
     │          *  ← spike
     │         / \
     │____    /   \____
     │    \  /
     │     \/
     └──────────────────→ tiempo
```
**Qué significa:**
En un período específico el valor fue extremadamente inusual y luego volvió al comportamiento normal.
Es un outlier temporal.

**Ejemplos reales:**
- Ventas altísimas el día de un promoción
- Temperatura inusualmente alta un día específico
- Pico de intentos de fraude en una fecha

**Insight:**
¿Qué pasó ese día/mes/período?
Los spikes casi siempre tienen una explicación externa concreta.

---

## 📊 El promedio móvil: tu mejor herramienta

El promedio móvil suaviza las fluctuaciones para revelar la tendencia real.

```python
# Diferentes ventanas = diferentes niveles de suavizado
fig, ax = plt.subplots(figsize=(14, 6))

# Datos originales
ax.plot(ventas_mensual['periodo'],
        ventas_mensual['ventas_total'] / 1000,
        color='lightsteelblue', linewidth=1,
        alpha=0.7, label='Original')

# Promedio móvil 3 meses — suavizado leve
ventas_mensual['rm_3'] = (ventas_mensual['ventas_total']
                           .rolling(3, center=True).mean())
ax.plot(ventas_mensual['periodo'],
        ventas_mensual['rm_3'] / 1000,
        color='steelblue', linewidth=2,
        label='Promedio móvil 3 meses')

# Promedio móvil 6 meses — suavizado fuerte
ventas_mensual['rm_6'] = (ventas_mensual['ventas_total']
                           .rolling(6, center=True).mean())
ax.plot(ventas_mensual['periodo'],
        ventas_mensual['rm_6'] / 1000,
        color='red', linewidth=2.5,
        linestyle='--',
        label='Promedio móvil 6 meses')

ax.set_title('Efecto del Promedio Móvil — Suavizado de Tendencia',
             fontsize=14)
ax.set_xlabel('Mes')
ax.set_ylabel('Ventas ($K)')
ax.tick_params(axis='x', rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/linechart_rolling_comparacion.png', dpi=150)
plt.show()
```

```
Ventana pequeña (3 meses):
→ Suaviza poco: ves las fluctuaciones mensuales
→ Útil para detectar cambios rápidos

Ventana grande (12 meses):
→ Suaviza mucho: ves solo la tendencia anual
→ Útil para ver dirección general a largo plazo

Regla práctica:
→ Datos diarios   → promedio móvil de 7 días
→ Datos semanales → promedio móvil de 4 semanas
→ Datos mensuales → promedio móvil de 3-6 meses
→ Datos anuales   → promedio móvil de 3-5 años
```

---

## 🎯 Line chart vs otras gráficas

| Situación | Usa |
|---|---|
| Variable cambia en el tiempo | Line chart |
| Comparar categorías sin tiempo | Bar chart |
| Distribución de una variable | Histograma |
| Relación entre dos numéricas | Scatter |
| Muchas series temporales a la vez | Line chart + leyenda o facets |
| Serie temporal + distribución | Line chart + histograma marginal |

**La regla del eje X:**
```
Eje X = tiempo → Line chart
Eje X = categoría → Bar chart
Eje X = número continuo → Scatter o Line chart
```

---

## ❌ Errores comunes

**Error 1: Usar line chart con datos categóricos**
Si el eje X es región, marca o segmento: no son datos continuos y la línea
implica una continuidad que no existe. Usa bar chart.

```python
# ❌ Incorrecto — región no es tiempo
ax.plot(['Mexico', 'Area 1', 'Area 2'], ventas_por_region)

# ✅ Correcto
ax.bar(['Mexico', 'Area 1', 'Area 2'], ventas_por_region)
```

**Error 2: No ordenar el eje temporal**
Si los períodos no están en orden cronológico, la línea va a zigzaguear sin sentido.

```python
# Siempre ordenar antes de graficar
df_tiempo = df_tiempo.sort_values('fecha')
```

**Error 3: Cortar el eje Y para exagerar cambios**
Empezar el eje Y en un valor distinto de 0
puede hacer que cambios pequeños parezcan enormes.

```python
# ❌ Engañoso — amplifica visualmente el cambio
ax.set_ylim(490000, 720000)

# ✅ Honesto — muestra el cambio en perspectiva
ax.set_ylim(0, df['ventas'].max() * 1.1)
```

**Error 4: Demasiadas líneas en una gráfica**
Con más de 5-6 líneas, el gráfico
se convierte en un espagueti ilegible.

```python
# ❌ Ilegible con 7 segmentos
sns.lineplot(data=df, x='periodo',
             y='ventas', hue='segmento')

# ✅ Mejor: destacar 1-2 líneas importantes y agrupar el resto como "otros"
top_segmentos = ['Bleach', 'Liquid & Gel']
df['segmento_plot'] = df['segmento'].apply(
    lambda x: x if x in top_segmentos else 'Otros')
sns.lineplot(data=df, x='periodo',
             y='ventas', hue='segmento_plot')
```

**Error 5: No manejar datos faltantes**
Si hay meses sin datos, la línea va a saltar o desaparecer.

```python
# Verificar continuidad temporal
periodos_esperados = pd.date_range(
    start=df['fecha'].min(),
    end=df['fecha'].max(),
    freq='MS')  # MS = inicio de mes

periodos_reales = df['fecha'].dt.to_period('M').unique()
faltantes = set(periodos_esperados.to_period('M')) - set(periodos_reales)

if faltantes:
    print(f"⚠️ Períodos sin datos: {sorted(faltantes)}")
```

**Error 6: Olvidar el contexto en datos incompletos**
Si el último período tiene datos incompletos (como julio 2023 en mi proyecto),
debes indicarlo claramente.

```python
# Marcar período incompleto
ax.axvline(x='2023-07', color='red',
           linestyle=':', alpha=0.7,
           label='Datos incompletos')
ax.annotate('Jul 2023\n(incompleto)',
            xy=('2023-07', ventas_max * 0.5),
            fontsize=8, color='red',
            ha='center')
```

---

## <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzMxY200dzhyc2RhbG5zbnk3MWZoZHE2ODZnM3I3Y2lseHNtYXI4diZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/WmvhTzEl9ihy9TqPme/giphy.gif" width="40" Analogía con Medicina Tradicional China

El line chart es como el seguimiento longitudinal de un paciente en MTC:

| Line chart | Seguimiento clínico MTC |
|---|---|
| Eje X: tiempo | Consultas a lo largo del tratamiento |
| Eje Y: métrica | Intensidad del síntoma principal |
| Tendencia bajista | Mejoría progresiva con el tratamiento |
| Tendencia alcista | Empeoramiento → revisar protocolo |
| Estacionalidad | Exacerbaciones estacionales del patrón |
| Spike | Recaída puntual → ¿qué pasó ese día? |
| Plateau | El tratamiento llegó a su límite → ajustar |
| Promedio móvil | Tendencia general ignorando variaciones diarias |
| Ruptura estructural | Cambio de protocolo que transformó el patrón |

> En MTC, el seguimiento longitudinal del paciente es un line chart mental.
> Cada consulta es un punto de datos.
> La línea entre consultas es la evolución.
> El objetivo es una tendencia bajista sostenida en los síntomas.

---

## ⚡ Cheat Sheet rápido

```
CUÁNDO USAR LINE CHART:
→ Eje X es tiempo (días, meses, años, trimestres)
→ Los puntos están conectados secuencialmente
→ Quieres mostrar cambio, tendencia o ciclos

PATRONES Y SU SIGNIFICADO:
→ Línea sube consistentemente  = tendencia alcista = crecimiento
→ Línea baja consistentemente  = tendencia bajista = ⚠️ alerta
→ Sube y baja regularmente     = estacionalidad = planear por ciclos
→ Cambio abrupto en un punto   = ruptura estructural = buscar el evento
→ Pico puntual y vuelve        = spike = ¿qué pasó ese día?
→ Se estabiliza en un nivel    = plateau = ¿límite natural o bloqueo?
→ Fluctúa sin patrón           = alta volatilidad = usar promedio móvil

PROMEDIO MÓVIL:
→ Datos diarios   → ventana 7
→ Datos semanales → ventana 4
→ Datos mensuales → ventana 3-6
→ Mayor ventana   = más suavizado = menos detalle

SIEMPRE:
→ Ordenar por fecha antes de graficar
→ Marcar períodos incompletos
→ No cortar el eje Y (empieza en 0)
→ Máximo 5-6 líneas por gráfica
→ Agregar promedio móvil si hay mucho ruido
→ Anotar picos y valles importantes

NUNCA:
→ Usar line chart con categorías en eje X
→ Conectar puntos que no son consecutivos en el tiempo
→ Mostrar 10+ líneas en una sola gráfica
→ Olvidar indicar datos incompletos
```

---

## 🔗 Aplicado a mis proyectos

### Proyecto EDA: Ventas Retail

```python
# Los line charts que hice y qué encontré:

# LINE CHART 1: Tendencia mensual general
# → Ventas entre $494K y $715K por mes
# → Sin tendencia clara alcista ni bajista
# → Pico en jul 2022 ($702K) y ene 2023 ($715K)
# → Valle en feb 2022 ($494K)
# → Comportamiento estable con variabilidad moderada
# → Jul 2023 excluido por datos incompletos

# LINE CHART 2: Tendencia por segmento
# → Bleach: línea muy por encima de todos los demás
# → Escala normal: los otros segmentos son invisibles
# → Solución: escala logarítmica en eje Y
# → Con log: todos los segmentos son legibles
# → Todos mantienen tendencia relativamente estable
# → Ningún segmento muestra crecimiento o caída fuerte

# PROMEDIO MÓVIL 3 MESES:
# → Suavizó las fluctuaciones mensuales
# → Reveló que no hay tendencia fuerte
#   en ninguna dirección
# → Confirmó: negocio estable, no en crecimiento
#   explosivo ni en caída

# INSIGHT TEMPORAL CLAVE:
# → Los picos de ventas coinciden con
#   enero (inicio de año — limpieza post-fiestas)
#   y julio (mitad de año — temporada de limpieza)
# → Posible estacionalidad semestral
#   que valdría confirmar con más años de datos
```

### Posibles proyectos futuros
| Dataset | Variable temporal | Patrón esperado |
|---|---|---|
| SEMARNAT emisiones | Emisiones CO₂ por año | Alcista a largo plazo + estacional |
| OMS mortalidad | Esperanza de vida por año | Alcista con ruptura en años de pandemia |
| INEGI | PIB trimestral México | Alcista con caída en 2020 (COVID) |
| Kaggle fraude | Fraudes por mes | Spikes en fechas específicas |
| IMSS | Consultas por mes | Estacionalidad fuerte (invierno = pico) |

---

*Nota creada: 2026 · Serie: Visualizaciones para Ciencia de Datos*
*Parte del certificado Científico de Datos — EBAC*
*🪷 github.com/ReginaPema*
