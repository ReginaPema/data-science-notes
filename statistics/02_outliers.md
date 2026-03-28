# 📊 Outliers
## Valores Atípicos/Extremos

> ¿Este valor extremo es un error, un caso especial, o la información más importante de tu dataset?

---

## Definición rápida

Un **outlier** es un valor que se aleja significativamente del resto de los datos. 
No es automáticamente un error, a veces es la señal más importante de todo el análisis.

```python
import pandas as pd
import numpy as np

# Método 1: IQR (el más robusto y recomendado)
Q1 = df['columna'].quantile(0.25)
Q3 = df['columna'].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers = df[(df['columna'] < limite_inferior) | 
              (df['columna'] > limite_superior)]

print(f"Q1: {Q1:.2f}")
print(f"Q3: {Q3:.2f}")
print(f"IQR: {IQR:.2f}")
print(f"Límite inferior: {limite_inferior:.2f}")
print(f"Límite superior: {limite_superior:.2f}")
print(f"Outliers encontrados: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

# Método 2: Z-Score (mejor para distribuciones normales)
from scipy import stats
z_scores = np.abs(stats.zscore(df['columna']))
outliers_z = df[z_scores > 3]  # más de 3 desviaciones estándar
```

---

## Los 4 tipos de outliers y qué hacer con cada uno

### Tipo 1: Error de datos
**Qué es:** Un valor imposible o claramente equivocado.

**Ejemplos reales:**
- Edad: 450 años
- Temperatura corporal: -200°C
- Precio de producto: $0.00001
- Fecha de nacimiento: año 1150

**Qué hacer:** **Eliminar o corregir.**
Investiga primero si es un error de captura, unidades
incorrectas (ej. dólares vs centavos), o dato corrupto.

```python
# Identificar valores imposibles
print(df['edad'].describe())
print(df[df['edad'] > 120])  # imposible biológicamente

# Eliminar
df = df[df['edad'] <= 120]

# O reemplazar con NaN para tratarlo como dato faltante
df.loc[df['edad'] > 120, 'edad'] = np.nan
```

---

### Tipo 2: Caso extremo real, pero válido
**Qué es:** Un valor real, no es un error, pero es extraordinariamente alto o bajo comparado con el resto.

**Ejemplos reales:**
- Una transacción de $12,000 en un dataset donde la mayoría son de $15–$100 *(como en tu proyecto EDA)*
- Un penthouse de $50M en un análisis de precios de departamentos donde el promedio es $3M
- Una empresa con emisiones 1000x mayores que el resto en un análisis de SEMARNAT
- Un paciente con 47 consultas al año cuando el promedio es 3 *(datos OMS o IMSS)*

**Qué hacer:** **Conservar, pero analizar por separado.**
No lo elimines, puede ser tu insight más valioso.
Segmenta el análisis: con y sin outliers.
```python
# Analizar con y sin outliers
df_sin_outliers = df[df['venta'] <= limite_superior]
df_outliers = df[df['venta'] > limite_superior]

print("=== DATASET COMPLETO ===")
print(df['venta'].describe())

print("\n=== SIN OUTLIERS ===")
print(df_sin_outliers['venta'].describe())

print("\n=== SOLO OUTLIERS ===")
print(df_outliers['venta'].describe())
print(df_outliers['segmento'].value_counts())  # ¿de dónde vienen?
```

---

### Tipo 3: Outlier que es el negocio real
**Qué es:** Lo que parece outlier es en realidad el segmento más importante de tu negocio.

**Ejemplo clásico:**
En un banco, el 80% de las transacciones son pequeñas ($10–$500) pero el 1% de clientes VIP hace transacciones de $100,000+. 
Ese "outlier" genera el 60% de los ingresos.

**En mi proyecto EDA:**
Las transacciones mayoristas de Bleach en la región México parecen outliers, pero son el corazón del negocio.
Eliminarlas habría destruido el análisis.

**Qué hacer:** **Nunca eliminar sin investigar primero.**
Pregúntate: ¿este outlier representa un segmento de clientes, un canal de venta, o un producto especial?

---

### Tipo 4: Outlier como señal de fraude o anomalía
**Qué es:** Un valor atípico que indica que algo está mal en el proceso, no en los datos.

**Ejemplos reales:**
- Una cuenta bancaria con 200 transacciones en 1 hora (detección de fraude)
- Una estación meteorológica que reporta 80°C (sensor dañado)
- Un médico que prescribe 10x más opioides que sus colegas (alerta de fraude en salud)
- Una fábrica con emisiones 500% sobre su promedio histórico (posible accidente o reporte incorrecto)

**Qué hacer:** **Investigar inmediatamente.**
Este tipo de outlier es una alerta, no un dato a eliminar.

```python
# Detectar outliers temporales (picos inusuales)
df['rolling_mean'] = df['valor'].rolling(window=7).mean()
df['desviacion'] = abs(df['valor'] - df['rolling_mean'])
alertas = df[df['desviacion'] > df['desviacion'].std() * 3]
print(f"Alertas detectadas: {len(alertas)}")
```

---

## Métodos de detección — cuándo usar cada uno

| Método | Cuándo usarlo | Código |
|---|---|---|
| **IQR** | Siempre como primera opción — robusto ante cualquier distribución | `Q1 - 1.5*IQR` / `Q3 + 1.5*IQR` |
| **Z-Score** | Cuando tus datos son aproximadamente normales | `z > 3` |
| **Percentiles** | Cuando quieres un umbral específico de negocio | `quantile(0.99)` |
| **Visualización** | Siempre, complementa cualquier método numérico | boxplot, histograma |

```python
# Visualización siempre primero
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Boxplot — muestra outliers como puntos individuales
df.boxplot(column='columna', ax=axes[0])
axes[0].set_title('Boxplot — outliers visibles')

# Histograma — muestra la distribución general
df['columna'].hist(bins=50, ax=axes[1])
axes[1].set_title('Distribución general')

plt.tight_layout()
plt.show()
```

---

## El proceso de decisión
```
¿Es un outlier?
      ↓
¿Es un error de datos? (valor imposible, captura incorrecta)
  → SÍ: Corregir o eliminar
  → NO: ↓
¿Es un caso real pero extremo?
  → ¿Representa un segmento importante del negocio?
      → SÍ: Conservar y analizar por separado
      → NO: Analizar con y sin él, reportar ambos
  → ¿Es una señal de fraude, falla o anomalía?
      → SÍ: Investigar inmediatamente, es el hallazgo más importante
```

---

## <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzMxY200dzhyc2RhbG5zbnk3MWZoZHE2ODZnM3I3Y2lseHNtYXI4diZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/WmvhTzEl9ihy9TqPme/giphy.gif" width="40"> Analogía con Medicina Tradicional China

En diagnóstico clínico, un síntoma "outlier", algo que no encaja con el patrón general del paciente, a veces es el síntoma más importante.

Un paciente con fatiga crónica "normal" que de repente tiene un dolor específico en el hipocondrio derecho no es un dato a ignorar, puede ser el diagnóstico real.

| MTC | Estadística |
|---|---|
| Síntoma que no encaja con el patrón | Outlier |
| Síntoma de captura incorrecta (confusión del paciente) | Error de datos |
| Síntoma raro que es el diagnóstico principal | Outlier tipo 3 |
| Señal de urgencia médica | Outlier tipo 4, alerta |

La pregunta siempre es la misma:
**¿Este valor atípico es ruido o señal?**

---

## ⚡ Cheat Sheet rápido
```
Outlier imposible      →  Error de datos       →  Corregir/eliminar
Outlier real extremo   →  Caso especial        →  Analizar por separado
Outlier = el negocio   →  Segmento importante  →  NUNCA eliminar sin investigar
Outlier = anomalía     →  Alerta               →  Investigar de inmediato

Regla de oro: NUNCA elimines un outlier solo porque "se ve raro".
Siempre pregunta primero: ¿por qué existe este valor?
```

---

## 🔗 Aplicado a mis proyectos

### Proyecto EDA - Ventas Retail
```python
# Lo que encontré:
outliers_encontrados = 14241   # 11.67% del dataset
venta_maxima = 12236.76
# → Origen: segmento Bleach + región México
# → Decisión: conservar, son transacciones mayoristas reales
# → Insight: ese 11.67% representa una parte desproporcionada
#            del ingreso total
```

### Posibles proyectos futuros
| Dataset | Tipo de outlier esperado | Qué investigar |
|---|---|---|
| SEMARNAT emisiones | Tipo 3 o 4 | Industrias gigantes vs accidentes/errores |
| OMS mortalidad | Tipo 2 | Países en crisis humanitaria |
| Kaggle fraude | Tipo 4 | Transacciones fraudulentas son los outliers |
| IMSS consultas | Tipo 2 y 4 | Pacientes crónicos vs fraude médico |

---

*Nota creada: 2026 · Parte del certificado Científico de Datos — EBAC*  
*🪷 github.com/ReginaPema*
