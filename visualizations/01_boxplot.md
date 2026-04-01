# 📊 01 · Boxplot (Box & Whisker Plot)
## Diagrama de Caja y Bigotes

> **En una sola gráfica te dice:**
> dónde está el centro, qué tan dispersos están los datos, si hay sesgo, y dónde están los outliers.
> Es la gráfica más informativa por unidad de espacio.

---

## 🧱 Anatomía de un boxplot
```
                    Outliers extremos
                         *  *
                         |
    ┌────────────────────┤  ← Bigote superior (máximo sin outliers)
    │                    │    = Q3 + 1.5 × IQR
    │         ┌──────────┤
    │         │          │  ← Q3 (percentil 75)
    │         │  IQR     │    75% de los datos están DEBAJO de aquí
    │         │          │
    │         │──────────│  ← Mediana (Q2, percentil 50)
    │         │          │    línea del centro
    │         │          │
    │         │          │  ← Q1 (percentil 25)
    │         └──────────┤    25% de los datos están DEBAJO de aquí
    │                    │
    └────────────────────┤  ← Bigote inferior (mínimo sin outliers)
                         │    = Q1 - 1.5 × IQR
                    *  *
                Outliers extremos
```

| Elemento | Nombre técnico | Qué significa |
|---|---|---|
| Línea central de la caja | Mediana (Q2) | El valor del medio: más robusto que la media |
| Borde inferior de la caja | Q1 (percentil 25) | El 25% de datos está por debajo |
| Borde superior de la caja | Q3 (percentil 75) | El 75% de datos está por debajo |
| Altura de la caja | IQR (Q3 - Q1) | Donde vive el 50% central de los datos |
| Bigote inferior | Q1 - 1.5×IQR | El mínimo "esperado" sin outliers |
| Bigote superior | Q3 + 1.5×IQR | El máximo "esperado" sin outliers |
| Puntos fuera de bigotes | Outliers | Valores atípicos: merecen investigación |

---

## 🐍 Cómo hacerlo en Python
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ── BOXPLOT BÁSICO (una variable) ──
fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(df['ventas'])
ax.set_title('Distribución de Ventas')
ax.set_ylabel('Ventas ($)')
plt.show()

# ── CON SEABORN (más bonito, más opciones) ──

# Una variable
sns.boxplot(data=df, y='ventas')

# Por grupos: el más útil en análisis real
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df,
            x='segmento',      # grupos en el eje X
            y='ventas',        # valores en el eje Y
            palette='pastel',  # colores suaves
            showfliers=True)   # mostrar outliers (True por defecto)
ax.set_title('Distribución de Ventas por Segmento', fontsize=14)
ax.set_xlabel('Segmento')
ax.set_ylabel('Ventas ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Escala logarítmica: cuando hay outliers muy extremos
# (exactamente como en tu proyecto EDA)
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df, x='segmento', y='ventas', palette='pastel')
ax.set_yscale('log')   # ← esto cambia todo
ax.set_title('Distribución de Ventas por Segmento (Escala Log)')
plt.tight_layout()
plt.show()

# ── CALCULAR LOS VALORES MANUALMENTE ──
Q1  = df['ventas'].quantile(0.25)
Q3  = df['ventas'].quantile(0.75)
IQR = Q3 - Q1
bigote_inf = Q1 - 1.5 * IQR
bigote_sup = Q3 + 1.5 * IQR

print(f"Q1 (percentil 25): ${Q1:,.2f}")
print(f"Mediana:           ${df['ventas'].median():,.2f}")
print(f"Q3 (percentil 75): ${Q3:,.2f}")
print(f"IQR:               ${IQR:,.2f}")
print(f"Bigote inferior:   ${bigote_inf:,.2f}")
print(f"Bigote superior:   ${bigote_sup:,.2f}")

outliers = df[(df['ventas'] < bigote_inf) |
              (df['ventas'] > bigote_sup)]
print(f"Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
```

---

## 🧭 Cómo leer un boxplot: los 5 patrones clave

### Patrón 1: Caja simétrica, bigotes iguales
```
     |
  ┌──┼──┐
  │  │  │
  └──┼──┘
     |
```
**Qué significa:**
Distribución simétrica y equilibrada.
La mediana está en el centro de la caja.
Los bigotes tienen longitud similar.

**Insight:** 
Los datos son predecibles y estables.
La media y la mediana son similares.
Puedes usar el promedio con confianza.

**Ejemplos reales:**
- Temperatura diaria en una ciudad estable
- Tiempo de entrega de un proceso muy estandarizado
- Calificaciones en un grupo homogéneo

---

### Patrón 2: Caja con sesgo positivo (skewed right)
```
  ┌─┼────┐
  │ │    │        *  *  *
  └─┼────┘
    |         →
```
**Qué significa:**
La mediana está cerca del borde inferior de la caja.
El bigote superior es mucho más largo que el inferior.
Hay outliers en el lado alto.

**Insight:** La mayoría de los valores son bajos, pero hay algunos casos de valor muy alto jalando la distribución hacia la derecha.

**Ejemplos reales:**
- Ventas retail: la mayoría son pequeñas, pocas son enormes
- Ingresos por hogar en México
- Precios de vivienda
- Tiempo de resolución de tickets de soporte

**Acción:** Investiga los outliers altos.
Pueden ser tu segmento más valioso o errores de datos.

---

### Patrón 3: Caja con sesgo negativo (skewed left)
```
         ┌────┼─┐
*  *  *  │    │ │
         └────┼─┘
                |
```
**Qué significa:**
La mediana está cerca del borde superior de la caja.
El bigote inferior es más largo.
Hay outliers en el lado bajo.

**Insight:** La mayoría de los valores son altos, pero hay algunos casos de valor muy bajo.

**Ejemplos reales:**
- Calificaciones en examen muy fácil (mayoría saca 90-100, pocos sacan 40)
- Tiempo de vida de productos de calidad (mayoría dura años, pocos fallan pronto)
- Puntuaciones de satisfacción del cliente en un servicio muy bueno

---

### Patrón 4: Caja muy pequeña, bigotes largos
```
      *  *
      |
   ┌──┼──┐
   └──┼──┘
      |
   *  *  *
```
**Qué significa:**
El 50% central está muy concentrado (IQR pequeño), pero hay valores extremos en ambos lados.
Alta concentración en el centro + muchos outliers.

**Insight:** Hay dos comportamientos distintos: el caso "normal" muy consistente, y casos excepcionales en ambos extremos.

**Ejemplos reales:**
- Transacciones financieras: la mayoría son normales, algunas son fraude (alto) o devoluciones (bajo)
- Temperatura con eventos climáticos extremos
- Consumo eléctrico: uso normal vs picos de calor o frío extremo

---

### Patrón 5: Comparando múltiples cajas
```
Segmento A:    Segmento B:    Segmento C:
               *  *
    ┌──┐       |              ┌──────┐
    │  │    ┌──┼──┐           │      │
    │──│    │  │  │           │──────│
    │  │    └──┼──┘           │      │
    └──┘       |              └──────┘
```
**Qué significa:**
- **A:** Distribución estrecha: muy consistente
- **B:** Distribución con outliers: hay casos extremos
- **C:** IQR grande: alta variabilidad natural

**Insight:**
A es el segmento más predecible.
B tiene casos excepcionales que investigar.
C tiene comportamiento naturalmente variable.

**Este es el uso más poderoso del boxplot:** comparar la distribución de múltiples grupos en una sola gráfica.

---

## 🎯 Cuándo usar boxplot vs otras gráficas

| Situación | Mejor gráfica |
|---|---|
| Ver distribución de UNA variable | Histograma |
| Ver distribución + outliers de UNA variable | Boxplot |
| Comparar distribución entre GRUPOS | Boxplot por grupos |
| Ver forma exacta de la distribución | Histograma o KDE |
| Datos pequeños (< 50 valores) | Dotplot o stripplot |
| Quieres ver todos los puntos + boxplot | Boxplot + stripplot |
```python
# Boxplot + puntos individuales (para datasets pequeños)
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='segmento', y='ventas',
            palette='pastel', ax=ax, width=0.5)
sns.stripplot(data=df, x='segmento', y='ventas',
              color='black', alpha=0.3, size=3, ax=ax)
plt.tight_layout()
plt.show()
```

---

## ❌ Errores comunes al interpretar boxplots

**Error 1: Confundir la caja con "todos los datos"**
La caja solo contiene el 50% central. El otro 50% está en los bigotes y outliers.

**Error 2: Ignorar los outliers**
Los puntos fuera de los bigotes no son "ruido a eliminar" son los datos más interesantes. Siempre investígalos.

**Error 3: Olvidar que la mediana ≠ media**
El boxplot muestra la MEDIANA, no la media. Si necesitas la media, calcúlala por separado.

**Error 4: Usar boxplot con pocos datos**
Con menos de 20-30 puntos, un boxplot puede ser engañoso. Usa dotplot mejor.

**Error 5: No usar escala log cuando hay outliers extremos**
Si tus outliers son 100x más grandes que la mediana, la caja va a verse aplastada y no podrás leer nada.
Usa `ax.set_yscale('log')`.

---

## <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzMxY200dzhyc2RhbG5zbnk3MWZoZHE2ODZnM3I3Y2lseHNtYXI4diZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/WmvhTzEl9ihy9TqPme/giphy.gif" width="40"> Analogía con Medicina Tradicional China

En diagnóstico de MTC, cuando evalúas a un grupo de pacientes con la misma condición:

| Elemento del boxplot | Diagnóstico MTC |
|---|---|
| Caja (IQR) | El rango de presentación típica del patrón |
| Mediana | El caso "clásico" del patrón |
| Bigotes | Variaciones esperadas del mismo patrón |
| Outliers | Casos atípicos que no siguen el patrón: los más interesantes clínicamente |
| Caja pequeña | Patrón muy definido y consistente |
| Caja grande | Mismo diagnóstico, presentaciones muy variables |

Un maestro de MTC diría:
*"El outlier es donde está el diagnóstico real."*
Un científico de datos diría lo mismo.

---

## ⚡ Cheat Sheet rápido
```
Mediana al centro de la caja  →  Distribución simétrica
Mediana cerca del borde inf   →  Sesgo positivo (cola larga arriba)
Mediana cerca del borde sup   →  Sesgo negativo (cola larga abajo)

IQR pequeño  →  Datos muy consistentes y predecibles
IQR grande   →  Alta variabilidad natural

Bigote sup mucho más largo    →  Outliers altos (investigar)
Muchos puntos fuera           →  Alta proporción de casos atípicos

Cajas a distintas alturas     →  Los grupos tienen medianas distintas
Cajas con distinto IQR        →  Los grupos tienen variabilidades distintas
```

---

## 🔗 Aplicado a mis proyectos

### Proyecto EDA — Ventas Retail
```python
# Boxplot general:
# → Mediana muy baja ($16.81) vs media ($90.51)
# → IQR estrecho pero bigote superior MUY largo
# → 14,241 outliers (11.67%): todos hacia arriba
# → Patrón: sesgo positivo extremo
# → Insight: mercado dominado por pocas compras mayoristas enormes

# Boxplot por segmento:
# → Bleach: IQR más grande y outliers más extremos
#   Confirma que es el segmento con más variabilidad
# → Others: IQR muy pequeño: comportamiento consistente pero volumen insignificante

# Escala log:
# → La venta máxima ($12,236) vs mediana ($16.81) = diferencia de 728x
# → Sin escala log, la caja se aplasta y no se lee nada
# → Con escala log, todos los segmentos son legibles
```

### Posibles proyectos futuros
| Dataset | Qué esperar del boxplot | Por qué |
|---|---|---|
| SEMARNAT emisiones CO₂ | Sesgo positivo extremo + outliers altos | Pocas industrias gigantes |
| OMS mortalidad infantil | Sesgo positivo + outliers (países en crisis) | Desigualdad global |
| INEGI ingresos por hogar | Sesgo positivo muy marcado | Desigualdad económica México |
| Kaggle fraude bancario | IQR pequeño + outliers extremos en monto | Transacciones fraudulentas son el outlier |
| IMSS consultas por paciente | Sesgo positivo + outliers altos | Pacientes crónicos como outliers |

---


*Nota creada: 2026 · Serie: Visualizaciones para Ciencia de Datos*
*Parte del certificado Científico de Datos — EBAC*
*🪷 github.com/ReginaPema*
