# 📊 Standard Deviation & Variance · Desviación Estándar y Varianza
## Qué tan dispersos están tus datos

> ¿Qué tan confiable es tu promedio?
> ¿Tus datos están agrupados cerca del centro o esparcidos por todos lados?

---

## Definiciones rápidas

| Métrica | Qué mide | Unidades |
|---|---|---|
| **Varianza** (Variance) | Dispersión promedio al cuadrado | Unidades² (difícil de interpretar) |
| **Desviación Estándar** (Std Dev) | Raíz cuadrada de la varianza | Mismas unidades que tus datos |

La desviación estándar es simplemente la varianza en unidades interpretables. 
Siempre usa std en lugar de varianza para comunicar resultados.

```python
import pandas as pd
import numpy as np

# Calcular
varianza = df['columna'].var()
std      = df['columna'].std()
media    = df['columna'].mean()

print(f"Media:    {media:.2f}")
print(f"Varianza: {varianza:.2f}")
print(f"Std Dev:  {std:.2f}")

# Con describe() — incluye std automáticamente
df['columna'].describe()
```

---

## Cómo interpretar la desviación estándar

### La regla intuitiva

**Std pequeña** → Los datos están agrupados cerca de la media.
La media es muy representativa.

**Std grande** → Los datos están muy dispersos.
La media puede no representar bien a nadie.
```
Std pequeña:        Std grande:
   ↓media              ↓media
 ██████              ░░████░░
░████████░         ░░░░████░░░░
████████████     ░░░░░░████░░░░░░
```

---

## La Regla Empírica (68-95-99.7)

Para distribuciones aproximadamente normales:
```
                           68% de los datos
                         ←——————————————————→
                           |      |      |
                 ←—————————————————————————————————→
                                 95%
         ←————————————————————————————————————————————————→
                                99.7%

  media-3σ  media-2σ  media-1σ  media  media+1σ  media+2σ  media+3σ
```

| Rango | Contiene |
|---|---|
| media ± 1 std | ~68% de los datos |
| media ± 2 std | ~95% de los datos |
| media ± 3 std | ~99.7% de los datos |

```python
media = df['columna'].mean()
std   = df['columna'].std()

rango_68 = df[(df['columna'] >= media - std) & 
              (df['columna'] <= media + std)]

rango_95 = df[(df['columna'] >= media - 2*std) & 
              (df['columna'] <= media + 2*std)]

print(f"Dentro de 1 std: {len(rango_68)/len(df)*100:.1f}%")
print(f"Dentro de 2 std: {len(rango_95)/len(df)*100:.1f}%")
```

⚠️ **Importante:** Esta regla solo aplica bien cuando tus datos tienen distribución normal (campana).
Con datos sesgados como ventas o ingresos, úsala como referencia aproximada, no como verdad absoluta.

---

## 🎯 Coeficiente de Variación para comparar entre grupos

El CV te permite comparar la dispersión de grupos con escalas distintas. Un % de variación relativa.

```python
# Coeficiente de Variación (CV)
cv = (df['columna'].std() / df['columna'].mean()) * 100
print(f"CV: {cv:.1f}%")

# Comparar entre segmentos
cv_por_segmento = df.groupby('segmento')['venta'].agg(
    media='mean',
    std='std',
    cv=lambda x: (x.std() / x.mean()) * 100
).round(2)
print(cv_por_segmento)
```

| CV | Interpretación |
|---|---|
| < 15% | Baja variabilidad: datos muy consistentes |
| 15–30% | Variabilidad moderada |
| > 30% | Alta variabilidad: datos muy dispersos |

---

## 🌍 Casos reales por industria

### Ventas y Retail
```python
# Escenario: dos tiendas con la misma media de ventas diarias
tienda_a_media = 1000  # $1,000/día
tienda_b_media = 1000  # $1,000/día

tienda_a_std = 50    # muy consistente
tienda_b_std = 800   # muy errática

# ¿Cuál es más predecible para planear inventario?
# → Tienda A: sabe que casi siempre venderá ~$1,000
# → Tienda B: impredecible, puede vender $200 o $1,800 
```

**Insight de negocio:**
• Std alta en ventas = riesgo de inventario, flujo de caja impredecible, difícil de planear. 
• Std baja = negocio estable y predecible.

---

### Medio Ambiente
```python
# Emisiones de CO₂ por estado
emisiones_std = df.groupby('estado')['co2_tons'].std()

# Estado con std muy alta:
# → Emisiones muy variables año con año
# → Puede indicar industria intermitente o datos inconsistentes

# Estado con std muy baja:
# → Emisiones muy consistentes
# → Más predecible para metas de reducción
```

**Insight de política pública:**
Los estados con alta std son más difíciles de regular porque su comportamiento es impredecible.
Los de baja std son más fáciles de modelar y proyectar.

---

### Salud
```python
# Tiempo de espera en consultorios
espera_media = df['minutos_espera'].mean()  # 45 min
espera_std   = df['minutos_espera'].std()   # ejemplo A: 5 min / ejemplo B: 40 min

# Clínica A: media=45, std=5
# → Casi todos esperan entre 40-50 min. Predecible.

# Clínica B: media=45, std=40
# → Algunos esperan 5 min, otros 2 horas.
# → Misma media, experiencia completamente diferente.
```

**Insight de salud:**
Std alta en tiempos de espera indica inequidad en el servicio.
Algunos pacientes reciben atención muy rápida y otros esperan horas. La media no captura eso.

---

### Fraude y Seguridad
```python
# Comportamiento de una cuenta normal vs sospechosa
cuenta_normal_media = 3     # 3 transacciones/día
cuenta_normal_std   = 1     # entre 2-4 al día, consistente

cuenta_fraude_media = 3     # también 3 transacciones/día en promedio
cuenta_fraude_std   = 15    # de 0 a 50 en un día,  muy errática

# Misma media, std completamente diferente
# → La std alta ES la señal de fraude
```

**Insight de seguridad:**
Un cambio repentino en la std del comportamiento de una cuenta es una de las señales más fuertes de compromiso o fraude. 
La media puede ser normal, pero la variabilidad delata el patrón anómalo.

---

### Inversiones y Finanzas
```python
# El concepto de "riesgo" en finanzas ES la desviación estándar
# Dos inversiones con el mismo rendimiento promedio:

inversion_a_media = 8   # 8% rendimiento anual promedio
inversion_a_std   = 2   # varía entre 6-10%, predecible

inversion_b_media = 8   # 8% rendimiento anual promedio
inversion_b_std   = 25  # varía entre -17% y +33%, muy riesgosa
```

**En finanzas:** std = volatilidad = riesgo.
Misma media, riesgo completamente diferente.

---

## <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzMxY200dzhyc2RhbG5zbnk3MWZoZHE2ODZnM3I3Y2lseHNtYXI4diZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/WmvhTzEl9ihy9TqPme/giphy.gif" width="40"> Analogía con Medicina Tradicional China

La desviación estándar en salud es como la **variabilidad del pulso** en diagnóstico de MTC.

Un pulso con ritmo muy consistente (std baja) indica estabilidad del sistema nervioso autónomo.

Un pulso muy errático (std alta) puede indicar arritmia, estrés crónico, o desequilibrio del Corazón en términos de MTC.

| MTC | Estadística |
|---|---|
| Pulso consistente y regular | Std baja: sistema estable |
| Pulso errático e irregular | Std alta: alta variabilidad |
| Misma frecuencia pero diferente regularidad | Misma media, diferente std |
| Cambio súbito en el patrón del pulso | Cambio repentino en std: señal de alerta |

La media te dice dónde está el centro.
La std te dice qué tan confiable es ese centro.

---

## ⚡ Cheat Sheet rápido
```
Std pequeña  →  Datos consistentes  →  Media confiable  →  Fácil de predecir
Std grande   →  Datos dispersos     →  Media engañosa   →  Difícil de predecir

Std igual en dos grupos con distinta media:
→ El grupo con media mayor tiene mayor variabilidad absoluta
→ Usa CV (%) para comparar justamente

Cambio repentino en std:
→ Puede indicar fraude, falla, evento extraordinario
→ Siempre investigar

Regla 68-95-99.7:
→ Solo válida para distribuciones normales
→ Con datos sesgados úsala como referencia aproximada
```

---

## 🔗 Aplicado a mis proyectos

### Proyecto EDA - Ventas Retail
```python
# Lo que encontré en el análisis:
media_ventas = 90.51
std_ventas   = 312.45  # (aproximado del proyecto)
cv           = (312.45 / 90.51) * 100  # ~345% ← std altísima

# CV de 345% indica:
# → Datos extremadamente dispersos
# → La media de $90.51 no representa a nadie en particular
# → Confirma que hay segmentos muy distintos mezclados (retail minorista + mayoristas grandes)
# → Conclusión: segmentar antes de analizar
```

### Posibles proyectos futuros
| Dataset | Qué esperar de la std | Insight probable |
|---|---|---|
| SEMARNAT emisiones | Std muy alta | Mezcla de industrias pequeñas y gigantes |
| OMS mortalidad infantil | Std alta entre países | Desigualdad global en sistemas de salud |
| Kaggle fraude | Std alta en cuentas fraudulentas | La erraticidad es la señal |
| INEGI ingresos hogar | Std muy alta | Desigualdad económica en México |

---

*Nota creada: 2026 · Parte del certificado Científico de Datos — EBAC*  
*🪷 github.com/ReginaPema*
