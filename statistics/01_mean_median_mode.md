# Mean, Median & Mode · Media, Mediana y Moda

> Cuando calculas estas tres métricas, 
> ¿qué te están diciendo sobre tu negocio o proyecto?

---

## Definiciones rápidas / Quick Definitions

| Métrica | Qué es | Cuándo usarla |
|---|---|---|
| **Media** (Mean) | Promedio aritmético — suma todos los valores y divide entre n | Cuando los datos son simétricos y sin outliers extremos |
| **Mediana** (Median) | El valor del centro cuando ordenas los datos de menor a mayor | Cuando hay outliers o distribución sesgada |
| **Moda** (Mode) | El valor que más se repite | Para datos categóricos o para ver qué es "lo más común" |
```python
import pandas as pd

# Calcular las tres en un DataFrame
media   = df['columna'].mean()
mediana = df['columna'].median()
moda    = df['columna'].mode()[0]

print(f"Media:   ${media:,.2f}")
print(f"Mediana: ${mediana:,.2f}")
print(f"Moda:    ${moda:,.2f}")

# O con describe() para ver todas las métricas
df['columna'].describe()
```

---

## <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExd3Z3NzNhbjh0a3R3b244dTVxcGZjb3AxZWkyMDFqeTQxODNwMnh1NSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/PDI7VTtxfx2rGRN3tm/giphy.gif" width="40"> Cómo leer la relación entre ellas

### Caso 1: Media ≈ Mediana ≈ Moda
### → Distribución simétrica / normal
```
     moda
     mediana
     media
      ↓
  ____↓____
 /         \
/           \
```

**Qué significa:**

Los datos están distribuidos de forma equilibrada.
No hay outliers significativos jalando los valores.
El "promedio" es representativo, puedes confiar en él.

**Ejemplos reales:**
- Temperatura diaria en una ciudad durante un mes normal
- Tiempo de entrega de un servicio muy estandarizado
- Calificaciones en un grupo estudiantil sin casos extremos

**En un proyecto de datos:**  
Puedes usar la media con confianza para reportes y proyecciones.
Si la media dice $50 de ticket promedio, la mayoría de los clientes realmente paga cerca de $50.

---

### Caso 2: Media > Mediana
### → Distribución sesgada a la derecha (sesgo positivo)
```
mediana media
    ↓   ↓
████░░░░░░░░░░░░░░░░░░░░···· outliers altos →
```

**Qué significa:**

Hay valores muy altos jalando la media hacia arriba.
La mayoría de los casos están por debajo del promedio.
La mediana es más 'honesta' que la media aquí.

**Ejemplos reales:**
- **Ingresos/Salarios:** La media salarial de un país puede ser $25,000/mes, pero la mediana es $8,000/mes
  porque unos pocos millonarios jalan el promedio hacia arriba. La mayoría gana mucho menos que 'el promedio'.
- **Ventas retail:** Media $90.51 vs Mediana $16.81 → Hay pocas transacciones mayoristas enormes que inflan el promedio.
  La mayoría de las compras son pequeñas.
- **Precios de vivienda:** Unos pocos penthouses suben el promedio del precio por metro cuadrado.
- **Tiempo de espera en urgencias:** La mayoría espera 30 min, pero algunos casos críticos esperan 8 horas y jalan el promedio hacia arriba.

**Insight de negocio:**
> ⚠️ Si usas la media para tomar decisiones, estás sobreestimando lo que es 'normal' en tu negocio.
> Usa la mediana para describir al cliente o caso típico.

**En posibles proyectos futuros**

*Datos SEMARNAT (medio ambiente):*
Si analizas emisiones de CO₂ por empresa, la media estará jalada por unas pocas industrias enormes.
La mediana te dice qué emite una empresa 'típica'.

*Datos OMS (salud):*
En mortalidad por enfermedad, la media puede estar inflada por países con sistemas de salud muy precarios.
La mediana representa mejor al 'país típico'.

*Datos INEGI (economía):*
En análisis de ingresos por hogar, siempre usa mediana. Es el estándar en economía precisamente por esto.

---

### Caso 3: Media < Mediana
### → Distribución sesgada a la izquierda (sesgo negativo)
```
outliers bajos ←···░░░░░░░░░░░░████
                      ↓  ↓
                   media mediana
```

**Qué significa:**

Hay valores muy bajos jalando la media hacia abajo.
La mayoría de los casos están por encima del promedio.

**Ejemplos reales:**
- **Calificaciones en un examen muy difícil:** La mayoría saca 70-90, pero unos pocos sacan 10-20 y jalan la media hacia abajo.

- **Tiempo de vida de productos con fallas tempranas:** La mayoría dura años, pero un lote defectuoso falla al primer mes y baja el promedio.

- **Temperaturas en un mes con una ola de frío:** La mayoría de los días fueron normales, pero una semana de frío extremo baja el promedio.

**Insight de negocio:**
> ⚠️ El promedio subestima lo que es 'normal'.
> Investiga qué está causando esos valores bajos, puede ser un problema de calidad, un evento atípico, o un segmento específico que necesita atención.

---

### Caso 4: Moda muy diferente de media y mediana
### → Hay grupos distintos en tus datos (bimodal o multimodal)
```
  pico 1        pico 2
    ↓              ↓
████            ████
████   ░░░░░░   ████
████░░░░░░░░░░░░████
```

**Qué significa:**

No tienes un grupo homogéneo. 
Tienes dos o más segmentos de clientes, productos o comportamientos completamente distintos mezclados en el mismo análisis.

**Ejemplos reales:**
- **Ticket de compra con dos modas: $15 y $450**
  → Tienes clientes minoristas Y mayoristas. Analizarlos juntos no tiene sentido.

- **Tiempo de ejercicio diario con moda en 0 y en 60 min**
  → Tienes personas sedentarias y deportistas. El 'promedio' de 30 min no describe a nadie.

- **Dosis de medicamento con dos picos**
  → Hay dos protocolos de tratamiento distintos.

**Insight de negocio:**
> Segmenta antes de analizar.
> Si ves dos modas, es casi siempre señal de que necesitas separar tus datos en grupos antes de sacar conclusiones.

---

## <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzMxY200dzhyc2RhbG5zbnk3MWZoZHE2ODZnM3I3Y2lseHNtYXI4diZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/WmvhTzEl9ihy9TqPme/giphy.gif" width="40"> Analogía con Medicina Tradicional China

En acupuntura, cuando diagnosticas a un paciente no te quedas con un solo síntoma, lees el patrón completo.

La media, mediana y moda funcionan igual:

| MTC | Estadística |
|---|---|
| Pulso promedio del paciente en reposo | Media |
| Síntoma que tiene con más frecuencia | Moda |
| Punto intermedio entre sus mejores y peores días | Mediana |
| Paciente con dolor crónico severo que sube el 'promedio de dolor' | Outlier que jala la media |

Ningún dato solo te da el diagnóstico completo.
Los tres juntos cuentan la historia real.

---

## ⚡ Cheat Sheet rápido
```
Media > Mediana  →  Outliers ALTOS   →  El promedio SOBREESTIMA
Media < Mediana  →  Outliers BAJOS   →  El promedio SUBESTIMA  
Media ≈ Mediana  →  Datos simétricos →  El promedio ES confiable
Dos modas        →  Dos grupos       →  SEGMENTA antes de analizar
```

---

## 🔗 Aplicado a mis proyectos

### Proyecto EDA - Ventas Retail
```python
media   = 90.51   # jalada por compras mayoristas enormes
mediana = 16.81   # ticket típico de la mayoría de clientes
# → Media > Mediana = sesgo positivo
# → Insight: no reportar la media como "ticket promedio"
#            usar mediana para describir al cliente típico
```

### Posibles proyectos futuros
| Dataset | Qué esperar | Por qué |
|---|---|---|
| INEGI - ingresos por hogar | Media >> Mediana | Desigualdad económica |
| OMS - mortalidad por país | Media > Mediana | Países con crisis extremas |
| SEMARNAT - emisiones CO₂ | Media >> Mediana | Pocas industrias enormes |
| Kaggle - detección de fraude | Moda muy baja | La mayoría son transacciones normales |

---

*Nota creada: 2026 · Parte del certificado Científico de Datos — EBAC*  
*🪷 github.com/ReginaPema*
