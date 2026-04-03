# ETL Process
## Extract, Transform, Load · Extracción, Transformación y Carga

> ETL es el proceso de tomar datos de donde están, limpiarlos y darles forma, y ponerlos donde los necesitas.
> Es lo que pasa ANTES de cualquier análisis. Sin ETL, no hay EDA. Sin EDA, no hay insights.

---

## 🗺️ El flujo completo
```
FUENTES DE DATOS           PROCESO ETL                DESTINO
─────────────────          ───────────────            ────────────────
                    
CSV                 →      [E] EXTRACT            →   Dataset limpio   
Excel               →      Extraer los datos          listo para:
Base de datos       →             ↓                   
API                 →      [T] TRANSFORM          →   • EDA       
JSON                →      Limpiar y transformar      • Dashboards              
                                  ↓                   • ML Models 
                           [L] LOAD               →   • Reportes
                           Cargar al destino             
                                   
```

---

## E: EXTRACT (Extracción)

### Qué es
El primer paso: **ir a buscar los datos** donde están y cargarlos en Python para poder trabajar con ellos.

### Para qué sirve
- Unificar datos que viven en distintos lugares
- Hacer accesibles datos que estaban "atrapados" en archivos, bases de datos o APIs
- Crear una copia de trabajo sin tocar los datos originales

### Tipos de fuentes y cómo extraerlas
```python
import pandas as pd

# ── ARCHIVOS PLANOS ──

# CSV: el más común
df = pd.read_csv('ventas.csv')
df = pd.read_csv('ventas.csv',
                 encoding='utf-8',        # importante para acentos
                 sep=';',                 # si el separador no es coma
                 decimal=',')             # si los decimales usan coma

# Excel: una o múltiples hojas
df = pd.read_excel('ventas.xlsx')
df = pd.read_excel('ventas.xlsx', sheet_name='Enero 2023')

# Múltiples hojas en un solo archivo
todas_hojas = pd.read_excel('ventas.xlsx', sheet_name=None)
for nombre_hoja, datos in todas_hojas.items():
    print(f"Hoja: {nombre_hoja} → {datos.shape[0]} filas")

# JSON
df = pd.read_json('datos.json')

# ── MÚLTIPLES ARCHIVOS A LA VEZ ──
import os

# Cargar todos los CSV de una carpeta
carpeta = 'datos/ventas_mensuales/'
archivos = [f for f in os.listdir(carpeta) if f.endswith('.csv')]
lista_dfs = [pd.read_csv(carpeta + archivo) for archivo in archivos]
df_completo = pd.concat(lista_dfs, ignore_index=True)
print(f"Total de filas combinadas: {df_completo.shape[0]}")

# ── BASES DE DATOS SQL ──
import sqlite3

conexion = sqlite3.connect('base_de_datos.db')
df = pd.read_sql_query("SELECT * FROM ventas WHERE año = 2023", conexion)
conexion.close()

# ── APIs (cuando llegues a ese nivel) ──
import requests

respuesta = requests.get('https://api.ejemplo.com/datos',
                         headers={'Authorization': 'Bearer TOKEN'})
datos = respuesta.json()
df = pd.DataFrame(datos)
```

### Buenas prácticas en la Extracción
```python
# 1. Siempre revisar inmediatamente después de cargar
def revisar_carga(df, nombre):
    print(f"\n{'='*50}")
    print(f"TABLA: {nombre}")
    print(f"{'='*50}")
    print(f"Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"\nPrimeras filas:")
    print(df.head(3))
    print(f"\nTipos de datos:")
    print(df.dtypes)
    print(f"\nValores nulos:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

# 2. Nunca modificar los datos originales
# Siempre trabajar sobre una copia
df_original = pd.read_csv('ventas.csv')
df = df_original.copy()   # ← trabajas sobre esta copia
```

---

## T: TRANSFORM (Transformación)

### Qué es
El paso más largo y más importante: **dar forma a los datos** para que sean útiles, consistentes y confiables.

### Para qué sirve
- Corregir errores e inconsistencias
- Unificar formatos distintos entre fuentes
- Crear nuevas variables útiles para el análisis
- Eliminar lo que no sirve
- Preparar los datos para que "hablen el mismo idioma"

### Las transformaciones más comunes

#### Limpieza básica
```python
# ── VALORES NULOS ──

# Identificar
print(df.isnull().sum())
print(f"Total nulos: {df.isnull().sum().sum()}")
print(f"% nulos: {df.isnull().sum().sum() / (df.shape[0]*df.shape[1])*100:.2f}%")

# Estrategias según el tipo de dato:

# Variables numéricas: rellenar con mediana (más robusto que media)
df['ventas'].fillna(df['ventas'].median(), inplace=True)

# Variables categóricas: rellenar con moda o 'Desconocido'
df['region'].fillna(df['region'].mode()[0], inplace=True)
df['categoria'].fillna('Desconocido', inplace=True)

# Eliminar filas donde el nulo es inaceptable
df.dropna(subset=['id_producto', 'fecha'], inplace=True)

# ── DUPLICADOS ──
print(f"Duplicados: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

# Duplicados solo en columnas clave
df.drop_duplicates(subset=['id_transaccion'], inplace=True)

# ── NOMBRES DE COLUMNAS ──
# Limpiar espacios y estandarizar
df.columns = (df.columns
                .str.strip()            # quitar espacios
                .str.lower()            # minúsculas
                .str.replace(' ', '_')  # espacios → guiones bajos
                .str.replace('á', 'a')  # quitar acentos
                .str.replace('é', 'e')
                .str.replace('í', 'i')
                .str.replace('ó', 'o')
                .str.replace('ú', 'u'))

# ── STRINGS / TEXTO ──
df['region'] = df['region'].str.strip()   # quitar espacios
df['region'] = df['region'].str.upper()   # estandarizar mayúsculas
df['marca']  = df['marca'].str.title()    # Primera Letra Mayúscula

# Corregir inconsistencias tipográficas
df['region'] = df['region'].replace({
    'MEXICO': 'Mexico',
    'MX': 'Mexico',
    'México': 'Mexico'
})
```

#### Transformaciones de fechas
```python
# Convertir strings a fechas
df['fecha'] = pd.to_datetime(df['fecha'])
df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y')

# Extraer componentes temporales
df['año']           = df['fecha'].dt.year
df['mes']           = df['fecha'].dt.month
df['trimestre']     = df['fecha'].dt.quarter
df['dia_semana']    = df['fecha'].dt.day_name()
df['es_fin_semana'] = df['fecha'].dt.dayofweek >= 5

# Nombre del mes en español
meses_es = {1:'Enero', 2:'Febrero', 3:'Marzo',
            4:'Abril', 5:'Mayo', 6:'Junio',
            7:'Julio', 8:'Agosto', 9:'Septiembre',
            10:'Octubre', 11:'Noviembre', 12:'Diciembre'}
df['mes_nombre'] = df['mes'].map(meses_es)
```

#### Transformaciones numéricas
```python
# Cambiar tipo de dato
df['precio'] = pd.to_numeric(df['precio'], errors='coerce')
df['unidades'] = df['unidades'].astype(int)

# Crear métricas calculadas
df['ingreso_total']    = df['precio'] * df['unidades']
df['precio_promedio']  = df['ingreso_total'] / df['unidades']
df['margen_estimado']  = df['ingreso_total'] * 0.30

# Categorizar variables numéricas
df['rango_venta'] = pd.cut(df['ventas'],
    bins=[0, 50, 200, 1000, float('inf')],
    labels=['Bajo', 'Medio', 'Alto', 'Premium'])

# Categorizar con percentiles
df['cuartil_venta'] = pd.qcut(df['ventas'],
    q=4,
    labels=['Q1_bajo', 'Q2_medio_bajo',
            'Q3_medio_alto', 'Q4_alto'])
```

#### Unir múltiples fuentes (el corazón del ETL)
```python
# LEFT JOIN: trae todo de la izquierda, agrega info de la derecha si existe
df_final = pd.merge(df_ventas,        # tabla principal
                    df_productos,     # tabla secundaria
                    on='id_producto',
                    how='left')

# Múltiples joins encadenados
df_final = (df_ventas
    .merge(df_productos,  on='id_producto',  how='left')
    .merge(df_categorias, on='id_categoria', how='left')
    .merge(df_regiones,   on='id_region',    how='left')
    .merge(df_calendario, on='fecha',        how='left'))

print(f"Dataset final: {df_final.shape}")

# Verificar que el merge no perdió filas
assert len(df_final) == len(df_ventas)
    "⚠️ El merge cambió el número de filas. Revisar duplicados en la clave"
```

#### Transformaciones estructurales
```python
# Eliminar columnas innecesarias o duplicadas
df.drop(columns=['col_duplicada', 'col_irrelevante'], inplace=True)

# Reordenar columnas lógicamente
cols_orden = ['fecha', 'año', 'mes', 'trimestre',
              'region', 'segmento', 'marca',
              'producto', 'precio', 'unidades', 'ventas_total']
df = df[cols_orden]

# Renombrar para mayor claridad
df.rename(columns={
    'TOTAL_VALUE_SALES': 'ventas_total',
    'TOTAL_UNIT_SALES':  'unidades_vendidas',
    'PRICE':             'precio_unitario'
}, inplace=True)

# Filtrar registros inválidos
df = df[df['precio'] > 0]             # eliminar precios cero o negativos
df = df[df['unidades'] >= 0]          # eliminar cantidades negativas
df = df[df['fecha'] >= '2020-01-01']  # solo datos recientes
```

---

## L — LOAD (Carga)

### Qué es
El último paso: **guardar el dataset transformado** en su destino final para que otros procesos puedan usarlo.

### Para qué sirve
- Hacer persistente el resultado de todo el trabajo
- Compartir el dataset limpio con otros
- Alimentar dashboards, modelos de ML, o reportes
- Tener un punto de partida confiable para futuros análisis

### Destinos comunes
```python
# ── ARCHIVOS ──

# CSV: el más universal
df.to_csv('data/processed/ventas_clean.csv',
          index=False,           # no guardar el índice
          encoding='utf-8-sig')  # para que Excel lo abra bien

# Excel: cuando el destino es un usuario no técnico
df.to_excel('data/processed/ventas_clean.xlsx',
            index=False,
            sheet_name='Ventas Limpias')

# Parquet: para datasets grandes (más eficiente que CSV)
df.to_parquet('data/processed/ventas_clean.parquet',
              index=False)

# ── BASES DE DATOS ──
import sqlite3

conexion = sqlite3.connect('base_de_datos.db')
df.to_sql('ventas_clean',         # nombre de la tabla
          conexion,
          if_exists='replace',    # replace / append / fail
          index=False)
conexion.close()
print("Datos cargados en la base de datos")

# ── VERIFICAR ANTES DE CERRAR ──
print(f"\nETL completado exitosamente")
print(f"   Filas finales:    {df.shape[0]:,}")
print(f"   Columnas finales: {df.shape[1]}")
print(f"   Nulos restantes:  {df.isnull().sum().sum()}")
print(f"   Duplicados:       {df.duplicated().sum()}")
print(f"   Archivo guardado: data/processed/ventas_clean.csv")
```

---

## 📁 Estructura de carpetas recomendada para ETL
```
proyecto/
├── data/
│   ├── raw/                  ← datos originales — NUNCA modificar
│   │   ├── ventas.csv
│   │   ├── productos.xlsx
│   │   └── categorias.csv
│   └── processed/            ← resultado del ETL
│       └── ventas_clean.csv
├── notebook/
│   └── etl_pipeline.ipynb    ← el proceso documentado
├── README.md
└── requirements.txt
```

> **Regla de oro:**
> Los datos en `raw/` son sagrados.
> Nunca los modifiques directamente.
> Siempre trabaja sobre una copia.

---

## 🔍 ETL vs ELT — la diferencia

Existe también el **ELT**: primero cargas, luego transformas.

| | ETL | ELT |
|---|---|---|
| **Orden** | Transforma → carga datos limpios | Carga datos crudos → transforma después |
| **Cuándo usarlo** | Datasets medianos, análisis en Python | Big Data, data warehouses en la nube |
| **Herramientas** | Python + Pandas | SQL, Spark, dbt, BigQuery |

---

## 🔎 Cómo documentar tu ETL en el notebook

Un ETL bien documentado debe responder estas preguntas:
```python
# Al inicio del notebook:
"""
ETL: Ventas Retail
==================
Fuentes:    5 archivos (3 CSV + 2 Excel)
Filas raw:  ~122,000 transacciones
Periodo:    Enero 2022 – Julio 2023
Objetivo:   Dataset unificado y limpio para EDA

Transformaciones aplicadas:
1. Carga y revisión de 5 fuentes independientes
2. Limpieza: nulos, duplicados, inconsistencias
3. Estandarización de nombres y formatos
4. Ingeniería de features temporales
5. Cálculo de métricas de ventas
6. Categorización de variables
7. Join de todas las fuentes en dataset único
8. Exportación del dataset final
"""
```

---

## <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzMxY200dzhyc2RhbG5zbnk3MWZoZHE2ODZnM3I3Y2lseHNtYXI4diZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/WmvhTzEl9ihy9TqPme/giphy.gif" width="40"> Analogía con Medicina Tradicional China

El proceso ETL es muy similar al proceso diagnóstico en MTC:

| ETL | Diagnóstico MTC |
|---|---|
| **Extract** → recopilar datos de múltiples fuentes | Recopilar síntomas: pulso, lengua, anamnesis, palpación |
| **Transform** → limpiar, unificar, dar coherencia | Interpretar síntomas según los patrones de la MTC |
| **Load** → diagnóstico final listo para tratamiento | Diagnóstico consolidado → protocolo de tratamiento |
| Datos crudos sin limpiar | Síntomas contradictorios sin interpretar |
| Dataset limpio | Diagnóstico claro y accionable |
| ETL mal hecho → análisis incorrecto | Diagnóstico incorrecto → tratamiento incorrecto |

> La calidad del análisis depende de la calidad del ETL.
> Igual que la calidad del tratamiento depende de la calidad del diagnóstico.

---

## ⚡ Cheat Sheet rápido
```
EXTRACT
├── pd.read_csv()        → archivos CSV
├── pd.read_excel()      → archivos Excel
├── pd.read_sql_query()  → bases de datos
└── pd.concat()          → combinar múltiples archivos

TRANSFORM — en este orden:
├── 1. Revisar: .info(), .describe(), .isnull().sum()
├── 2. Limpiar nombres: .columns.str.strip().str.lower()
├── 3. Tipos de datos: pd.to_datetime(), pd.to_numeric()
├── 4. Nulos: .fillna() o .dropna()
├── 5. Duplicados: .drop_duplicates()
├── 6. Strings: .str.strip(), .str.upper(), .replace()
├── 7. Features nuevas: columnas calculadas
├── 8. Joins: pd.merge()
└── 9. Verificación final

LOAD
├── .to_csv()       → universal
├── .to_excel()     → para usuarios de Excel
├── .to_parquet()   → datasets grandes
└── .to_sql()       → bases de datos

REGLAS DE ORO:
→ Nunca modificar datos raw
→ Siempre trabajar sobre .copy()
→ Documentar cada transformación
→ Verificar filas antes y después de cada join
→ Revisar nulos y duplicados al final
```

---

## 🔗 Aplicado a mis proyectos

### Proyecto 1: ETL Ventas Retail
```python
# Lo que hice en mi primer proyecto:

# EXTRACT:
# → 5 fuentes: DIM_CATEGORY, DIM_PRODUCT, DIM_SEGMENT, DIM_CALENDAR, FACT_SALES
# → Mezcla de CSV y Excel
# → ~122,000 filas en la tabla de hechos

# TRANSFORM (6 transformaciones):
# T1 → Eliminación de columnas duplicadas tras los joins
# T2 → Renombramiento y estandarización de columnas
# T3 → Feature engineering temporal (año, mes, trimestre)
# T4 → Métricas calculadas (ingreso total, precio promedio)
# T5 → Categorización de rangos de ventas
# T6 → Extracción de atributos de producto y región

# LOAD:
# → Dataset consolidado exportado a CSV
# → Usado como input del EDA

# Lo que aprendí:
# → Siempre verificar el shape después de cada merge
# → Los nombres de columnas con espacios son gran fuente de error en ETL con Pandas
# → Documentar cada transformación mientras la haces, no al final (después no recuerdas por qué la hiciste)
```

### Posibles proyectos futuros de ETL
| Dataset | Fuentes probables | Reto principal |
|---|---|---|
| SEMARNAT emisiones | CSV por estado + Excel por año | Unificar formatos entre años |
| OMS mortalidad | CSV + API | Manejo de API |
| INEGI ingresos | Múltiples CSV por entidad | Concatenar ~32 archivos |
| Kaggle fraude | CSV único | Desbalance de clases (pocos fraudes) |

---

*Nota creada: 2026 · Parte del certificado Científico de Datos — EBAC*
*🪷 github.com/ReginaPema*
