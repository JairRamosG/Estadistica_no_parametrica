# Guía de Referencia: Prueba de Dunn (Post-hoc)

## ¿Qué es?

La prueba de Dunn es una prueba **post-hoc** no paramétrica que compara **pares de grupos** para determinar **cuáles específicamente** difieren entre sí, después de que Kruskal-Wallis ha indicado que existe al menos una diferencia.

**Pregunta central:** ¿Entre qué pares de grupos existe la diferencia detectada por Kruskal-Wallis?

**Qué NO hace:**
- No es una prueba inicial (siempre va **después** de Kruskal-Wallis)
- No compara medias (usa rangos)
- No asume normalidad
- **No corrige por sí misma** los errores tipo I por múltiples comparaciones (necesitás aplicar Bonferroni o Benjamini-Hochberg)

---

## Cuándo usarla

### Flujo completo

```
1. Kruskal-Wallis → ¿hay diferencia entre 3+ grupos?
   │
   ├── p > 0.05 → no hay diferencia → NO hacer post-hoc
   │
   └── p < 0.05 → SÍ hay diferencia → hacer post-hoc
        │
        2. Dunn's test → ¿entre cuáles pares de grupos hay diferencia?
```

### Regla práctica

| Situación | Herramienta |
|---|---|
| Kruskal-Wallis p > 0.05 | **No hacer post-hoc** |
| Kruskal-Wallis p < 0.05 | **Dunn's test con corrección** |
| Solo 2 grupos | Mann-Whitney U (no necesitás post-hoc) |

### ¿Por qué corregir?

Cuando hacés múltiples comparaciones, la probabilidad de cometer un error tipo I **aumenta**:

| Comparaciones | Probabilidad acumulada de error tipo I |
|---|---|
| 1 comparación | 5% |
| 2 comparaciones | 9.75% |
| 3 comparaciones | 14.26% |
| 6 comparaciones (3 grupos) | 26.49% |

**Las correcciones reducen esta probabilidad.**

---

## Correcciones disponibles

### Bonferroni (conservadora)

```
p_corregido = p × número de comparaciones
```

- **Ventaja:** muy conservadora, minimiza errores tipo I
- **Desventaja:** puede pasar por alto diferencias reales (aumenta errores tipo II)
- **Cuándo usarla:** cuando es CRÍTICO no cometer falsos positivos

### Benjamini-Hochberg (FDR, menos conservadora)

```
Controla la tasa de falsos descubrimientos (FDR)
```

- **Ventaja:** más potente, detecta más diferencias reales
- **Desventaja:** acepta algo más de falsos positivos
- **Cuándo usarla:** cuando querés balance entre detectar diferencias y controlar errores

### ¿Cuál elegir?

| Escenario | Corrección recomendada |
|---|---|
| Investigación médica, ensayos clínicos | Bonferroni (conservadora) |
| Ciencia de datos, exploración | Benjamini-Hochberg (FDR) |
| No sabés cuál usar | Empezar con Bonferroni |

---

## Ejemplos por fase del proyecto de datos

### 1. Después de Kruskal-Wallis en EDA

**Ejemplo 1: Comparar tiempos de respuesta entre servidores**

> Kruskal-Wallis dio p < 0.05. ¿Qué servidores específicamente difieren?

```python
from scikit_posthocs import posthoc_dunn

# Dunn's test con Bonferroni
posthoc = posthoc_dunn(
    datos_largo,
    val_col='tiempo',
    group_col='servidor',
    p_adjust='bonferroni'
)

print(posthoc)
# Interpretar: valores p < 0.05 indican qué pares difieren
```

**Ejemplo 2: Comparar ventas entre regiones**

> Kruskal-Wallis indica diferencias. ¿Qué regiones tienen ventas distintas?

```python
posthoc = posthoc_dunn(
    datos_largo,
    val_col='monto_venta',
    group_col='region',
    p_adjust='fdr_bh'  # Benjamini-Hochberg
)

print(posthoc)
# AWS vs Azure: p = 0.001 → difieren
# AWS vs GCloud: p = 0.023 → difieren
# Azure vs GCloud: p = 0.45 → NO difieren
```

**Ejemplo 3: Evaluar satisfacción según plan**

> ¿Qué planes tienen satisfacción significativamente distinta?

```python
posthoc = posthoc_dunn(
    datos_largo,
    val_col='satisfaccion',
    group_col='plan',
    p_adjust='bonferroni'
)

# Filtrar solo diferencias significativas
import numpy as np
sig = posthoc[posthoc < 0.05]
print(sig.dropna(how='all').dropna(axis=1, how='all'))
```

**Ejemplo 4: Comparar tiempos de carga entre navegadores**

> ¿Qué navegadores son significativamente diferentes?

```python
posthoc = posthoc_dunn(
    datos_largo,
    val_col='tiempo_carga',
    group_col='navegador',
    p_adjust='fdr_bh'
)
```

---

### 2. Después de Kruskal-Wallis en Feature Engineering

**Ejemplo 5: Evaluar bins de una variable**

> Tu binning de `ingreso` tiene 4 categorías. ¿Cuáles son realmente diferentes?

```python
posthoc = posthoc_dunn(
    datos,
    dv='target',
    between='bin_ingreso',
    p_adjust='bonferroni'
)
# Si bajo vs medio NO difiere → podés fusionarlos
```

**Ejemplo 6: Comparar efecto de transformaciones**

> ¿Qué transformaciones generan distribuciones significativamente distintas?

```python
posthoc = posthoc_dunn(
    datos_transformados,
    val_col='valor',
    group_col='transformacion',
    p_adjust='fdr_bh'
)
```

---

### 3. Después de Kruskal-Wallis en Selección de modelos

**Ejemplo 7: Comparar rendimiento de múltiples modelos**

> Kruskal-Wallis dice que hay diferencia. ¿Qué modelos específicamente difieren?

```python
posthoc = posthoc_dunn(
    errores_largo,
    val_col='error_absoluto',
    group_col='modelo',
    p_adjust='bonferroni'
)

# Identificar el mejor modelo (menor error)
print(posthoc)
```

**Ejemplo 8: Evaluar estabilidad entre datasets**

> ¿Qué datasets tienen rendimiento significativamente distinto?

```python
posthoc = posthoc_dunn(
    datos_largo,
    val_col='accuracy',
    group_col='dataset',
    p_adjust='fdr_bh'
)
```

**Ejemplo 9: Comparar hiperparámetros**

> ¿Qué estrategias de regularización son realmente diferentes?

```python
posthoc = posthoc_dunn(
    datos_largo,
    val_col='f1_score',
    group_col='regularizacion',
    p_adjust='bonferroni'
)
```

---

### 4. Después de Kruskal-Wallis en Post-deploy

**Ejemplo 10: Comparar rendimiento por horario**

> ¿En qué horarios el modelo se comporta significativamente distinto?

```python
posthoc = posthoc_dunn(
    datos_largo,
    val_col='error_prediccion',
    group_col='turno',
    p_adjust='fdr_bh'
)
```

**Ejemplo 11: Evaluar latencia por región**

> ¿Qué regiones tienen latencia significativamente distinta?

```python
posthoc = posthoc_dunn(
    datos_largo,
    val_col='latencia',
    group_col='region',
    p_adjust='bonferroni'
)
```

**Ejemplo 12: Comparar versiones de app**

> ¿Qué versiones tienen tiempos de carga diferentes?

```python
posthoc = posthoc_dunn(
    datos_largo,
    val_col='tiempo_carga',
    group_col='version',
    p_adjust='fdr_bh'
)
```

---

### 5. Después de Kruskal-Wallis en Monitoreo

**Ejemplo 13: Detectar drift en múltiples períodos**

> ¿Entre qué meses hay drift significativo?

```python
posthoc = posthoc_dunn(
    datos_largo,
    val_col='metrica_comportamiento',
    group_col='mes',
    p_adjust='bonferroni'
)
# Enero vs Marzo: p = 0.001 → drift
# Enero vs Febrero: p = 0.12 → no hay drift
# Febrero vs Marzo: p = 0.03 → drift
```

**Ejemplo 14: Evaluar estabilidad de features**

> ¿Qué features son inestables entre meses?

```python
for feature in ['feature_1', 'feature_2', 'feature_3']:
    posthoc = posthoc_dunn(
        datos_largo,
        val_col=feature,
        group_col='mes',
        p_adjust='fdr_bh'
    )
    # Identificar qué meses difieren para cada feature
```

**Ejemplo 15: Comparar fuentes de datos**

> ¿Qué fuentes son significativamente diferentes?

```python
posthoc = posthoc_dunn(
    datos_largo,
    val_col='valor',
    group_col='fuente',
    p_adjust='bonferroni'
)
```

---

### 6. Después de Kruskal-Wallis en A/B Testing

**Ejemplo 16: Comparar 4 versiones de botón**

> ¿Qué diseños de botón generan clicks significativamente distintos?

```python
posthoc = posthoc_dunn(
    datos_largo,
    val_col='clicks',
    group_col='version_boton',
    p_adjust='fdr_bh'
)
# Identificar el mejor diseño
```

**Ejemplo 17: Evaluar 3 algoritmos de recomendación**

> ¿Qué algoritmos tienen engagement diferente?

```python
posthoc = posthoc_dunn(
    datos_largo,
    val_col='engagement',
    group_col='algoritmo',
    p_adjust='bonferroni'
)
```

**Ejemplo 18: Comparar 5 estrategias de pricing**

> ¿Qué estrategias generan ingresos distintos?

```python
posthoc = posthoc_dunn(
    datos_largo,
    val_col='ingreso',
    group_col='estrategia',
    p_adjust='fdr_bh'
)
```

---

### 7. Después de Kruskal-Wallis en Validación de modelos

**Ejemplo 19: Evaluar residuos por categoría**

> ¿Para qué categorías de usuarios el modelo se comporta distinto?

```python
posthoc = posthoc_dunn(
    datos_largo,
    val_col='residuo',
    group_col='categoria_usuario',
    p_adjust='bonferroni'
)
```

**Ejemplo 20: Comparar errores por región**

> ¿En qué regiones los errores son significativamente mayores?

```python
posthoc = posthoc_dunn(
    datos_largo,
    val_col='error_absoluto',
    group_col='region',
    p_adjust='fdr_bh'
)
```

---

## Interpretación de resultados

### Matriz de valores p

```
         AWS    Azure   GCloud
AWS      1.00   0.001   0.023
Azure    0.001  1.00    0.045
GCloud   0.023  0.045   1.00
```

**Cómo leerla:**
- La **diagonal** siempre es 1.00 (un grupo vs sí mismo)
- Cada celda es el **p-valor corregido** para ese par de grupos
- **p < 0.05** → diferencia significativa entre esos dos grupos
- **p ≥ 0.05** → NO hay diferencia significativa

### Reglas de decisión

```
p < 0.01  → evidencia muy fuerte de diferencia
p < 0.05  → evidencia fuerte de diferencia (estándar)
p < 0.10  → evidencia débil, posible tendencia
p ≥ 0.10  → sin evidencia de diferencia
```

### Ejemplo de interpretación

```python
# Resultado:
# AWS vs Azure: p = 0.001 → SÍ difieren
# AWS vs GCloud: p = 0.023 → SÍ difieren
# Azure vs GCloud: p = 0.045 → SÍ difieren

# Interpretación:
# Todos los servidores tienen tiempos de respuesta significativamente
# diferentes entre sí (con corrección de Bonferroni, α = 0.05)
```

---

## Errores comunes

### 1. Hacer post-hoc cuando Kruskal-Wallis p > 0.05

```python
# MAL: Kruskal-Wallis p = 0.12, pero igual hacen post-hoc
resultado_kw = pg.kruskal(data=datos_largo, dv='valor', between='grupo')
if resultado_kw['p-unc'].values[0] > 0.05:
    # ❌ NO hacer post-hoc
    posthoc = posthoc_dunn(datos_largo, val_col='valor', group_col='grupo')

# BIEN: solo hacer post-hoc si Kruskal-Wallis p < 0.05
if resultado_kw['p-unc'].values[0] < 0.05:
    posthoc = posthoc_dunn(datos_largo, val_col='valor', group_col='grupo')
```

### 2. No aplicar corrección

```python
# MAL: sin corrección
posthoc = posthoc_dunn(datos_largo, val_col='valor', group_col='grupo')  # ❌

# BIEN: con corrección
posthoc = posthoc_dunn(datos_largo, val_col='valor', group_col='grupo', p_adjust='bonferroni')  # ✅
```

### 3. Usar Dunn's test como prueba inicial

```python
# MAL: sin Kruskal-Wallis previo
posthoc = posthoc_dunn(datos_largo, val_col='valor', group_col='grupo')  # ❌

# BIEN: primero Kruskal-Wallis, luego Dunn
resultado_kw = pg.kruskal(data=datos_largo, dv='valor', between='grupo')
if resultado_kw['p-unc'].values[0] < 0.05:
    posthoc = posthoc_dunn(datos_largo, val_col='valor', group_col='grupo', p_adjust='bonferroni')  # ✅
```

### 4. Concluir sobre la media

```python
# MAL: "la media del grupo A es mayor que la del grupo B"
# BIEN: "la distribución del grupo A tiende a tener valores mayores que la del grupo B"
```

### 5. Ignorar el tamaño del efecto

```python
# Resultado: p = 0.001 pero la diferencia práctica es mínima
# MAL: "hay una diferencia altamente significativa"
# BIEN: "hay una diferencia estadísticamente significativa, pero evaluar si es práctica"
```

### 6. No reportar qué corrección se usó

```python
# MAL: "el post-hoc de Dunn mostró diferencias"
# BIEN: "el post-hoc de Dunn con corrección de Bonferroni mostró diferencias"
```

Siempre reportá qué corrección aplicaste.

---

## Flujo completo de código

```python
import pingouin as pg
from scikit_posthocs import posthoc_dunn
import pandas as pd

# 1. Preparar datos en formato largo
datos_largo = datos.melt(var_name='grupo', value_name='valor')

# 2. Kruskal-Wallis
resultado_kw = pg.kruskal(
    data=datos_largo,
    dv='valor',
    between='grupo'
)

print(f"H: {resultado_kw['H'].values[0]:.4f}")
print(f"p: {resultado_kw['p-unc'].values[0]:.6f}")

# 3. Solo si p < 0.05, hacer post-hoc
if resultado_kw['p-unc'].values[0] < 0.05:
    print("\nPost-hoc de Dunn (Bonferroni):")
    posthoc = posthoc_dunn(
        datos_largo,
        val_col='valor',
        group_col='grupo',
        p_adjust='bonferroni'
    )
    print(posthoc)
    
    # Identificar pares significativos
    import numpy as np
    pares_sig = []
    for i in range(len(posthoc)):
        for j in range(i+1, len(posthoc)):
            p_val = posthoc.iloc[i, j]
            if p_val < 0.05:
                pares_sig.append((posthoc.index[i], posthoc.columns[j], p_val))
    
    print("\nPares con diferencia significativa:")
    for grupo1, grupo2, p in pares_sig:
        print(f"  {grupo1} vs {grupo2}: p = {p:.6f}")
else:
    print("No hay diferencia significativa → no hacer post-hoc")
```

---

## Comparación de correcciones

```python
# Bonferroni (conservadora)
posthoc_bonf = posthoc_dunn(
    datos_largo,
    val_col='valor',
    group_col='grupo',
    p_adjust='bonferroni'
)

# Benjamini-Hochberg (FDR, menos conservadora)
posthoc_bh = posthoc_dunn(
    datos_largo,
    val_col='valor',
    group_col='grupo',
    p_adjust='fdr_bh'
)

# Comparar resultados
print("Bonferroni:")
print(posthoc_bonf)
print("\nBenjamini-Hochberg:")
print(posthoc_bh)
```

**Diferencia típica:**
- Bonferroni: menos pares significativos (más conservadora)
- Benjamini-Hochberg: más pares significativos (más potente)

---

## Resumen: flujo completo de análisis no paramétrico

```
¿Tenés 3+ grupos independientes?
│
├── Sí → Kruskal-Wallis
│   │
│   ├── p > 0.05 → No hay diferencia → FIN
│   │
│   └── p < 0.05 → Hay diferencia → Dunn's test (post-hoc)
│       │
│       ├── Con Bonferroni → más conservador
│       └── Con Benjamini-Hochberg → más potente
│
└── No → ¿Son 2 grupos?
    ├── Independientes → Mann-Whitney U
    └── Emparejados → Wilcoxon pareado
```

---

## Código rápido de referencia

```python
from scikit_posthocs import posthoc_dunn

# Post-hoc de Dunn con Bonferroni
posthoc = posthoc_dunn(
    datos_largo,
    val_col='valor',
    group_col='grupo',
    p_adjust='bonferroni'
)

# Post-hoc de Dunn con Benjamini-Hochberg
posthoc = posthoc_dunn(
    datos_largo,
    val_col='valor',
    group_col='grupo',
    p_adjust='fdr_bh'
)

# Ver resultado
print(posthoc)

# Filtrar solo significativos
sig = posthoc[posthoc < 0.05]
print(sig.dropna(how='all').dropna(axis=1, how='all'))
```
