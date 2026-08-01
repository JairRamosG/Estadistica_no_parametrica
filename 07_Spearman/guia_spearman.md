# Guía de Referencia: Correlación de Spearman

## ¿Qué es?

La correlación de Spearman (ρ o `rho`) es una prueba **no paramétrica** que mide la **fuerza y dirección** de la relación **monotónica** entre dos variables. Usa **rangos** en lugar de valores originales.

**Pregunta central:** ¿Existe una tendencia consistente (monotónica) entre dos variables?

**Qué NO hace:**
- No asume normalidad en los datos
- No requiere relación lineal (solo monotónica)
- No es robusta ante outliers (es **resistente** a ellos)
- **No confunde causalidad con correlación** (ninguna correlación lo hace)

---

## Cuándo usarla

### Flujo de decisión

```
¿Querés medir la relación entre dos variables?
│
├── ¿La relación es claramente LINEAL?
│   │
│   ├── ¿Hay outliers significativos?
│   │   ├── Sí → SPEARMAN
│   │   └── No → PEARSON (más potente)
│   │
│   └── No → SPEARMAN
│
├── ¿Alguna variable es ORDINAL (Likert, ranking)?
│   └── SPEARMAN
│
└── ¿Las distribuciones son muy SESGADAS?
    └── SPEARMAN
```

### Regla práctica

| Situación | Herramienta |
|---|---|
| Relación lineal clara, sin outliers | Pearson |
| Relación monotónica no lineal | Spearman |
| Datos ordinales / rankings | Spearman |
| Outliers extremos | Spearman |
| Distribuciones muy sesgadas | Spearman |
| No sabés cuál usar | Spearman (más seguro) |

### ¿Por qué Spearman como default?

- Si la relación **es lineal**, Spearman la detecta igual (solo pierde un poco de potencia)
- Si la relación **no es lineal**, Pearson te miente o da un valor bajo
- **Spearman nunca te va a errar feo, Pearson sí puede**

---

## Pearson vs Spearman

### Comparación directa

| Aspecto | Pearson ($r$) | Spearman ($\rho$) |
|---|---|---|
| **Mide** | Relación lineal | Relación monotónica |
| **Usa** | Valores originales | Rangos |
| **Requiere** | Relación lineal | Solo tendencia |
| **Outliers** | Sensible | Resistente |
| **Datos ordinales** | No aplica | Sí aplica |
| **Potencia en lineal** | Mayor | Menor |
| **Rango de valores** | -1 a 1 | -1 a 1 |

### Interpretación del valor

| Valor de $\rho$ | Interpretación |
|---|---|
| 0.9 a 1.0 (o -0.9 a -1.0) | Correlación muy fuerte |
| 0.7 a 0.9 (o -0.7 a -0.9) | Correlación fuerte |
| 0.5 a 0.7 (o -0.5 a -0.7) | Correlación moderada |
| 0.3 a 0.5 (o -0.3 a -0.5) | Correlación débil |
| 0.0 a 0.3 (o 0.0 a -0.3) | Correlación muy débil o nula |

### ¿Qué significa "monotónica"?

Una relación monotónica es cuando **a medida que X aumenta, Y siempre aumenta (o siempre disminuye)**, aunque no sea una línea recta.

```
Monotónica positiva:     Monotónica negativa:     No monotónica:
    ↑                        ↓                       ↑
   /                        \                      / \
  /                          \                    /   \
 /                            \                  /     \
```

---

## Matemáticas detrás de Spearman

### Fórmula

$$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$

Donde:
- $d_i$ = diferencia entre los rangos de cada par de valores
- $n$ = número de observaciones

### Proceso

1. Rankear cada variable por separado
2. Calcular la diferencia ($d$) entre los rangos de cada par
3. Aplicar la fórmula

### Ejemplo manual

```python
# Datos
X = [10, 20, 30, 40, 50]
Y = [12, 25, 28, 45, 48]

# Rangos de X: [1, 2, 3, 4, 5]
# Rangos de Y: [1, 2, 3, 4, 5]

# d = [0, 0, 0, 0, 0]
# ρ = 1 - (6 * 0) / (5 * 24) = 1.0
```

---

## Ejemplos por fase del proyecto de datos

### 1. En EDA (Exploratory Data Analysis)

**Ejemplo 1: Relación tiempo de sesión vs compras en app**

> ¿Más tiempo de juego implica más compras?

```python
from pingouin import corr

resultados = corr(
    x=datos["tiempo_sesion"],
    y=datos["compras"],
    alternative="two-sided",
    method="spearman"
)

print(resultados)
# ρ = 0.96 → correlación positiva muy fuerte
# p < 0.05 → estadísticamente significativo
```

**Ejemplo 2: Detectar outliers antes de Pearson**

> ¿Mis datos tienen outliers que distorsionarían Pearson?

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Visualizar outliers
sns.scatterplot(data=datos, x='variable_x', y='variable_y')
plt.title('Scatter plot - Detectar outliers visualmente')
plt.show()

# Si hay outliers → Spearman
rho, p = stats.spearmanr(datos['variable_x'], datos['variable_y'])
print(f"Spearman: ρ = {rho:.3f}, p = {p:.6f}")
```

**Ejemplo 3: Variables ordinales**

> ¿Qué tan relacionadas están satisfacción (1-5) y NPS (-100 a 100)?

```python
# Nivel educativo: 1=Secundaria, 2=Terciario, 3=Universitario
# Satisfacción: 1-5 (Likert)

rho, p = stats.spearmanr(datos['nivel_educativo'], datos['satisfaccion'])
# Spearman es el único válido para ordinales
```

**Ejemplo 4: Matriz de correlación mixta**

> Dataset con variables continuas y ordinales

```python
import pandas as pd
import seaborn as sns

# Para variables mixtas, usar Spearman para todo
matriz_correlacion = datos.corr(method='spearman')

sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de Correlación (Spearman)')
plt.show()
```

---

### 2. En Feature Engineering

**Ejemplo 5: Evaluar relación entre features**

> ¿Qué features están altamente correlacionados? (multicolinealidad)

```python
# Spearman es mejor para detectar relaciones no lineales
corr_matrix = datos.corr(method='spearman')

# Identificar pares con alta correlación
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            print(f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.2f}")
```

**Ejemplo 6: Evaluar transformaciones**

> ¿Qué transformación crea la mejor relación con el target?

```python
import numpy as np

# Probar diferentes transformaciones
transformaciones = {
    'original': datos['ingreso'],
    'log': np.log1p(datos['ingreso']),
    'sqrt': np.sqrt(datos['ingreso']),
    'cuadrado': datos['ingreso'] ** 2
}

for nombre, transformacion in transformaciones.items():
    rho, p = stats.spearmanr(transformacion, datos['target'])
    print(f"{nombre}: ρ = {rho:.3f}, p = {p:.6f}")
```

**Ejemplo 7: Selección de features**

> ¿Qué features tienen mayor relación con el target?

```python
# Spearman para feature selection (robusto a outliers)
correlaciones = {}
for col in datos.columns:
    if col != 'target':
        rho, _ = stats.spearmanr(datos[col], datos['target'])
        correlaciones[col] = abs(rho)

# Ordenar por fuerza de correlación
correlaciones_ordenadas = sorted(correlaciones.items(), key=lambda x: x[1], reverse=True)
for feature, rho in correlaciones_ordenadas:
    print(f"{feature}: ρ = {rho:.3f}")
```

---

### 3. En Selección de Modelos

**Ejemplo 8: Evaluar relación entre métricas**

> ¿Qué tan relacionados están RMSE y MAE?

```python
# Si las métricas tienen outliers (modelos con errores extremos)
rho, p = stats.spearmanr(resultados['rmse'], resultados['mae'])
print(f"Correlación entre métricas: ρ = {rho:.3f}")

# Si ρ ≈ 1, son intercambiables
# Si ρ < 0.8, Considerá usar ambas
```

**Ejemplo 9: Comparar rendimiento de modelos**

> ¿El ranking de modelos es consistente entre datasets?

```python
# Modelo A y B en 10 datasets
rho, p = stats.spearmanr(ranking_modelo_a, ranking_modelo_b)
# ρ alto → el ranking es consistente
```

---

### 4. En Post-deploy / Monitoreo

**Ejemplo 10: Detectar drift**

> ¿La distribución de features cambió entre entrenamiento y producción?

```python
# Comparar rangos entre períodos
rho, p = stats.spearmanr(datos_entrenamiento['feature'], datos_produccion['feature'])
# ρ bajo → posible drift
```

**Ejemplo 11: Evaluar estabilidad de predicciones**

> ¿Las predicciones son estables a lo largo del tiempo?

```python
# Predicciones de enero vs febrero
rho, p = stats.spearmanr(predicciones_enero, predicciones_febrero)
print(f"Estabilidad: ρ = {rho:.3f}")
# ρ alto → predicciones estables
```

**Ejemplo 12: Correlación entre métricas de negocio**

> ¿Más usuarios activos implica más ingresos?

```python
rho, p = stats.spearmanr(datos['usuarios_activos'], datos['ingresos'])
# Spearman es ideal porque:
# 1. Puede haber outliers (días con picos)
# 2. La relación puede no ser lineal
```

---

### 5. En A/B Testing

**Ejemplo 13: Evaluar consistencia entre métricas**

> ¿Click-through rate y tiempo en página están relacionados?

```python
rho, p = stats.spearmanr(datos['ctr'], datos['tiempo_pagina'])
# Si ρ es alto, podés usar solo una métrica
```

**Ejemplo 14: Detectar efectos en subgrupos**

> ¿La relación entre dos variables cambia entre control y tratamiento?

```python
# Grupo control
rho_control, p_control = stats.spearmanr(
    datos[datos['grupo'] == 'control']['x'],
    datos[datos['grupo'] == 'control']['y']
)

# Grupo tratamiento
rho_tratamiento, p_tratamiento = stats.spearmanr(
    datos[datos['grupo'] == 'tratamiento']['x'],
    datos[datos['grupo'] == 'tratamiento']['y']
)

print(f"Control: ρ = {rho_control:.3f}")
print(f"Tratamiento: ρ = {rho_tratamiento:.3f}")
# Si difieren mucho, el tratamiento cambió la relación
```

---

### 6. En Validación de Modelos

**Ejemplo 15: Evaluar residuos**

> ¿Los residuos tienen relación con alguna variable?

```python
# Si los residuos tienen outliers, Spearman es mejor
rho, p = stats.spearmanr(datos['residuos'], datos['feature_x'])
# Si ρ ≈ 0, el modelo capturó bien la relación
```

**Ejemplo 16: Comparar modelos en múltiples datasets**

> ¿Qué modelo es consistentemente mejor?

```python
# Ranking de modelos por dataset
rankings = {
    'modelo_A': [1, 2, 1, 3, 2],
    'modelo_B': [2, 1, 3, 1, 1],
    'modelo_C': [3, 3, 2, 2, 3]
}

# Spearman entre rankings
rho_AB, _ = stats.spearmanr(rankings['modelo_A'], rankings['modelo_B'])
rho_AC, _ = stats.spearmanr(rankings['modelo_A'], rankings['modelo_C'])
rho_BC, _ = stats.spearmanr(rankings['modelo_B'], rankings['modelo_C'])

print(f"A vs B: ρ = {rho_AB:.3f}")
print(f"A vs C: ρ = {rho_AC:.3f}")
print(f"B vs C: ρ = {rho_BC:.3f}")
```

---

## Interpretación de resultados

### Salida de pingouin

```
            n         r         CI95%          p-val  power
spearman  183  0.965778  [0.95, 0.97]  5.094948e-108    1.0
```

**Cómo leerla:**
- **n**: número de observaciones
- **r**: valor de ρ (correlación de Spearman)
- **CI95%**: intervalo de confianza del 95%
- **p-val**: valor p
- **power**: potencia post-hoc

### Reglas de decisión

```
p < 0.01  → correlación estadísticamente significativa (muy fuerte)
p < 0.05  → correlación estadísticamente significativa (estándar)
p < 0.10  → posible tendencia (débil)
p ≥ 0.10  → sin evidencia de correlación
```

### ¿Qué reportar?

```python
# Formato completo
print(f"Correlación de Spearman: ρ = {rho:.3f}, IC95% = [{ci_low:.3f}, {ci_high:.3f}], p = {p:.6f}")

# Ejemplo:
# "Se encontró una correlación positiva muy fuerte y estadísticamente
# significativa entre tiempo de sesión y compras (ρ = 0.97, IC95% [0.95, 0.97],
# p < 0.001, n = 183)."
```

---

## Errores comunes

### 1. Usar Pearson cuando hay outliers

```python
# MAL: datos con outliers, usando Pearson
r, p = stats.pearsonr(datos['ingreso'], datos['gasto'])  # ❌ Outliers distorsionan

# BIEN: Spearman es resistente a outliers
rho, p = stats.spearmanr(datos['ingreso'], datos['gasto'])  # ✅
```

### 2. Usar Spearman en relación puramente lineal sin outliers

```python
# MAL: Spearman cuando Pearson tiene más potencia
rho, p = stats.spearmanr(datos['x_lineal'], datos['y_lineal'])  # ❌ Pierde potencia

# BIEN: Pearson para relaciones lineales claras
r, p = stats.pearsonr(datos['x_lineal'], datos['y_lineal'])  # ✅ Mayor potencia
```

### 3. Concluir causalidad

```python
# MAL: "Más tiempo de juego CAUSA más compras"
# BIEN: "Existe una correlación positiva entre tiempo de juego y compras"
```

### 4. Ignorar el valor p

```python
# MAL: "ρ = 0.3, hay correlación"
# BIEN: verificar si p < 0.05 antes de concluir
if p < 0.05:
    print(f"Correlación significativa: ρ = {rho:.3f}")
else:
    print("No hay evidencia de correlación")
```

### 5. No reportar el método usado

```python
# MAL: "la correlación es 0.9"
# BIEN: "la correlación de Spearman es 0.9"
```

### 6. Confundir monotónica con lineal

```python
# MAL: "ρ = 0.95 indica relación lineal"
# BIEN: "ρ = 0.95 indica relación monotónica fuerte (puede no ser lineal)"
```

### 7. Usar Spearman en datos nominales

```python
# MAL: Spearman con categorías sin orden (color: rojo, azul, verde)
# BIEN: Spearman solo para ordinales o continuas
```

---

## Flujo completo de código

```python
import pandas as pd
import numpy as np
from scipy import stats
from pingouin import corr
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================
# FLUJO COMPLETO: Análisis de correlación
# ============================================

# 1. Cargar datos
datos = pd.read_csv('tu_dataset.csv')

# 2. Exploración visual
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot
sns.scatterplot(data=datos, x='variable_x', y='variable_y', ax=axes[0])
axes[0].title.set_text('Scatter plot')

# Boxplot para detectar outliers
sns.boxplot(data=datos[['variable_x', 'variable_y']], ax=axes[1])
axes[1].title.set_text('Boxplots - Detectar outliers')

plt.tight_layout()
plt.show()

# 3. Decidir método
tiene_outliers = False  # Evaluá visualmente
es_ordinal = False      # Evaluá el tipo de dato

if tiene_outliers or es_ordinal:
    metodo = 'spearman'
    print("Usando Spearman (outliers o datos ordinales)")
else:
    metodo = 'pearson'
    print("Usando Pearson (relación lineal limpia)")

# 4. Calcular correlación
if metodo == 'spearman':
    rho, p = stats.spearmanr(datos['variable_x'], datos['variable_y'])
    print(f"Spearman: ρ = {rho:.3f}, p = {p:.6f}")
else:
    r, p = stats.pearsonr(datos['variable_x'], datos['variable_y'])
    print(f"Pearson: r = {r:.3f}, p = {p:.6f}")

# 5. Con pingouin (más detalles)
resultados = corr(
    x=datos['variable_x'],
    y=datos['variable_y'],
    alternative='two-sided',
    method=metodo
)
print(resultados)

# 6. Interpretar
alpha = 0.05
if p < alpha:
    print(f"\n✅ Correlación estadísticamente significativa (p = {p:.6f} < {alpha})")
    if metodo == 'spearman':
        if abs(rho) > 0.7:
            print("   Fuerza: fuerte")
        elif abs(rho) > 0.5:
            print("   Fuerza: moderada")
        else:
            print("   Fuerza: débil")
else:
    print(f"\n❌ No hay evidencia de correlación (p = {p:.6f} >= {alpha})")

# 7. Visualización final
plt.figure(figsize=(8, 6))
sns.regplot(data=datos, x='variable_x', y='variable_y', scatter_kws={'alpha':0.5})
plt.title(f'Correlación {metodo}: ρ = {rho:.3f}, p = {p:.6f}')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.show()
```

---

## Resumen: cuándo usar cada correlación

```
¿Qué querés medir?
│
├── Relación LINEAL entre dos variables continuas
│   │
│   ├── ¿Hay outliers? 
│   │   ├── Sí → SPEARMAN
│   │   └── No → PEARSON
│   │
│   └── ¿Los datos son normales?
│       ├── Sí → PEARSON
│       └── No → SPEARMAN
│
├── Relación MONOTÓNICA (tendencia, no necesariamente recta)
│   └── SPEARMAN
│
├── Datos ORDINALES (Likert, rankings)
│   └── SPEARMAN
│
└── No sabés
    └── SPEARMAN (más seguro)
```

---

## Código rápido de referencia

```python
from scipy import stats
from pingouin import corr

# ============================================
# SCIPY (básico)
# ============================================

# Spearman
rho, p = stats.spearmanr(datos['x'], datos['y'])
print(f"ρ = {rho:.3f}, p = {p:.6f}")

# Pearson
r, p = stats.pearsonr(datos['x'], datos['y'])
print(f"r = {r:.3f}, p = {p:.6f}")

# ============================================
# PINGOUIN (con intervalos de confianza)
# ============================================

# Spearman
resultados = corr(
    x=datos['x'],
    y=datos['y'],
    alternative='two-sided',
    method='spearman'
)
print(resultados)

# Pearson
resultados = corr(
    x=datos['x'],
    y=datos['y'],
    alternative='two-sided',
    method='pearson'
)
print(resultados)

# ============================================
# PANDAS (matrices de correlación)
# ============================================

# Matriz de Spearman
matriz_spearman = datos.corr(method='spearman')

# Matriz de Pearson
matriz_pearson = datos.corr(method='pearson')

# Visualizar
import seaborn as sns
sns.heatmap(matriz_spearman, annot=True, cmap='coolwarm', center=0)
```

---

## Checklist de análisis

| Paso | Acción |
|------|--------|
| 1 | ¿Qué tipo de datos tengo? (continuos, ordinales, nominales) |
| 2 | ¿Hay outliers? (visualizar con boxplot/scatter) |
| 3 | ¿La relación es lineal o monotónica? (visualizar scatter) |
| 4 | Elegir método: Pearson o Spearman |
| 5 | Calcular correlación y valor p |
| 6 | Interpretar: ¿es significativa? (p < 0.05) |
| 7 | Interpretar: ¿cuán fuerte es? (valor de ρ o r) |
| 8 | Reportar: método, valor, IC, p, n |
