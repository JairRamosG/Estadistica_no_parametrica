# Guía de Referencia: Prueba de Kruskal-Wallis

## ¿Qué es?

La prueba de Kruskal-Wallis es una prueba **no paramétrica** que compara las **distribuciones** de **tres o más muestras independientes** para determinar si provienen de la misma distribución.

Es considerada la versión no paramétrica del **ANOVA de un factor**.

**Pregunta central:** ¿Al menos uno de los grupos tiene una distribución diferente a los demás?

**Qué NO hace:**
- No compara medias (usa rangos, no valores absolutos)
- No asume normalidad
- No asume varianzas iguales
- No te dice **cuáles** grupos difieren (para eso necesitás un post-hoc)

---

## Cuándo usarla

### Regla práctica

| Situación | Herramienta |
|---|---|
| Dos grupos **independientes**, datos **NO normales** | Mann-Whitney U |
| Dos grupos **emparejados**, datos **NO normales** | Wilcoxon pareado |
| Tres o más grupos **independientes**, datos **NO normales** | **Kruskal-Wallis** |
| Tres o más grupos **emparejados**, datos **NO normales** | Friedman |
| Tres o más grupos **independientes**, datos **normales** | ANOVA |

### Condiciones ideales

1. **Tres o más grupos independientes** — cada grupo tiene datos que no están relacionados con los demás
2. **Datos ordinales o continuos** — la prueba trabaja con rangos
3. **Distribuciones con forma similar** — las distribuciones deben tener forma parecida. Si las formas son muy distintas, Kruskal-Wallis compara distribuciones completas, no solo medianas

### Flujo de decisión

```
¿Tenés 3+ grupos?
│
├── ¿Son independientes?
│   ├── ¿Son normales? → ANOVA
│   └── ¿NO son normales? → Kruskal-Wallis
│
└── ¿Son emparejados?
    ├── ¿Son normales? → ANOVA de medidas repetidas
    └── ¿NO son normales? → Friedman
```

---

## Ejemplos por fase del proyecto de datos

### 1. Exploración de datos (EDA)

**Ejemplo 1: Comparar tiempo de respuesta entre servidores**

> Tenés datos de tiempos de respuesta de una API desplegada en AWS, Azure y Google Cloud. ¿El tipo de servidor afecta el tiempo de respuesta?

```python
import pingouin as pg

# Formato largo
datos_largo = datos.melt(var_name="servidor", value_name="tiempo")

# Kruskal-Wallis
resultado = pg.kruskal(
    data=datos_largo,
    dv='tiempo',
    between='servidor'
)
# p < 0.05 → hay diferencia significativa entre al menos dos servidores
```

**Ejemplo 2: Comparar ventas entre regiones**

> Tenés ventas de 4 regiones. ¿El monto de venta difiere según la región?

```python
resultado = pg.kruskal(
    data=datos_largo,
    dv='monto_venta',
    between='region'
)
# p < 0.05 → al menos una región tiene ventas significativamente distintas
```

**Ejemplo 3: Evaluar satisfacción según plan de servicio**

> ¿Los usuarios del plan Free, Basic y Premium tienen niveles de satisfacción diferentes?

```python
resultado = pg.kruskal(
    data=datos_largo,
    dv='satisfaccion',
    between='plan'
)
# p < 0.05 → el plan influye en la satisfacción
```

**Ejemplo 4: Comparar tiempos de carga entre navegadores**

> ¿Chrome, Firefox y Safari tienen tiempos de carga distintos para tu sitio?

```python
resultado = pg.kruskal(
    data=datos_largo,
    dv='tiempo_carga',
    between='navegador'
)
# p < 0.05 → el navegador afecta el tiempo de carga
```

---

### 2. Feature Engineering

**Ejemplo 5: Evaluar si una feature categórica separa bien**

> Creaste una feature `rango_edad` con 4 categorías. ¿Las distribuciones de `gasto` son distintas por rango?

```python
resultado = pg.kruskal(
    data=datos,
    dv='gasto',
    between='rango_edad'
)
# p < 0.05 → la feature tiene poder discriminativo
```

**Ejemplo 6: Comparar efecto de múltiples transformaciones**

> Aplicaste log, Box-Cox y Yeo-Johnson a una variable. ¿Las distribuciones resultantes son diferentes?

```python
import numpy as np
from scipy.stats import boxcox, yeojohnson

datos['log_precio'] = np.log1p(datos['precio'])
datos['boxcox_precio'], _ = boxcox(datos['precio'] + 1)
datos['yeojohnson_precio'], _ = yeojohnson(datos['precio'] + 1)

datos_transformados = datos[['log_precio', 'boxcox_precio', 'yeojohnson_precio']].melt(
    var_name='transformacion', value_name='valor'
)

resultado = pg.kruskal(
    data=datos_transformados,
    dv='valor',
    between='transformacion'
)
# p < 0.05 → las transformaciones generan distribuciones distintas
# p > 0.05 → las transformaciones preservan la estructura
```

**Ejemplo 7: Evaluar bins de una variable continua**

> Agrupaste `ingreso` en 3 categorías: bajo, medio, alto. ¿Los bins son realmente distintos en términos de la variable target?

```python
resultado = pg.kruskal(
    data=datos,
    dv='target',
    between='bin_ingreso'
)
# p < 0.05 → el binning tiene poder discriminativo
```

---

### 3. Selección de modelos

**Ejemplo 8: Comparar rendimiento de múltiples modelos**

> Tenés 4 modelos y querés saber si hay diferencia en sus errores absolutos.

```python
# Errores de cada modelo
errores = pd.DataFrame({
    'modelo_A': abs(y_real - y_pred_A),
    'modelo_B': abs(y_real - y_pred_B),
    'modelo_C': abs(y_real - y_pred_C),
    'modelo_D': abs(y_real - y_pred_D)
})

errores_largo = errores.melt(var_name='modelo', value_name='error_absoluto')

resultado = pg.kruskal(
    data=errores_largo,
    dv='error_absoluto',
    between='modelo'
)
# p < 0.05 → al menos un modelo tiene errores significativamente distintos
```

**Ejemplo 9: Evaluar estabilidad de un modelo entre datasets**

> Validaste tu modelo en 5 datasets diferentes. ¿El rendimiento es estable?

```python
resultado = pg.kruskal(
    data=datos_largo,
    dv='accuracy',
    between='dataset'
)
# p > 0.05 → el modelo es estable entre datasets
# p < 0.05 → el rendimiento varía según el dataset
```

**Ejemplo 10: Comparar hiperparámetros**

> Probaste 3 estrategias de regularización. ¿Cuál da mejor rendimiento?

```python
resultado = pg.kruskal(
    data=datos_largo,
    dv='f1_score',
    between='regularizacion'
)
# p < 0.05 → la estrategia de regularización importa
```

---

### 4. Evaluación post-deploy

**Ejemplo 11: Comparar rendimiento de un modelo en producción por horario**

> ¿El modelo se comporta igual en mañana, tarde y noche?

```python
resultado = pg.kruskal(
    data=datos_largo,
    dv='error_prediccion',
    between='turno'
)
# p > 0.05 → el modelo es estable durante el día
# p < 0.05 → el rendimiento varía según el turno
```

**Ejemplo 12: Evaluar latencia de API en múltiples regiones**

> ¿La latencia de tu API difiere entre Norteamérica, Europa y Asia?

```python
resultado = pg.kruskal(
    data=datos_largo,
    dv='latencia',
    between='region'
)
# p < 0.05 → la región afecta la latencia
```

**Ejemplo 13: Comparar tiempos de respuesta de diferentes versiones**

> ¿Tu app versión 1, 2 y 3 tienen tiempos de carga diferentes?

```python
resultado = pg.kruskal(
    data=datos_largo,
    dv='tiempo_carga',
    between='version'
)
# p < 0.05 → al menos una versión tiene tiempos distintos
```

---

### 5. Monitoreo y data drift

**Ejemplo 14: Detectar drift en múltiples períodos**

> Comparaste el comportamiento de usuarios en enero, febrero y marzo. ¿Cambiaron los patrones?

```python
resultado = pg.kruskal(
    data=datos_largo,
    dv='metrica_comportamiento',
    between='mes'
)
# p < 0.05 → hay drift entre los meses
```

**Ejemplo 15: Evaluar estabilidad de features en producción**

> ¿Las distribuciones de tus features son estables en 3 meses de producción?

```python
for feature in ['feature_1', 'feature_2', 'feature_3']:
    resultado = pg.kruskal(
        data=datos_largo,
        dv=feature,
        between='mes'
    )
    print(f"{feature}: p = {resultado['p-unc'].values[0]:.4f}")
# Si todas las p > 0.05 → las features son estables
```

**Ejemplo 16: Comparar distribuciones entre múltiples fuentes**

> Tenés datos de 4 fuentes distintas. ¿Las distribuciones son comparables?

```python
resultado = pg.kruskal(
    data=datos_largo,
    dv='valor',
    between='fuente'
)
# p > 0.05 → las fuentes son comparables
# p < 0.05 → hay diferencias entre fuentes (posible sesgo)
```

---

### 6. A/B Testing (con múltiples variantes)

**Ejemplo 17: Comparar 4 versiones de un botón**

> Probaste 4 diseños de botón de compra. ¿Alguno genera más clicks?

```python
resultado = pg.kruskal(
    data=datos_largo,
    dv='clicks',
    between='version_boton'
)
# p < 0.05 → al menos un diseño es diferente
# → Post-hoc para saber cuál es mejor
```

**Ejemplo 18: Evaluar 3 algoritmos de recomendación**

> ¿Los usuarios interactúan diferente con 3 algoritmos distintos?

```python
resultado = pg.kruskal(
    data=datos_largo,
    dv='engagement',
    between='algoritmo'
)
# p < 0.05 → el algoritmo influye en el engagement
```

**Ejemplo 19: Comparar 5 estrategias de pricing**

> ¿5 estrategias de precio generan ingresos diferentes?

```python
resultado = pg.kruskal(
    data=datos_largo,
    dv='ingreso',
    between='estrategia'
)
# p < 0.05 → la estrategia de pricing importa
```

---

### 7. Validación de supuestos en modelos

**Ejemplo 20: Evaluar residuos en múltiples grupos**

> ¿Los residuos de tu modelo son similares para diferentes categorías de usuarios?

```python
resultado = pg.kruskal(
    data=datos_largo,
    dv='residuo',
    between='categoria_usuario'
)
# p > 0.05 → el modelo se comporta similar para todos los grupos
# p < 0.05 → hay sesgo en el modelo según la categoría
```

**Ejemplo 21: Comparar distribuciones de errores por región**

> ¿Los errores del modelo varían según la región geográfica?

```python
resultado = pg.kruskal(
    data=datos_largo,
    dv='error_absoluto',
    between='region'
)
# p < 0.05 → los errores varían por región → considerar modelo por separado
```

---

## Post-hoc: Dunn's test

Cuando Kruskal-Wallis da p < 0.05, necesitás saber **qué grupos específicos** difieren. Para eso usás el test de Dunn con corrección de Bonferroni:

```python
from scikit_posthocs import posthoc_dunn

# Dunn's test
posthoc = posthoc_dunn(
    datos_largo,
    val_col='tiempo',
    group_col='servidor',
    p_adjust='bonferroni'
)

print(posthoc)
# Valores p < 0.05 indican qué pares de grupos difieren
```

### Interpretación del post-hoc

```
         AWS    Azure   GCloud
AWS      1.00   0.001   0.023
Azure    0.001  1.00    0.045
GCloud   0.023  0.045   1.00

→ AWS vs Azure: p = 0.001 (diferencia significativa)
→ AWS vs GCloud: p = 0.023 (diferencia significativa)
→ Azure vs GCloud: p = 0.045 (diferencia significativa)
```

---

## Interpretación de resultados

### Valores que devuelve la prueba

| Salida | Qué significa |
|---|---|
| **H** | Estadística de Kruskal-Wallis. Valores altos indican diferencias |
| **p-unc** | Probabilidad de observar esta diferencia si H₀ fuera cierta |
| **ddof1** | Grados de libertad (número de grupos - 1) |

### Reglas de decisión para p-valor

```
p < 0.01  → evidencia muy fuerte contra H₀
p < 0.05  → evidencia fuerte contra H₀ (estándar)
p < 0.10  → evidencia débil, posible tendencia
p ≥ 0.10  → sin evidencia suficiente
```

### Tamaño del efecto (ε²)

$$\epsilon^2 = \frac{H}{N-1}$$

Donde H es la estadística de Kruskal-Wallis y N es el total de observaciones.

```
ε² ≤ 0.08  → efecto pequeño
ε² ≤ 0.26  → efecto mediano
ε² > 0.26  → efecto grande
```

### Regla importante

> **Kruskal-Wallis solo te dice SI hay diferencia, NO CUÁL.** Siempre seguí con un post-hoc (Dunn's test) cuando p < 0.05.

---

## Errores comunes

### 1. Usar Kruskal-Wallis con solo 2 grupos

```python
# MAL: solo 2 grupos
pg.kruskal(data=datos_largo, dv='valor', between='grupo')  # ❌

# BIEN: usar Mann-Whitney U para 2 grupos
pg.mwu(x=datos_grupo_A, y=datos_grupo_B)  # ✅
```

### 2. No hacer post-hoc después de Kruskal-Wallis

```python
# MAL: concluir sin post-hoc
resultado = pg.kruskal(data=datos_largo, dv='valor', between='grupo')
print("Hay diferencia")  # ❌ ¿Entre cuáles?

# BIEN: hacer post-hoc
if resultado['p-unc'].values[0] < 0.05:
    posthoc = posthoc_dunn(datos_largo, val_col='valor', group_col='grupo', p_adjust='bonferroni')
    print(posthoc)  # ✅
```

### 3. Concluir sobre la media

```python
# MAL: "la media del grupo A es mayor que la del grupo B"
# BIEN: "la distribución del grupo A tiende a tener valores mayores que la del grupo B"
```

### 4. Ignorar el tamaño del efecto

```python
# Resultado: p = 0.001, ε² = 0.02
# MAL: "hay una diferencia altamente significativa"
# BIEN: "hay una diferencia estadísticamente significativa pero con un efecto muy pequeño (ε² = 0.02)"
```

### 5. Usar Kruskal-Wallis cuando los datos son emparejados

```python
# MAL: mismos usuarios medidos en 3 momentos
pg.kruskal(data=datos_largo, dv='valor', between='momento')  # ❌

# BIEN: usar Friedman
pg.friedman(data=datos_largo, dv='valor', within='momento', subject='usuario')  # ✅
```

### 6. No verificar la forma de las distribuciones

Si las distribuciones tienen formas **muy** distintas (una sesgada a la derecha, otra a la izquierda), Kruskal-Wallis compara las **distribuciones completas**, no solo las medianas. Podría dar p < 0.05 aunque las medianas sean iguales.

```python
# Verificar forma con boxplots
import seaborn as sns
sns.boxplot(data=datos_largo, x='grupo', y='valor')
```

---

## Fórmulas de referencia

### Cálculo del tamaño de muestra

1. Definir el tamaño del efecto deseado (ε²)
2. Convertir a f de Cohen: $f = \sqrt{\frac{\epsilon^2}{1-\epsilon^2}}$
3. Usar `FTestAnovaPower` para estimar $n_{ANOVA}$
4. Corregir: $n_{KW} \approx \frac{n_{ANOVA}}{0.85}$

### Código de cálculo de muestra

```python
from statsmodels.stats.power import FTestAnovaPower
import math

# Diseño del experimento
e2 = 0.2     # Tamaño del efecto deseado
alpha = 0.05
power = 0.9
N = 3        # Número de grupos

# ANOVA: cálculo del número de muestras por grupo
f_anova = math.sqrt(e2 / (1 - e2))
analisis = FTestAnovaPower()
n_anova = analisis.solve_power(
    effect_size=f_anova,
    alpha=alpha,
    power=power,
    k_groups=N
)

# Kruskal-Wallis: ajuste
n_kw = math.ceil(n_anova / 0.85)

print(f"Tamaño efecto ANOVA (f): {f_anova:.2f}")
print(f"Muestra ANOVA: {math.ceil(n_anova)}")
print(f"Muestra Kruskal-Wallis: {n_kw}")
```

### Potencia post-hoc

```python
import math
from statsmodels.stats.power import FTestAnovaPower

# Calcular ε² post-hoc
N_total = len(datos_largo)
e2_posthoc = resultado['H'].values[0] / (N_total - 1)

# Convertir a f de Cohen
f_cohen = math.sqrt(e2_posthoc / (1 - e2_posthoc))

# Calcular potencia
analisis = FTestAnovaPower()
potencia = analisis.power(
    effect_size=f_cohen,
    nobs=N_total / N_grupos,  # muestras por grupo
    alpha=0.05,
    k_groups=N_grupos
)

print(f"ε² post-hoc: {e2_posthoc:.4f}")
print(f"Potencia: {potencia:.4f}")
```

---

## Código rápido de referencia

```python
import pingouin as pg
import pandas as pd
from scikit_posthocs import posthoc_dunn

# Preparar datos en formato largo
datos_largo = datos.melt(var_name='grupo', value_name='valor')

# Kruskal-Wallis
resultado = pg.kruskal(
    data=datos_largo,
    dv='valor',
    between='grupo'
)

print(f"H: {resultado['H'].values[0]:.4f}")
print(f"p: {resultado['p-unc'].values[0]:.6f}")

# Tamaño del efecto
N_total = len(datos_largo)
e2 = resultado['H'].values[0] / (N_total - 1)
print(f"ε²: {e2:.4f}")

# Decisión
alpha = 0.05
if resultado['p-unc'].values[0] < alpha:
    print("Rechazar H₀ → hay diferencia significativa")
    
    # Post-hoc de Dunn
    posthoc = posthoc_dunn(
        datos_largo,
        val_col='valor',
        group_col='grupo',
        p_adjust='bonferroni'
    )
    print("\nPost-hoc (Dunn):")
    print(posthoc)
else:
    print("No rechazar H₀ → no hay diferencia significativa")
```

---

## Resumen: ¿cuándo usar cada prueba?

```
¿Tenés 2 o más grupos?
│
├── 2 grupos
│   ├── Independientes → Mann-Whitney U
│   └── Emparejados → Wilcoxon pareado
│
├── 3+ grupos
│   ├── Independientes → Kruskal-Wallis (+ Dunn post-hoc)
│   └── Emparejados → Friedman (+ post-hoc correspondiente)
│
└── ¿Querés comparar con un valor fijo?
    ├── 1 muestra → Wilcoxon una muestra
    └── 1 muestra normal → t-test una muestra
```

---

## Comparación con pruebas paramétricas

| No paramétrica | Paramétrica | Cuándo usar la no paramétrica |
|---|---|---|
| Mann-Whitney U | t-test independiente | Datos NO normales, 2 grupos independientes |
| Wilcoxon pareado | t-test pareado | Datos NO normales, 2 grupos emparejados |
| **Kruskal-Wallis** | ANOVA | Datos NO normales, 3+ grupos independientes |
| Friedman | ANOVA medidas repetidas | Datos NO normales, 3+ grupos emparejados |

---

## Ejemplo completo: tu tesis

Si en tu tesis estás comparando **múltiples algoritmos** en el **mismo conjunto de datos** (emparejado):

```python
import pingouin as pg

# Si los algoritmos se evalúan en los MISMOS datos (emparejado)
resultado = pg.friedman(
    data=datos_largo,
    dv='metrica_rendimiento',
    within='algoritmo',
    subject='dataset'  # o 'fold', 'instancia', etc.
)

# Si los algoritmos se evalúan en datos DIFERENTES (independiente)
resultado = pg.kruskal(
    data=datos_largo,
    dv='metrica_rendimiento',
    between='algoritmo'
)
```

**La clave:** ¿cada algoritmo se probó en los mismos datos o en datos diferentes Eso define si usás Kruskal-Wallis (independiente) o Friedman (emparejado).
