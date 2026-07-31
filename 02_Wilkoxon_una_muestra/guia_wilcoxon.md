# Guía de Referencia: Prueba de Wilcoxon de Rangos con Signo (Una Muestra)

## ¿Qué es?

La prueba de Wilcoxon de rangos con signo para una muestra es una prueba **no paramétrica** que compara la **mediana** de una muestra contra un **valor de referencia conocido**.

**Pregunta central:** ¿la mediana de mis datos es igual a un valor conocido (por ejemplo, 200ms)?

**Qué NO hace:**
- No compara medias (usa rangos de las diferencias)
- No asume normalidad
- No compara dos grupos (para eso está Mann-Whitney o Wilcoxon pareado)

---

## Cuándo usarla

### Regla práctica

| Situación | Herramienta |
|---|---|
| Un grupo vs. valor fijo, datos **normales** | t-test una muestra |
| Un grupo vs. valor fijo, datos **NO normales** | **Wilcoxon una muestra** |
| Dos grupos emparejados | Wilcoxon pareado |
| Dos grupos independientes | Mann-Whitney U |

### Condiciones ideales

1. **Datos ordinales o continuos** — la prueba trabaja con rangos
2. **Diferencias simétricas** — las diferencias entre los datos y el valor de referencia deben ser aproximadamente simétricas
3. **Independencia** — los datos dentro de la muestra son independientes entre sí

### Cómo funciona

No compara los datos directamente contra el valor de referencia. Calcula las **diferencias** y testea si esas diferencias son simétricas alrededor de cero:

```python
datos = [190, 210, 185, 205, 195]
referencia = 200

diferencias = datos - referencia  # [-10, 10, -15, 5, -5]
# Wilcoxon testea si estas diferencias son simétricas alrededor de 0
```

---

## Ejemplos por fase del proyecto de datos

### 1. Exploración de datos (EDA)

**Ejemplo 1: Comparar latencia de un modelo contra un benchmark**

> Tu modelo actual tiene una latencia mediana de 200ms. ¿El nuevo modelo es más rápido?

```python
from scipy.stats import wilcoxon

latencia_nueva = [185, 192, 178, 200, 195, ...]

W, p = wilcoxon(latencia_nueva - 200, alternative='less')
# p < 0.05 → el nuevo modelo es estadísticamente más rápido
```

**Ejemplo 2: Verificar si una métrica cumple un SLA**

> Tu SLA dice que el tiempo de respuesta debe ser ≤ 150ms. ¿Los datos cumplen?

```python
W, p = wilcoxon(tiempos_respuesta - 150, alternative='greater')
# p < 0.05 → los tiempos son significativamente mayores a 150ms (no cumple)
```

**Ejemplo 3: Comparar contra un valor histórico**

> El año pasado el tiempo promedio de entrega era de 48 horas. ¿Este año mejoró?

```python
W, p = wilcoxon(tiempos_entrega_este_año - 48, alternative='less')
# p < 0.05 → los tiempos este año son menores
```

**Ejemplo 4: Validar si una distribución es simétrica**

> Querés saber si `tiempo_respuesta` es simétrica alrededor de su mediana.

```python
mediana = df['tiempo_respuesta'].median()
W, p = wilcoxon(df['tiempo_respuesta'] - mediana, alternative='two-sided')
# p > 0.05 → no se rechaza simetría (buena señal)
# p < 0.05 → la distribución es asimétrica
```

---

### 2. Preprocesamiento y limpieza

**Ejemplo 5: Validar que una transformación no cambió la mediana**

> Aplicaste log-transform a `ingreso`. ¿La mediana se mantuvo?

```python
import numpy as np

W, p = wilcoxon(
    np.log1p(df['ingreso']) - df['ingreso'].median(),
    alternative='two-sided'
)
# p > 0.05 → la transformación no desplazó la mediana (bueno)
```

**Ejemplo 6: Detectar sesgo en imputación de nulos**

> Imputaste los valores faltantes de `edad` con la mediana. ¿La distribución imputada es consistente?

```python
W, p = wilcoxon(
    datos_imputados['edad'] - datos_originales['edad'].median(),
    alternative='two-sided'
)
```

**Ejemplo 7: Comparar datos antes y después de outlier removal**

> Filtraste outliers con IQR. ¿La mediana cambió?

```python
W, p = wilcoxon(
    datos_crudos['precio'] - datos_limpios['precio'].median(),
    alternative='two-sided'
)
# p > 0.05 → la mediana se preservó (bueno)
# p < 0.05 → el filtrado cambió la mediana (cuidado)
```

---

### 3. Feature engineering

**Ejemplo 8: Validar si un feature tiene poder discriminativo**

> ¿El tiempo de sesión de los usuarios que compran es mayor que 120 segundos?

```python
W, p = wilcoxon(tiempos_sesion_compradores - 120, alternative='greater')
# p < 0.05 → los compradores pasan más de 120 segundos
```

**Ejemplo 9: Verificar si una feature está dentro de un rango esperado**

> La variable `edad` debería tener una mediana de 35 años en tu dataset.

```python
W, p = wilcoxon(df['edad'] - 35, alternative='two-sided')
# p > 0.05 → la mediana es consistente con lo esperado
```

**Ejemplo 10: Evaluar estabilidad de una feature entre lotes**

> La mediana de `gasto_promedio` debería ser estable entre meses.

```python
W, p = wilcoxon(gasto_enero - gasto_febrero.median(), alternative='two-sided')
# p > 0.05 → la feature es estable
```

---

### 4. Selección de modelos

**Ejemplo 11: Validar si un modelo es mejor que un baseline aleatorio**

> ¿Tu modelo predice mejor que simplemente predecir la mediana?

```python
errores_modelo = abs(y_real - y_predicho)
errores_baseline = abs(y_real - np.median(y_real))

W, p = wilcoxon(errores_modelo - errores_baseline, alternative='less')
# p < 0.05 → tu modelo es mejor que el baseline
```

**Ejemplo 12: Comparar errores contra un umbral aceptable**

> ¿El error MAE de tu modelo es menor a 10?

```python
W, p = wilcoxon(errores_mae - 10, alternative='less')
# p < 0.05 → los errores son significativamente menores a 10
```

**Ejemplo 13: Validar que las predicciones no tienen sesgo sistemático**

> ¿Los residuos de tu modelo son simétricos alrededor de cero?

```python
W, p = wilcoxon(residuos, alternative='two-sided')
# p > 0.05 → los residuos son simétricos (bueno)
# p < 0.05 → hay sesgo sistemático (problema)
```

---

### 5. Evaluación post-deploy

**Ejemplo 14: Verificar que una optimización mejoró la performance**

> Optimizaste una query de base de datos. ¿Los tiempos de respuesta mejoraron?

```python
W, p = wilcoxon(
    tiempos_después - tiempos_antes.median(),
    alternative='less'
)
# p < 0.05 → la optimización redujo los tiempos
```

**Ejemplo 15: Comparar métricas de un modelo antes y después de reentrenar**

> Reentrenaste el modelo con datos nuevos. ¿El error cambió?

```python
W, p = wilcoxon(
    errores_nuevos - errores_viejos.median(),
    alternative='two-sided'
)
# p < 0.05 → el reentrenamiento cambió la performance
```

**Ejemplo 16: Validar que un cambio de configuración no empeoró nada**

> Cambiaste el threshold de clasificación de 0.5 a 0.4. ¿El F1-score cambió?

```python
W, p = wilcoxon(
    f1_scores_threshold_04 - f1_scores_threshold_05.median(),
    alternative='two-sided'
)
# p > 0.05 → no hay diferencia significativa (el cambio es seguro)
```

---

### 6. Monitoreo y detección de anomalías

**Ejemplo 17: Detectar data drift en una feature**

> Comparás la distribución de `latencia` de hoy contra la mediana de la semana pasada.

```python
W, p = wilcoxon(
    latencia_hoy - latencia_semana_pasada.median(),
    alternative='two-sided'
)
# p < 0.05 → la latencia cambió (posible drift)
```

**Ejemplo 18: Validar estabilidad de un modelo en producción**

> ¿Las predicciones del modelo de hoy son consistentes con las del mes pasado?

```python
W, p = wilcoxon(
    predicciones_hoy - predicciones_mes_pasado.median(),
    alternative='two-sided'
)
# p > 0.05 → el modelo es estable
```

**Ejemplo 19: Detectar degradación de servicio**

> ¿El tiempo de respuesta del API aumentó respecto a la semana pasada?

```python
W, p = wilcoxon(
    tiempos_hoy - tiempos_semana_pasada.median(),
    alternative='greater'
)
# p < 0.05 → los tiempos empeoraron (degradación)
```

---

### 7. Validación de supuestos

**Ejemplo 20: Verificar normalidad de residuos**

> En un modelo lineal, ¿los residuos son simétricos alrededor de cero?

```python
W, p = wilcoxon(residuos, alternative='two-sided')
# p > 0.05 → los residuos son simétricos (bueno para el modelo)
# p < 0.05 → los residuos son asimétricos (problema)
```

**Ejemplo 21: Comparar distribución predicha vs. real**

> ¿La distribución de las predicciones es consistente con la distribución real?

```python
W, p = wilcoxon(
    y_predicho - y_real.median(),
    alternative='two-sided'
)
# p > 0.05 → no hay diferencia sistemática (bueno)
```

**Ejemplo 22: Validar que una muestra es representativa**

> La mediana de tu muestra debería ser similar a la mediana poblacional conocida.

```python
W, p = wilcoxon(muestra - mediana_poblacional, alternative='two-sided')
# p > 0.05 → la muestra es representativa
```

---

## Interpretación de resultados

### Valores que devuelve la prueba (Scipy)

| Salida | Qué significa |
|---|---|
| **W** | Estadística de Wilcoxon. Valores extremos indican diferencias grandes |
| **p** | Probabilidad de observar esta diferencia si H₀ fuera cierta |

### Valores que devuelve la prueba (Pingouin)

| Salida | Qué significa |
|---|---|
| **W-val** | Misma estadística W |
| **p-val** | p-valor |
| **RBC** | Tamaño del efecto (rank-biserial correlation). Rango: -1 a 1 |
| **CLES** | Probabilidad de que un valor al azar sea mayor que el valor de referencia |

### Reglas de decisión para p-valor

```
p < 0.01  → evidencia muy fuerte contra H₀
p < 0.05  → evidencia fuerte contra H₀ (estándar)
p < 0.10  → evidencia débil, posible tendencia
p ≥ 0.10  → sin evidencia suficiente
```

### Reglas de decisión para RBC (tamaño del efecto)

```
|RBC| ≥ 0.1  → efecto pequeño
|RBC| ≥ 0.3  → efecto mediano
|RBC| ≥ 0.5  → efecto grande
```

### Reglas de decisión para CLES

```
CLES = 0.50 → no hay diferencia (azar puro)
CLES = 0.55 → efecto pequeño
CLES = 0.60 → efecto mediano
CLES = 0.70 → efecto grande
```

**Siempre reportá el tamaño del efecto junto al p-valor.** Un p < 0.05 con RBC = 0.02 indica significancia estadística pero un efecto prácticamente irrelevante.

---

## Diseño experimental: potencia y tamaño de muestra

### Estimación previa (antes de recolectar datos)

Como Wilcoxon es no paramétrico, se usa una **aproximación basada en la distribución t**:

```python
import math
from statsmodels.stats.power import TTestPower

rbc = 0.4        # Tamaño del efecto esperado
potencia = 0.9   # Potencia deseada
alpha = 0.05     # Nivel de significancia

# 1. Convertir RBC a d de Cohen
d_cohen = (2 * rbc) / math.sqrt(1 - rbc**2)

# 2. Estimar n para t-test
analisis = TTestPower()
nttest = analisis.solve_power(
    effect_size=d_cohen,
    power=potencia,
    alpha=alpha,
    alternative='smaller'
)

# 3. Ajustar para Wilcoxon
n_wilcoxon = math.ceil(nttest[0] / 0.955)
print(f"Se necesitan al menos {n_wilcoxon} muestras")
```

### Potencia post-hoc (después de la prueba)

```python
from pingouin import power_ttest

n = len(datos)
rbc = -0.3138  # Obtenido de la prueba

d_obs = abs((2 * rbc) / math.sqrt(1 - rbc**2))

power = power_ttest(
    d=-d_obs,
    n=n / 0.955,
    alpha=0.05,
    contrast='one-sample',
    alternative='less'
)
print(f"Potencia: {power:.4f}")
# Si power > 0.8 → la muestra fue suficiente
# Si power < 0.8 → un resultado no significativo no es conclusivo
```

---

## Errores comunes

### 1. No pasar las diferencias

```python
# MAL: pasar los datos directamente
wilcoxon(datos, alternative='less')  # ❌ Esto compara contra 0

# BIEN: pasar la diferencia contra el valor de referencia
wilcoxon(datos - 200, alternative='less')  # ✅
```

### 2. Concluir sobre la media

```python
# MAL: "la media disminuyó de 200 a 196.5"
# BIEN: "la mediana pasó de 200ms a 196.5ms"
#        "la distribución de los tiempos se desplazó hacia abajo"
```

### 3. Ignorar el tamaño del efecto

```python
# Resultado: p = 0.001, RBC = 0.05
# MAL: "hay una diferencia altamente significativa"
# BIEN: "hay una diferencia estadísticamente significativa pero con efecto muy pequeño"
```

### 4. No verificar la asunción de simetría

Aunque Wilcoxon no asume normalidad, sí asume que las diferencias son **aproximadamente simétricas**.

```python
import seaborn as sns

diferencias = datos - referencia
sns.histplot(diferencias)
plt.axvline(0, color='red', linestyle='--')
plt.title("Distribución de diferencias (debe ser aproximadamente simétrica)")
plt.show()
```

### 5. Usar con muestras independientes

```python
# MAL: dos grupos de usuarios distintos
wilcoxon(grupo_A - grupo_B)  # ❌ No tiene sentido, no están emparejados

# BIEN: usar Mann-Whitney
from pingouin import mwu
mwu(x=grupo_A, y=grupo_B)  # ✅
```

### 6. Olvidar que la prueba es sobre diferencias

Wilcoxon no compara los datos contra el valor de referencia directamente. Calcula las **diferencias** y testea si son simétricas alrededor de cero.

```python
# Ejemplo conceptual
datos = [190, 210, 185, 205, 195]
referencia = 200

diferencias = datos - referencia  # [-10, 10, -15, 5, -5]
# Los rangos se calculan sobre el valor ABSOLUTO de estas diferencias
# Los signos indican si cada dato está por encima o por debajo de la referencia
```

---

## Código rápido de referencia

### Con Scipy

```python
from scipy.stats import wilcoxon

valor_referencia = 200

# Unilateral: ¿la mediana es MENOR que 200?
W, p = wilcoxon(datos - valor_referencia, alternative='less')

# Unilateral: ¿la mediana es MAYOR que 200?
W, p = wilcoxon(datos - valor_referencia, alternative='greater')

# Bilateral: ¿la mediana es DIFERENTE de 200?
W, p = wilcoxon(datos - valor_referencia, alternative='two-sided')

print(f"W: {W}, p: {p}")
```

### Con Pingouin (más información)

```python
import pingouin as pg

diferencias = datos - valor_referencia
resultados = pg.wilcoxon(diferencias, alternative='less')

print(f"W: {resultados['W-val'].values[0]}")
print(f"p: {resultados['p-val'].values[0]}")
print(f"RBC: {resultados['RBC'].values[0]}")
print(f"CLES: {resultados['CLES'].values[0]}")

# Decisión
alpha = 0.05
if resultados['p-val'].values[0] < alpha:
    print("Rechazar H₀ → hay diferencia significativa")
else:
    print("No rechazar H₀ → no hay diferencia significativa")
```

### Potencia post-hoc

```python
import math
from pingouin import power_ttest

n = len(datos)
rbc = resultados['RBC'].values[0]
alpha = 0.05

d_obs = abs((2 * rbc) / math.sqrt(1 - rbc**2))

power = power_ttest(
    d=-d_obs,
    n=n / 0.955,
    alpha=alpha,
    contrast='one-sample',
    alternative='less'
)
print(f"Potencia: {power:.4f}")
```
