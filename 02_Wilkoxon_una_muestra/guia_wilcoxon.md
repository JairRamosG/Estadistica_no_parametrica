# Guía de Referencia: Prueba de Wilcoxon de Rangos con Signo

## ¿Qué es?

La prueba de Wilcoxon de rangos con signo es una prueba **no paramétrica** que compara las **medianas** para determinar si hay una diferencia significativa. Tiene dos variantes:

1. **Una muestra** — compara una muestra contra un valor de referencia conocido
2. **Pareada** — compara dos muestras que están emparejadas (el mismo individuo medido dos veces)

**Pregunta central:**
- Una muestra: ¿la mediana de mis datos es igual a un valor conocido?
- Pareada: ¿la mediana de las diferencias entre ambos grupos es cero?

**Qué NO hace:**
- No compara medias (usa rangos de las diferencias)
- No asume normalidad
- No funciona con muestras independientes (para eso está Mann-Whitney)

---

## Cuándo usarla

### Regla práctica

| Situación | Herramienta |
|---|---|
| Un grupo contra un valor fijo, datos normales | t-test una muestra |
| Un grupo contra un valor fijo, datos NO normales | **Wilcoxon una muestra** |
| Dos grupos emparejados, datos normales | t-test pareado |
| Dos grupos emparejados, datos NO normales | **Wilcoxon pareado** |
| Dos grupos independientes, datos NO normales | Mann-Whitney U |

### Condiciones ideales

1. **Datos ordinales o continuos** — la prueba trabaja con rangos
2. **Diferencias simétricas** — las diferencias entre pares (o entre datos y referencia) deben ser aproximadamente simétricas
3. **Independencia dentro de cada grupo** — los datos dentro de cada muestra son independientes entre sí (aunque los dos grupos están emparejados)

### Diferencia clave con Mann-Whitney

| | Wilcoxon | Mann-Whitney |
|---|---|---|
| Tipo de datos | **Emparejados** o un grupo vs. valor fijo | **Independientes** |
| Qué compara | Diferencias dentro de pares | Rangos entre grupos distintos |
| Ejemplo | Antes/después en el mismo usuario | Usuario A vs. usuario B |

---

## Ejemplos por fase del proyecto de datos

### 1. Exploración de datos (EDA)

**Ejemplo 1: Comparar una métrica contra un benchmark**

> Tu modelo actual tiene una latencia mediana de 200ms. ¿El nuevo modelo es más rápido?

```python
from scipy.stats import wilcoxon

# Datos de latencia del nuevo modelo
latencia_nueva = [185, 192, 178, 200, 195, ...]

# Valor de referencia: 200ms
W, p = wilcoxon(latencia_nueva - 200, alternative='less')

# p < 0.05 → el nuevo modelo es estadísticamente más rápido
```

**Ejemplo 2: Verificar si una columna tiene distribución simétrica**

> Querés saber si la columna `tiempo_respuesta` es simétrica alrededor de su mediana. Si lo es, las diferencias contra la mediana deberían ser balanceadas.

```python
mediana = df['tiempo_respuesta'].median()
W, p = wilcoxon(df['tiempo_respuesta'] - mediana, alternative='two-sided')

# p > 0.05 → no se rechaza simetría (buena señal)
# p < 0.05 → la distribución es asimétrica
```

**Ejemplo 3: Comparar dos mediciones del mismo sensor**

> Tenés un sensor de temperatura medido cada hora. ¿La lectura de la mañana difiere de la de la tarde en los mismos días?

```python
W, p = wilcoxon(
    lecturas_manana - lecturas_tarde,
    alternative='two-sided'
)
# p < 0.05 → hay diferencia sistemática entre mañana y tarde
```

**Ejemplo 4: Validar si una métrica está dentro de un rango aceptable**

> Tu SLA dice que el tiempo de respuesta debe ser ≤ 150ms. ¿Los datos cumplen?

```python
W, p = wilcoxon(tiempos_respuesta - 150, alternative='greater')
# Si p < 0.05 → los tiempos son significativamente mayores a 150ms (no cumple SLA)
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
# Comparar la distribución original (sin nulos) contra la imputada
W, p = wilcoxon(
    datos_originales['edad'].dropna() - datos_imputados['edad'].median(),
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
W, p = wilcoxon(
    tiempos_sesion_compradores - 120,
    alternative='greater'
)
# p < 0.05 → los compradores pasan más de 120 segundos
```

**Ejemplo 9: Comparar dos features candidatas**

> Tenés `ingreso` y `gasto_mensual`. ¿Sus medianas son significativamente distintas?

```python
# Esto solo funciona si ambas variables están en la misma escala
W, p = wilcoxon(df['ingreso'] - df['gasto_mensual'], alternative='two-sided')
```

**Ejemplo 10: Evaluar estabilidad de una feature entre folds de cross-validation**

> En cada fold, la mediana de `antigüedad` debería ser similar. ¿Lo es?

```python
# Supongamos que tenés las medianas de cada fold
fold_medians = [mediana_fold_1, mediana_fold_2, mediana_fold_3, ...]
W, p = wilcoxon(fold_medians - np.median(fold_medians), alternative='two-sided')
# p > 0.05 → la feature es estable entre folds
```

---

### 4. Selección de modelos

**Ejemplo 11: Comparar errores de dos modelos en el mismo test set**

> ¿El modelo A tiene errores menores que el modelo B en las mismas observaciones?

```python
# Errores absolutos de ambos modelos en el mismo test set
W, p = wilcoxon(
    abs(errores_modelo_A) - abs(errores_modelo_B),
    alternative='less'
)
# p < 0.05 → modelo A tiene errores significativamente menores
```

**Ejemplo 12: Validar si un modelo es mejor que un baseline aleatorio**

> ¿Tu modelo predice mejor que simplemente predecir la mediana?

```python
errores_modelo = abs(y_real - y_predicho)
errores_baseline = abs(y_real - np.median(y_real))

W, p = wilcoxon(errores_modelo - errores_baseline, alternative='less')
# p < 0.05 → tu modelo es mejor que el baseline
```

**Ejemplo 13: Comparar AUC de dos modelos en cross-validation**

> Tenés los AUC de 10 folds para dos modelos. ¿Son significativamente distintos?

```python
W, p = wilcoxon(aucs_modelo_A - aucs_modelo_B, alternative='two-sided')
# p < 0.05 → hay diferencia significativa entre modelos
```

---

### 5. Evaluación post-deploy

**Ejemplo 14: Verificar que una optimización mejoró la performance**

> Optimizaste una query de base de datos. ¿Los tiempos de respuesta mejoraron?

```python
# Medir la misma query 50 veces antes y después
W, p = wilcoxon(
    tiempos_antes - tiempos_después,
    alternative='greater'
)
# p < 0.05 → la optimización redujo los tiempos
```

**Ejemplo 15: Comparar métricas de un modelo antes y después de reentrenar**

> Reentrenaste el modelo con datos nuevos. ¿El error cambió?

```python
W, p = wilcoxon(
    errores_viejo - errores_nuevo,
    alternative='two-sided'
)
# p < 0.05 → el reentrenamiento cambió la performance
# Si p < 0.05 Y errores_nuevo < errores_viejo → mejoró
```

**Ejemplo 16: Validar que un cambio de configuración no empeoró nada**

> Cambiaste el threshold de clasificación de 0.5 a 0.4. ¿El F1-score cambió?

```python
W, p = wilcoxon(
    f1_scores_threshold_05 - f1_scores_threshold_04,
    alternative='two-sided'
)
# p > 0.05 → no hay diferencia significativa (el cambio es seguro)
```

---

### 6. A/B Testing con datos pareados

**Ejemplo 17: Comparar dos versiones de un modelo en los mismos usuarios**

> ¿La versión nueva del modelo genera predicciones más precisas que la vieja en los mismos usuarios?

```python
W, p = wilcoxon(
    errores.modelo_vieja - errores.modelo_nueva,
    alternative='greater'
)
# p < 0.05 → la versión nueva es más precisa
```

**Ejemplo 18: Evaluar dos estrategias de email marketing**

> Enviaste email A la semana pasada y email B esta semana a los mismos usuarios. ¿La tasa de apertura cambió?

```python
# Nota: esto solo funciona si cada usuario recibió ambos emails
W, p = wilcoxon(
    aperturas_email_A - aperturas_email_B,
    alternative='two-sided'
)
```

**Ejemplo 19: Comparar dos interfaces en el mismo dispositivo**

> Probaste dos diseños de UI en los mismos 30 dispositivos. ¿El tiempo de carga difiere?

```python
W, p = wilcoxon(
    tiempos_ui_A - tiempos_ui_B,
    alternative='two-sided'
)
```

---

### 7. Monitoreo y detección de anomalías

**Ejemplo 20: Detectar data drift en una feature emparejada**

> Comparás la distribución de `latencia` entre la misma hora de la semana pasada y esta semana.

```python
# Latencia a las 10am de cada lunes
latencia_semana_pasada = [...]
latencia_esta_semana = [...]

W, p = wilcoxon(
    latencia_semana_pasada - latencia_esta_semana,
    alternative='two-sided'
)
# p < 0.05 → la latencia cambió (posible drift)
```

**Ejemplo 21: Validar estabilidad de un modelo entre lotes**

> ¿Las predicciones del modelo en el lote de enero son consistentes con las de febrero (mismo conjunto de clientes)?

```python
W, p = wilcoxon(
    predicciones_enero - predicciones_febrero,
    alternative='two-sided'
)
# p > 0.05 → el modelo es estable
```

**Ejemplo 22: Detectar degradación de servicio**

> ¿El tiempo de respuesta del API aumentó respecto a la semana pasada?

```python
# Tiempos de respuesta de los mismos endpoints
W, p = wilcoxon(
    tiempos_semana_pasada - tiempos_esta_semana,
    alternative='greater'
)
# p < 0.05 → los tiempos empeoraron (degradación)
```

---

### 8. Validación de supuestos

**Ejemplo 23: Verificar normalidad de residuos**

> En un modelo lineal, ¿los residuos son simétricos alrededor de cero?

```python
W, p = wilcoxon(residuos, alternative='two-sided')
# p > 0.05 → los residuos son simétricos (bueno para el modelo)
# p < 0.05 → los residuos son asimétricos (problema)
```

**Ejemplo 24: Comparar distribución predicha vs. real**

> ¿La distribución de las predicciones es consistente con la distribución real?

```python
W, p = wilcoxon(
    y_predicho - y_real,
    alternative='two-sided'
)
# p > 0.05 → no hay diferencia sistemática (bueno)
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
| **CLES** | Probabilidad de que un valor al azar de un grupo sea mayor que uno del otro |

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
    alternative='smaller'  # unilateral: menor que
)

# 3. Ajustar para Wilcoxon
n_wilcoxon = math.ceil(nttest[0] / 0.955)
print(f"Se necesitan al menos {n_wilcoxon} muestras")
```

### Potencia post-hoc (después de la prueba)

```python
from pingouin import power_ttest

n = 85  # Tamaño de muestra real
rbc = -0.3138  # Obtenido de la prueba

d_obs = abs((2 * rbc) / math.sqrt(1 - rbc**2))

power = power_ttest(
    d=-d_obs,
    n=n / 0.955,
    alpha=0.05,
    contrast='one-sample',
    alternative='less'
)
print(f"Potencia post-hoc: {power:.4f}")
# Si power > 0.8 → la muestra fue suficiente
# Si power < 0.8 → un resultado no significativo no es conclusivo
```

---

## Errores comunes

### 1. Usar Wilcoxon cuando los datos son independientes

```python
# MAL: dos grupos de usuarios distintos
wilcoxon(grupo_A, grupo_B)  # ❌

# BIEN: usar Mann-Whitney
from pingouin import mwu
mwu(x=grupo_A, y=grupo_B)  # ✅
```

### 2. Pasar los datos crudos en lugar de las diferencias

```python
# MAL: pasar los datos directamente contra el valor de referencia
wilcoxon(datos, alternative='less')  # ❌ Esto compara contra 0

# BIEN: pasar la diferencia contra el valor de referencia
wilcoxon(datos - 200, alternative='less')  # ✅
```

### 3. Concluir sobre la media

```python
# MAL: "la media disminuyó de 200 a 196.5"
# BIEN: "la distribución de los tiempos se desplazó hacia abajo"
#        "la mediana pasó de 200ms a 196.5ms"
```

### 4. Ignorar el tamaño del efecto

```python
# Resultado: p = 0.001, RBC = 0.05
# MAL: "hay una diferencia altamente significativa"
# BIEN: "hay una diferencia estadísticamente significativa pero con efecto muy pequeño"
```

### 5. Olvidar que la prueba es sobre diferencias

Wilcoxon no compara las muestras directamente. Calcula las **diferencias** entre cada par (o contra el valor de referencia) y testea si esas diferencias son simétricas alrededor de cero.

```python
# Ejemplo conceptual
datos = [190, 210, 185, 205, 195]
referencia = 200

diferencias = datos - referencia  # [-10, 10, -15, 5, -5]
# Wilcoxon testea si estas diferencias son simétricas alrededor de 0
```

### 6. No verificar la asunción de simetría

Aunque Wilcoxon no asume normalidad, sí asume que las diferencias son **aproximadamente simétricas**. Si las diferencias son muy asimétricas, el resultado puede ser engañoso.

```python
# Verificar simetría visualmente
import seaborn as sns
sns.histplot(diferencias)
plt.axvline(0, color='red', linestyle='--')
plt.title("Distribución de diferencias (debe ser aproximadamente simétrica)")
plt.show()
```

---

## Resumen: ¿cuándo usar cada prueba?

```
¿Tenés dos grupos de datos?
│
├── ¿Son emparejados (mismo individuo medido dos veces)?
│   ├── ¿Son normales? → t-test pareado
│   └── ¿NO son normales? → Wilcoxon pareado
│
├── ¿Son independientes (distintos individuos)?
│   ├── ¿Son normales? → t-test independiente
│   └── ¿NO son normales? → Mann-Whitney U
│
└── ¿Es un grupo contra un valor fijo?
    ├── ¿Es normal? → t-test una muestra
    └── ¿NO es normal? → Wilcoxon una muestra
```

---

## Código rápido de referencia

### Wilcoxon una muestra (vs. valor de referencia)

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

### Wilcoxon pareado (dos muestras emparejadas)

```python
from scipy.stats import wilcoxon

# Unilateral: ¿antes > después? (¿mejoró?)
W, p = wilcoxon(antes - después, alternative='greater')

# Unilateral: ¿antes < después? (¿empeoró?)
W, p = wilcoxon(antes - después, alternative='less')

# Bilateral: ¿hay diferencia?
W, p = wilcoxon(antes - después, alternative='two-sided')

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
