# Guía de Referencia: Prueba de Wilcoxon para Muestras Pareadas

## ¿Qué es?

La prueba de Wilcoxon para muestras pareadas (también llamada *Wilcoxon signed-rank test*) es una prueba **no paramétrica** que compara las **medianas de las diferencias** entre dos mediciones **emparejadas** para determinar si la mediana de las diferencias es distinta de cero.

**Pregunta central:** ¿La mediana de las diferencias entre las dos mediciones es significativamente distinta de cero?

**Qué NO hace:**
- No compara medias (usa rangos de las diferencias, no valores absolutos)
- No asume normalidad de los datos originales (pero sí de las diferencias)
- No requiere varianzas iguales

---

## Cuándo usarla

### Regla práctica

| Situación | Herramienta |
|---|---|
| Dos grupos **independientes**, datos **normales** | t-test de muestras independientes |
| Dos grupos **independientes**, datos **NO normales** | Mann-Whitney U |
| Dos mediciones **emparejadas** (mismo individuo), datos **normales** | t-test pareado |
| Dos mediciones **emparejadas** (mismo individuo), datos **NO normales** | **Wilcoxon pareado** |
| Un grupo contra un **valor fijo** | Wilcoxon una muestra |

### Condiciones ideales

1. **Muestras emparejadas** — cada observación en un grupo tiene una correspondencia natural en el otro (mismo individuo, mismo objeto, par emparejado)
2. **Datos ordinales o continuos** — la prueba trabaja con rangos de las diferencias
3. **Diferencias simétricas** — las diferencias entre pares deben tener una distribución simétrica (aunque no necesariamente normal)

---

## Ejemplos por fase del proyecto de datos

### 1. Exploración de datos (EDA)

**Ejemplo 1: Comparar tiempos de respuesta antes y después de una optimización**

> Tenés datos de tiempos de respuesta de una API antes y después de una mejora. ¿Los tiempos disminuyeron?

```python
import pingouin as pg

# Calcular diferencias
diferencias = datos['desp'] - datos['antes']

# Prueba unilateral: ¿las diferencias son menores que cero?
resultado = pg.wilcoxon(diferencias, alternative='less')
# p < 0.05 → los tiempos disminuyeron significativamente
# p > 0.05 → no hay evidencia de mejora
```

**Ejemplo 2: Evaluar el efecto de un tratamiento en pacientes**

> Mediciste la presión arterial de 30 pacientes antes y después de un fármaco. ¿El fármaco reduce la presión?

```python
resultado = pg.wilcoxon(
    datos['presion_despues'] - datos['presion_antes'],
    alternative='less'
)
# p < 0.05 → el fármaco reduce significativamente la presión
# RBC negativo → confirma la dirección del efecto
```

**Ejemplo 3: Comparar rendimiento antes y después de un curso**

> Querés saber si un curso de estadística mejoró el rendimiento de los estudiantes.

```python
resultado = pg.wilcoxon(
    datos['nota_despues'] - datos['nota_antes'],
    alternative='greater'
)
# p < 0.05 → el curso mejoró significativamente el rendimiento
```

**Ejemplo 4: Detectar si una variable tiene sesgo sistemático**

> Tenés mediciones de temperatura de 50 sensores tomadas a las 8am y a las 2pm. ¿Hay una diferencia consistente?

```python
resultado = pg.wilcoxon(
    datos['temp_14h'] - datos['temp_08h'],
    alternative='two-sided'
)
# p < 0.05 → hay una diferencia significativa entre horarios
```

---

### 2. Feature Engineering

**Ejemplo 5: Validar si una normalización preservó la relación entre pares**

> Aplicaste una transformación logarítmica a tus datos. ¿La relación antes/después se mantuvo?

```python
import numpy as np

# Antes de transformar
diferencias_original = datos['desp'] - datos['antes']

# Después de transformar
datos_log = np.log1p(datos[['antes', 'desp']])
diferencias_log = datos_log['desp'] - datos_log['antes']

# Comparar si las diferencias cambiaron
resultado = pg.wilcoxon(diferencias_log - diferencias_original, alternative='two-sided')
# p > 0.05 → la transformación no alteró la relación entre pares
```

**Ejemplo 6: Evaluar si un Feature Engineering creó separación**

> Creaste una feature que es la diferencia entre dos variables. ¿Esa diferencia es significativa?

```python
# Crear feature de diferencia
datos['diferencia_tiempo'] = datos['tiempo_after'] - datos['tiempo_before']

# Evaluar si la diferencia es significativa
resultado = pg.wilcoxon(datos['diferencia_tiempo'], alternative='two-sided')
# p < 0.05 → la feature tiene poder discriminativo
```

**Ejemplo 7: Comparar efecto de dos transformaciones**

> Aplicaste StandardScaler y MinMaxScaler a tus datos. ¿Los rangos resultantes son similares?

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler_std = StandardScaler()
scaler_minmax = MinMaxScaler()

datos_std = scaler_std.fit_transform(datos[['antes', 'desp']])
datos_minmax = scaler_minmax.fit_transform(datos[['antes', 'desp']])

diferencias_std = datos_std[:, 1] - datos_std[:, 0]
diferencias_minmax = datos_minmax[:, 1] - datos_minmax[:, 0]

resultado = pg.wilcoxon(diferencias_std - diferencias_minmax, alternative='two-sided')
# p > 0.05 → ambas transformaciones preservan la relación entre pares
```

---

### 3. Selección de modelos

**Ejemplo 8: Comparar errores de dos modelos en los mismos datos**

> Tenés dos modelos de regresión evaluados en el mismo conjunto de test. ¿Los errores del modelo A son sistematicamente menores?

```python
# Errores absolutos de cada modelo en los mismos puntos de test
errores_A = np.abs(y_real - y_predicho_A)
errores_B = np.abs(y_real - y_predicho_B)

resultado = pg.wilcoxon(errores_A - errores_B, alternative='less')
# p < 0.05 → modelo A tiene errores significativamente menores
```

**Ejemplo 9: Evaluar estabilidad de un modelo entre folds**

> Validaste tu modelo con cross-validation. ¿El rendimiento es estable entre folds?

```python
# Rendimiento (por ejemplo, AUC) en cada fold
rendimiento_fold1 = [0.85, 0.82, 0.88, 0.84, 0.86]
rendimiento_fold2 = [0.83, 0.81, 0.87, 0.83, 0.85]

resultado = pg.wilcoxon(
    np.array(rendimiento_fold1) - np.array(rendimiento_fold2),
    alternative='two-sided'
)
# p > 0.05 → el modelo es estable entre folds
```

**Ejemplo 10: Comparar rendimiento antes y después de hyperparameter tuning**

> Optimizaste los hiperparámetros de tu modelo. ¿El rendimiento mejoró?

```python
# Accuracy antes y después de tuning en cada fold de CV
accuracy_antes = [0.78, 0.80, 0.79, 0.81, 0.77]
accuracy_despues = [0.82, 0.84, 0.83, 0.85, 0.81]

resultado = pg.wilcoxon(
    np.array(accuracy_despues) - np.array(accuracy_antes),
    alternative='greater'
)
# p < 0.05 → el tuning mejoró significativamente el rendimiento
```

---

### 4. Evaluación post-deploy

**Ejemplo 11: Validar que una optimización de base de datos funcionó**

> Desplegaste una optimización de consultas. ¿El tiempo de respuesta disminuyó?

```python
# Tiempos de las mismas consultas antes y después
resultado = pg.wilcoxon(
    datos['tiempo_after'] - datos['tiempo_before'],
    alternative='less'
)
# p < 0.05 → la optimización redujo los tiempos significativamente
```

**Ejemplo 12: Evaluar mejoras en un modelo en producción**

> Actualizaste el modelo en producción. ¿El error MAE disminuyó?

```python
# Errores en los mismos puntos de datos
errores_v1 = np.abs(y_real - y_pred_v1)
errores_v2 = np.abs(y_real - y_pred_v2)

resultado = pg.wilcoxon(errores_v2 - errores_v1, alternative='less')
# p < 0.05 → la nueva versión tiene errores menores
```

**Ejemplo 13: Verificar que un cambio de UI mejoró la experiencia**

> Rediseñaste el flujo de checkout. ¿El tiempo que tarda el usuario en completar la compra disminuyó?

```python
# Mismos usuarios, tiempo antes y después del rediseño
resultado = pg.wilcoxon(
    datos['tiempo_checkout_nuevo'] - datos['tiempo_checkout_viejo'],
    alternative='less'
)
# p < 0.05 → el nuevo diseño es más rápido
```

---

### 5. Monitoreo y data drift

**Ejemplo 14: Detectar drift en métricas de usuarios**

> Comparaste el comportamiento de los mismos usuarios en dos períodos. ¿Cambiaron sus patrones?

```python
# Mismos usuarios, métricas en dos períodos
resultado = pg.wilcoxon(
    datos['metrica_periodo2'] - datos['metrica_periodo1'],
    alternative='two-sided'
)
# p < 0.05 → hay drift en el comportamiento de los usuarios
```

**Ejemplo 15: Evaluar estabilidad de sensores**

> Tenés lecturas de los mismos sensores en dos momentos distintos. ¿Los sensores son estables?

```python
resultado = pg.wilcoxon(
    datos['lectura_tiempo2'] - datos['lectura_tiempo1'],
    alternative='two-sided'
)
# p > 0.05 → los sensores son estables
# p < 0.05 → posible drift o degradación del sensor
```

**Ejemplo 16: Comparar rendimiento de un modelo en dos ventanas temporales**

> Evaluaste el mismo modelo en enero y en marzo. ¿El rendimiento se mantuvo?

```python
# Métricas del modelo en los mismos usuarios
metricas_enero = datos['prediccion_enero']
metricas_marzo = datos['prediccion_marzo']

resultado = pg.wilcoxon(
    metricas_marzo - metricas_enero,
    alternative='two-sided'
)
# p > 0.05 → el modelo es estable en el tiempo
```

---

### 6. A/B Testing (con emparejamiento)

**Ejemplo 17: Test A/B con usuarios que probaron ambas versiones**

> Tus usuarios probaron la versión A y luego la versión B. ¿Prefieren la B?

```python
# Mismo usuario, puntuación con A y con B
resultado = pg.wilcoxon(
    datos['puntuacion_B'] - datos['puntuacion_A'],
    alternative='greater'
)
# p < 0.05 → la versión B genera mayor satisfacción
```

**Ejemplo 18: Evaluar dos algoritmos de recomendación en los mismos usuarios**

> Cada usuario recibió recomendaciones del algoritmo viejo y del nuevo. ¿El nuevo es mejor?

```python
# engagement del mismo usuario con cada algoritmo
resultado = pg.wilcoxon(
    datos['clicks_nuevo'] - datos['clicks_viejo'],
    alternative='greater'
)
# p < 0.05 → el algoritmo nuevo genera más clicks
```

**Ejemplo 19: Comparar dos estrategias de pricing en el mismo mercado**

> Testeaste precio fijo vs precio dinámico en los mismos productos. ¿Cuál genera más ingresos?

```python
resultado = pg.wilcoxon(
    datos['ingresos_dinamico'] - datos['ingresos_fijo'],
    alternative='greater'
)
# p < 0.05 → el precio dinámico genera más ingresos
```

---

### 7. Validación de supuestos en modelos lineales

**Ejemplo 20: Evaluar si los residuos son simétricos**

> En un modelo de regresión, ¿los residuos tienen distribución simétrica?

```python
resultado = pg.wilcoxon(residuos, alternative='two-sided')
# p > 0.05 → los residuos son simétricos (bueno para modelos lineales)
# p < 0.05 → los residuos son asimétricos (posible problema)
```

**Ejemplo 21: Comparar residuos en subgrupos**

> ¿El modelo se comporta igual para usuarios nuevos y veteranos?

```python
# Mismos usuarios, residuos en dos condiciones
resultado = pg.wilcoxon(
    datos['residuo_nuevo'] - datos['residuo_veterano'],
    alternative='two-sided'
)
# p > 0.05 → el modelo se comporta similar para ambos grupos
# p < 0.05 → hay sesgo en el modelo según antigüedad del usuario
```

---

## Interpretación de resultados

### Valores que devuelve la prueba

| Salida | Qué significa |
|---|---|
| **W-val** | Estadística de Wilcoxon. Valores extremos (muy bajos o muy altos) indican diferencias |
| **p-val** | Probabilidad de observar esta diferencia si H₀ fuera cierta |
| **RBC** | Tamaño del efecto (*rank-biserial correlation*). Rango: -1 a 1 |
| **CLES** | Probabilidad de que una diferencia al azar sea mayor que otra |

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

**Importante:** un p < 0.05 con RBC = 0.05 significa que hay una diferencia estadísticamente significativa pero **prácticamente irrelevante**. Siempre reportá el tamaño del efecto junto al p-valor.

### Reglas de decisión para CLES

```
CLES = 0.50 → no hay diferencia (azaro puro)
CLES = 0.55 → efecto pequeño
CLES = 0.60 → efecto mediano
CLES = 0.70 → efecto grande
```

### Cómo interpretar el signo de RBC

```
RBC negativo → la segunda medición tiende a ser MENOR que la primera (mejora si esperabas reducción)
RBC positivo → la segunda medición tiende a ser MAYOR que la primera (mejora si esperabas incremento)
```

---

## Errores comunes

### 1. Usar Wilcoxon pareado cuando los datos son independientes

```python
# MAL: grupos diferentes (independientes)
wilcoxon(datos_grupo_A, datos_grupo_B)  # ❌

# BIEN: usar Mann-Whitney U
from pingouin import mwu
mwu(x=datos_grupo_A, y=datos_grupo_B)  # ✅
```

### 2. No calcular las diferencias correctamente

```python
# MAL: pasar las dos columnas directamente
pg.wilcoxon(datos['antes'], datos['despues'])  # ❌ (así lo hace scipy, no pingouin)

# BIEN: calcular la diferencia primero
diferencias = datos['despues'] - datos['antes']
pg.wilcoxon(diferencias, alternative='less')  # ✅
```

### 3. Olvidar que la prueba es sobre las diferencias

```python
# MAL: verificar normalidad de cada columna por separado
shapiro(datos['antes'])
shapiro(datos['despues'])

# BIEN: verificar normalidad de las diferencias
diferencias = datos['despues'] - datos['antes']
shapiro(diferencias)
# Si p > 0.05 → las diferencias son normales → podés usar t-test pareado
# Si p < 0.05 → las diferencias NO son normales → usás Wilcoxon
```

### 4. Concluir sobre la media en vez de la mediana

```python
# MAL: "la media de las diferencias es menor que cero"
# BIEN: "la mediana de las diferencias es menor que cero"
```

Wilcoxon no testea medias. Testea si la mediana de las diferencias es distinta de cero.

### 5. Ignorar el tamaño del efecto

```python
# Resultado: p = 0.001, RBC = 0.03
# MAL: "hay una diferencia altamente significativa"
# BIEN: "hay una diferencia estadísticamente significativa pero con un efecto muy pequeño (RBC = 0.03)"
```

### 6. Usar p como medida de importancia

```python
# MAL: "p = 0.001, esto es súper importante"
# BIEN: "p = 0.001 indica que es muy improbable que esta diferencia sea por azar"
# La importancia práctica depende del contexto, no del p-valor
```

### 7. No reportar la dirección del efecto

```python
# MAL: "hay una diferencia significativa (p = 0.03)"
# BIEN: "los tiempos después de la optimización son significativamente menores (p = 0.03, RBC = -0.45)"
```

El signo de RBC te dice la dirección. Siempre reportalo.

---

## Resumen: ¿cuándo usar cada prueba?

```
¿Tenés dos mediciones?
│
├── ¿Son del mismo individuo/objeto (emparejadas)?
│   ├── ¿Las diferencias son normales? → t-test pareado
│   └── ¿Las diferencias NO son normales? → Wilcoxon pareado
│
├── ¿Son de grupos independientes?
│   ├── ¿Son normales? → t-test independiente
│   └── ¿NO son normales? → Mann-Whitney U
│
└── ¿Es un grupo contra un valor fijo?
    ├── ¿Es normal? → t-test una muestra
    └── ¿NO es normal? → Wilcoxon una muestra
```

---

## Código rápido de referencia

```python
import pingouin as pg
import numpy as np

# Calcular diferencias (segundo menos primero)
diferencias = datos['medicion_2'] - datos['medicion_1']

# Prueba bilateral (¿hay diferencia?)
resultado = pg.wilcoxon(diferencias, alternative='two-sided')

# Prueba unilateral (¿disminuyó?)
resultado = pg.wilcoxon(diferencias, alternative='less')

# Prueba unilateral (¿aumentó?)
resultado = pg.wilcoxon(diferencias, alternative='greater')

# Ver resultado
print(f"W: {resultado['W-val'].values[0]}")
print(f"p: {resultado['p-val'].values[0]}")
print(f"RBC: {resultado['RBC'].values[0]}")
print(f"CLES: {resultado['CLES'].values[0]}")

# Decisión
alpha = 0.05
if resultado['p-val'].values[0] < alpha:
    print("Rechazar H₀ → hay diferencia significativa")
else:
    print("No rechazar H₀ → no hay diferencia significativa")

# Tamaño del efecto
rbc = resultado['RBC'].values[0]
if abs(rbc) >= 0.5:
    print("Efecto grande")
elif abs(rbc) >= 0.3:
    print("Efecto mediano")
elif abs(rbc) >= 0.1:
    print("Efecto pequeño")
else:
    print("Efecto despreciable")
```

---

## Fórmulas de referencia

### Conversión RBC a d de Cohen

$$d = \frac{2 \times RBC}{\sqrt{1 - RBC^2}}$$

### Cálculo de tamaño de muestra

1. Definir el tamaño del efecto deseado (RBC)
2. Convertir a d de Cohen: $d = \frac{2 \times RBC}{\sqrt{1 - RBC^2}}$
3. Usar `TTestPower` para estimar $n_{ttest}$
4. Corregir: $n_{Wilcoxon} = \frac{n_{ttest}}{0.955}$

### Potencia post-hoc

```python
import math
from statsmodels.stats.power import TTestPower

n = len(datos)
rbc = resultado['RBC'].values[0]

# Convertir RBC al d de Cohen
d_obs = 2 * rbc / math.sqrt(1 - rbc**2)

# Usar power_ttest pero multiplicar n por 0.955
power = pg.power_ttest(
    d=d_obs,
    n=n * 0.955,
    alpha=0.05,
    contrast='paired',
    alternative='less'  # según tu hipótesis
)

print(f"Potencia Post-hoc: {power:.4f}")
```
