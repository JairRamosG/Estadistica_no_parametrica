# Guía de Referencia: Prueba U de Mann-Whitney

## ¿Qué es?

La prueba U de Mann-Whitney (también llamada Wilcoxon rank-sum test) es una prueba **no paramétrica** que compara las **medianas** de dos muestras **independientes** para determinar si provienen de la misma distribución.

**Pregunta central:** ¿Un valor al azar del grupo A tiende a ser mayor (o menor) que un valor al azar del grupo B?

**Qué NO hace:**
- No compara medias (usa rangos, no valores absolutos)
- No asume normalidad
- No requiere varianzas iguales

---

## Cuándo usarla

### Regla práctica

| Situación | Herramienta |
|---|---|
| Dos grupos **independientes**, datos **normales** | t-test de muestras independientes |
| Dos grupos **independientes**, datos **NO normales** | **Mann-Whitney U** |
| Dos grupos **emparejados** (mismo individuo) | Wilcoxon pareado |
| Un grupo contra un **valor fijo** | Wilcoxon una muestra |

### Condiciones ideales

1. **Muestras independientes** — ningún dato de un grupo está relacionado con un dato del otro
2. **Datos ordinales o continuos** — la prueba trabaja con rangos, así que funciona con escalas ordinales también
3. **Distribuciones con forma similar** — las distribuciones deben tener forma parecida (aunque no sean normales). Si las formas son muy distintas, Mann-Whitney compara distribuciones completas, no solo medianas

---

## Ejemplos por fase del proyecto de datos

### 1. Exploración de datos (EDA)

**Ejemplo 1: Comparar una métrica entre ciudades**

> Tenés datos de ventas de una tienda online. ¿El gasto por usuario difiere entre Buenos Aires y Córdoba?

```python
from pingouin import mwu

mwu(
    x=df[df['ciudad'] == 'Buenos Aires']['gasto'],
    y=df[df['ciudad'] == 'Córdoba']['gasto'],
    alternative='two-sided'
)
# p < 0.05 → las medianas de gasto son distintas
# p > 0.05 → no hay diferencia significativa
```

**Ejemplo 2: Detectar si una variable separa clases**

> Querés saber si `tiempo_en_sesion` es un buen predictor de compra. ¿Los usuarios que compran passan más tiempo que los que no?

```python
mwu(
    x=df[df['compró'] == 0]['tiempo_en_sesion'],
    y=df[df['compró'] == 1]['tiempo_en_sesion'],
    alternative='less'
)
# p < 0.05 → sí, los que compran tienden a pasar más tiempo
# Esta variable es candidata a feature en tu modelo
```

**Ejemplo 3: Comparar distribuciones entre géneros**

> ¿La edad de los usuarios que abandonan el carrito es distinta según género?

```python
mwu(
    x=df[df['género'] == 'M']['edad'],
    y=df[df['género'] == 'F']['edad'],
    alternative='two-sided'
)
```

**Ejemplo 4: Verificar si una columna tiene valores atípicos sistemáticos**

> ¿Los usuarios del plan Premium tienen un patrón de gasto distinto al plan Free?

```python
mwu(
    x=df[df['plan'] == 'Free']['gasto_mensual'],
    y=df[df['plan'] == 'Premium']['gasto_mensual'],
    alternative='less'
)
# Si p < 0.05, el plan importa → candidato a feature o segmentación
```

---

### 2. Feature Engineering

**Ejemplo 5: Validar si un binning creó separación**

> Aplicaste binning a `ingreso` en categorías 'bajo', 'medio', 'alto'. ¿Los grupos resultantes tienen medianas distintas en la variable target?

```python
# Comparar bin bajo vs bin alto
mwu(
    x=df[df['bin_ingreso'] == 'bajo']['target'],
    y=df[df['bin_ingreso'] == 'alto']['target'],
    alternative='two-sided'
)
# p < 0.05 → el binning tiene poder discriminativo
# p > 0.05 → el binning no aporta, considerar otra estrategia
```

**Ejemplo 6: Comparar antes y después de una transformación**

> Aplicaste log-transform a una columna. ¿La distribución cambió significativamente?

```python
import numpy as np

mwu(
    x=df['precio_original'],
    y=np.log1p(df['precio_original']),
    alternative='two-sided'
)
# No esperas que las medianas cambien mucho
# pero es un sanity check de que la transformación no rompió nada
```

**Ejemplo 7: Evaluar normalización por grupos**

> Aplicaste StandardScaler por grupo (por ejemplo, por país). ¿La distribución de `ingreso` quedó similar entre países después de escalar?

```python
mwu(
    x=df[df['país'] == 'Argentina']['ingreso_escalado'],
    y=df[df['país'] == 'Chile']['ingreso_escalado'],
    alternative='two-sided'
)
# p > 0.05 → la normalización equalizó las distribuciones (bueno)
# p < 0.05 → aún hay diferencia (la normalización por grupo no fue suficiente)
```

---

### 3. Selección de modelos

**Ejemplo 8: Comparar errores absolutos entre modelos**

> Tenés dos modelos de regresión. ¿Los errores del modelo A son sistematicamente menores que los del modelo B?

```python
mwu(
    x=abs(residuos_modelo_A),  # errores absolutos del modelo A
    y=abs(residuos_modelo_B),  # errores absolutos del modelo B
    alternative='less'
)
# p < 0.05 → modelo A tiene errores menores
```

**Ejemplo 9: Comparar probabilidades predichas entre clases**

> En un modelo de clasificación, ¿las probabilidades predichas para la clase positiva son más altas en los casos que realmente son positivos?

```python
mwu(
    x=probabilidades[etiquetas_reales == 0],
    y=probabilidades[etiquetas_reales == 1],
    alternative='less'
)
# p < 0.05 → el modelo separa bien las clases
# p > 0.05 → el modelo no está discriminando
```

**Ejemplo 10: Validar si un feature importa en un modelo tree-based**

> ¿Los valores de `antigüedad` son distintos entre los clientes que cancelan y los que no?

```python
# H₀: la antigüedad de los que NO cancelan ≤ la de los que SÍ cancelan
# H₁: la antigüedad de los que NO cancelan > la de los que SÍ cancelan

mwu(
    x=df[df['canceló'] == 0]['antigüedad'],  # Grupo A: los que NO cancelaron
    y=df[df['canceló'] == 1]['antigüedad'],  # Grupo B: los que SÍ cancelaron
    alternative='greater'  # ¿x > y?
)
# p < 0.05 → rechazar H₀ → los que no cancelan tienen mayor antigüedad
# Esta variable es candidata a feature en tu modelo
```

---

### 4. Evaluación post-deploy

**Ejemplo 11: Validar que una mejora de performance es real**

> Desplegás una optimización de base de datos. ¿La latencia de consultas disminuyó?

```python
mwu(
    x=latencia_vieja,
    y=latencia_nueva,
    alternative='greater'
)
# p < 0.05 → la nueva versión es más rápida
```

**Ejemplo 12: Comparar métricas entre versiones de un modelo**

> ¿La nueva versión del modelo tiene menor error MAE que la anterior?

```python
mwu(
    x=errores_modelo_v1,
    y=errores_modelo_v2,
    alternative='greater'
)
# p < 0.05 → v2 tiene errores menores
```

**Ejemplo 13: Verificar que un cambio de UI mejoró la experiencia**

> Rediseñaste el flujo de checkout. ¿El tiempo que tarda el usuario en completar la compra disminuyó?

```python
mwu(
    x=tiempo_checkout_viejo,
    y=tiempo_checkout_nuevo,
    alternative='greater'
)
# p < 0.05 → el nuevo diseño es más rápido
```

---

### 5. Monitoreo y data drift

**Ejemplo 14: Detectar drift en una feature**

> ¿La distribución de `monto_transacción` cambió entre la semana 1 y la semana 4?

```python
mwu(
    x=semana_1['monto_transacción'],
    y=semana_4['monto_transacción'],
    alternative='two-sided'
)
# p < 0.05 → hay drift en esa feature
```

**Ejemplo 15: Comparar distribuciones entre regiones geográficas**

> ¿El comportamiento de compra en Buenos Aires es distinto al del resto del país?

```python
mwu(
    x=df[df['region'] == 'Buenos Aires']['gasto_promedio'],
    y=df[df['region'] != 'Buenos Aires']['gasto_promedio'],
    alternative='two-sided'
)
# p < 0.05 → conviene modelar por separado o incluir región como feature
```

**Ejemplo 16: Validar estabilidad de un modelo en producción**

> ¿Las probabilidades predichas por el modelo en enero son distintas a las de marzo?

```python
mwu(
    x=predicciones_enero,
    y=predicciones_marzo,
    alternative='two-sided'
)
# p > 0.05 → el modelo es estable
# p < 0.05 → posiblemente hay concept drift
```

---

### 6. A/B Testing

**Ejemplo 17: Comparar dos diseños de página**

> El equipo de marketing lanzó una nueva versión del sitio. ¿Los usuarios gastan más en la versión B?

```python
mwu(
    x=gasto_grupo_A,
    y=gasto_grupo_B,
    alternative='less'
)
# p < 0.05 → versión B genera más gasto
# CLES = 0.58 → 58% de probabilidad de que un usuario B gaste más que uno de A
```

**Ejemplo 18: Evaluar dos estrategias de pricing**

> ¿La estrategia de precio dinámico genera más ingresos que el precio fijo?

```python
mwu(
    x=ingresos_precio_fijo,
    y=ingresos_precio_dinámico,
    alternative='less'
)
```

**Ejemplo 19: Comparar dos algoritmos de recomendación**

> ¿Los usuarios del algoritmo nuevo hacen más clics en los productos recomendados?

```python
mwu(
    x=clicks_algoritmo_viejo,
    y=clicks_algoritmo_nuevo,
    alternative='less'
)
```

---

### 7. Validación de supuestos en modelos lineales

**Ejemplo 20: Comparar residuos entre grupos**

> En un modelo de regresión, ¿los residuos son similares para hombres y mujeres? Si no lo son, hay un problema de equidad.

```python
mwu(
    x=residuos[df['género'] == 'M'],
    y=residuos[df['género'] == 'F'],
    alternative='two-sided'
)
# p > 0.05 → el modelo se comporta similar para ambos grupos
# p < 0.05 → hay sesgo de género en el modelo
```

---

## Interpretación de resultados

### Valores que devuelve la prueba

| Salida | Qué significa |
|---|---|
| **U-val** | Estadística de Mann-Whitney. Valores extremos (muy bajos o muy altos) indican diferencias |
| **p-val** | Probabilidad de observar esta diferencia si H₀ fuera cierta |
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

**Importante:** un p < 0.05 con RBC = 0.05 significa que hay una diferencia estadísticamente significativa pero **prácticamente irrelevante**. Siempre reportá el tamaño del efecto junto al p-valor.

### Reglas de decisión para CLES

```
CLES = 0.50 → no hay diferencia (azaro puro)
CLES = 0.55 → efecto pequeño
CLES = 0.60 → efecto mediano
CLES = 0.70 → efecto grande
```

---

## Errores comunes

### 1. Usar Mann-Whitney cuando los datos están emparejados

```python
# MAL: mismos usuarios medidos antes y después
mwu(x=antes, y=después)  # ❌

# BIEN: usar Wilcoxon pareado
from scipy.stats import wilcoxon
wilcoxon(antes, después)  # ✅
```

### 2. Concluir sobre la media

```python
# MAL: "la media del grupo A es mayor que la del grupo B"
# BIEN: "la distribución del grupo A tiende a tener valores mayores que la del grupo B"
```

Mann-Whitney no testea medias. Testea si una distribución tiende a tener valores mayores/menores.

### 3. Ignorar el tamaño del efecto

```python
# Resultado: p = 0.001, RBC = 0.03
# MAL: "hay una diferencia altamente significativa"
# BIEN: "hay una diferencia estadísticamente significativa pero con un efecto muy pequeño (RBC = 0.03)"
```

### 4. Usar p como medida de importancia

```python
# MAL: "p = 0.001, esto es súper importante"
# BIEN: "p = 0.001 indica que es muy improbable que esta diferencia sea por azar"
# La importancia práctica depende del contexto, no del p-valor
```

### 5. Asumir que Mann-Whitney solo compara medianas

Si las distribuciones tienen formas muy distintas (una sesgada a la derecha, otra a la izquierda), Mann-Whitney compara las **distribuciones completas**, no solo las medianas. Podría dar p < 0.05 aunque las medianas sean iguales.

---

## Resumen: ¿cuándo usar cada prueba?

```
¿Tás comparando dos grupos?
│
├── ¿Son independientes (distintos individuos)?
│   ├── ¿Son normales? → t-test independiente
│   └── ¿NO son normales? → Mann-Whitney U
│
├── ¿Son emparejados (mismo individuo)?
│   ├── ¿Son normales? → t-test pareado
│   └── ¿NO son normales? → Wilcoxon pareado
│
└── ¿Es un grupo contra un valor fijo?
    ├── ¿Es normal? → t-test una muestra
    └── ¿NO es normal? → Wilcoxon una muestra
```

---

## Código rápido de referencia

```python
from pingouin import mwu

# Prueba bilateral (¿hay diferencia?)
resultado = mwu(x=grupo_A, y=grupo_B, alternative='two-sided')

# Prueba unilateral (¿A < B?)
resultado = mwu(x=grupo_A, y=grupo_B, alternative='less')

# Prueba unilateral (¿A > B?)
resultado = mwu(x=grupo_A, y=grupo_B, alternative='greater')

# Ver resultado
print(f"U: {resultado['U-val'].values[0]}")
print(f"p: {resultado['p-val'].values[0]}")
print(f"RBC: {resultado['RBC'].values[0]}")
print(f"CLES: {resultado['CLES'].values[0]}")

# Decisión
alpha = 0.05
if resultado['p-val'].values[0] < alpha:
    print("Rechazar H₀ → hay diferencia significativa")
else:
    print("No rechazar H₀ → no hay diferencia significativa")
```
