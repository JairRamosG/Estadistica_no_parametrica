# Guía de Referencia: Prueba Chi-cuadrado de Independencia

## ¿Qué es?

La prueba Chi-cuadrado (χ²) de independencia es una prueba **no paramétrica** que verifica si dos variables **categóricas** están relacionadas o son independientes entre sí.

**Pregunta central:** ¿Existe una asociación estadísticamente significativa entre las dos variables?

**Qué NO hace:**
- No compara medias o medianas (trabaja con frecuencias observadas vs esperadas)
- No asume normalidad
- No requiere datos numéricos (funciona con categorías)
- No indica la dirección de la relación (solo si existe)

---

## Cuándo usarla

### Regla práctica

| Situación | Herramienta |
|---|---|
| Dos variables **categóricas**, muestra **grande** (n ≥ 30, esperados ≥ 5) | **Chi-cuadrado** |
| Dos variables **categóricas**, muestra **pequeña** (esperados < 5) | **Test exacto de Fisher** |
| Una variable **categórica** contra una **esperada teórica** | Chi-cuadrado de bondad de ajuste |
| Variables **numéricas** continuas | U de Mann-Whitney o t-test |

### Condiciones ideales

1. **Variables categóricas** — ambas variables deben ser nominales u ordinales
2. **Independencia de observaciones** — cada dato pertenece a una sola celda de la tabla
3. **Tamaño muestral suficiente** — al menos 80% de las celdas esperadas deben tener frecuencia ≥ 5
4. **Datos no emparejados** — los individuos de una categoría no deben estar relacionados con los de otra

---

## Ejemplos por fase del proyecto de datos

### 1. Exploración de datos (EDA)

**Ejemplo 1: Asociación entre dos categorías**

> Tenés datos de un e-commerce. ¿El color del botón afecta la decisión de compra?

```python
from pingouin import chi2_independence

esperado, observado, resultados = chi2_independence(
    datos, x='grupo', y='compro'
)
# p < 0.05 → hay asociación entre color del botón y compra
# p > 0.05 → son independientes
```

**Ejemplo 2: Verificar si una variable separa clases**

> ¿Los usuarios de diferentes regiones tienen preferencias de producto distintas?

```python
esperado, observado, resultados = chi2_independence(
    datos, x='region', y='producto_favorito'
)
# Si p < 0.05 → la región influye en la preferencia
# Cramer's V indica el tamaño del efecto
```

**Ejemplo 3: Detectar sesgo en categorías**

> ¿La tasa de aprobación de un préstamo es la misma para hombres y mujeres?

```python
esperado, observado, resultados = chi2_independence(
    datos, x='género', y='aprobado'
)
# p < 0.05 → hay diferencia significativa en tasas de aprobación
```

**Ejemplo 4: Comparar proporciones entre grupos**

> ¿La satisfacción del cliente varía según el canal de soporte?

```python
esperado, observado, resultados = chi2_independence(
    datos, x='canal_soporte', y='satisfacción'
)
```

---

### 2. Feature Engineering

**Ejemplo 5: Validar si una codificación one-hot tiene poder discriminativo**

> Codificaste `color_ohe` y querés saber si esa categoría impacta en `compró`.

```python
esperado, observado, resultados = chi2_independence(
    datos, x='color_ohe', y='compró'
)
# p < 0.05 → la categoría es relevante para predecir
# p > 0.05 → considerar eliminar el feature
```

**Ejemplo 6: Evaluar si un binning categórico crea separación**

> Agrupaste `ingreso` en categorías 'bajo', 'medio', 'alto'. ¿Estos grupos tienen distribuciones distintas del target?

```python
esperado, observado, resultados = chi2_independence(
    datos, x='bin_ingreso', y='target'
)
# p < 0.05 → el binning tiene poder discriminativo
```

**Ejemplo 7: Verificar equilibrio de clases**

> ¿Tu dataset de entrenamiento tiene distribución equilibrada entre clases?

```python
from scipy.stats import chisquare

frecuencias = datos['clase'].value_counts().values
chi2, p = chisquare(frecuencias)
# p > 0.05 → las clases están equilibradas
# p < 0.05 → hay desbalance (considerar oversampling/undersampling)
```

---

### 3. Selección de modelos

**Ejemplo 8: Evaluar si un feature categórico importa en un modelo**

> ¿La categoría de `plan_usuario` (Free, Pro, Enterprise) está asociada con `canceló`?

```python
esperado, observado, resultados = chi2_independence(
    datos, x='plan_usuario', y='canceló'
)
# Si p < 0.05 y Cramer's V alto → feature candidato importante
```

**Ejemplo 9: Comparar rendimiento entre clasificadores**

> Dos modelos clasifican en categorías. ¿Sus distribuciones de predicciones son distintas?

```python
# Crear tabla de contingencia entre predicciones de ambos modelos
import pandas as pd
tabla = pd.crosstab(predicciones_modelo_A, predicciones_modelo_B)
from scipy.stats import chi2_contingency
chi2, p, dof, expected = chi2_contingency(tabla)
# p < 0.05 → los modelos predicen differently
```

**Ejemplo 10: Validar si una variable categórica es buena para estratificación**

> ¿`nivel_educación` segmenta bien el comportamiento de compra?

```python
esperado, observado, resultados = chi2_independence(
    datos, x='nivel_educación', y='compró'
)
# Cramer's V cercano a 0 → mala estratificación
# Cramer's V > 0.3 → buena estratificación
```

---

### 4. Evaluación post-deploy

**Ejemplo 11: Validar que un cambio de UI mejoró la experiencia**

> Rediseñaste el flujo de checkout. ¿La tasa de abandono cambió entre la versión vieja y nueva?

```python
esperado, observado, resultados = chi2_independence(
    datos, x='version_ui', y='completó_compra'
)
# p < 0.05 → el cambio tuvo efecto significativo
```

**Ejemplo 12: Comparar tasas de conversión entre campañas**

> ¿La campaña de email genera más conversiones que la de SMS?

```python
esperado, observado, resultados = chi2_independence(
    datos, x='canal_marketing', y='conversión'
)
# p < 0.05 → hay diferencia entre canales
```

**Ejemplo 13: Verificar que un cambio de precios no afectó la distribución de productos**

> ¿Los usuarios siguen comprando las mismas categorías de producto después del ajuste de precios?

```python
esperado, observado, resultados = chi2_independence(
    datos, x='periodo', y='categoría_producto'
)
# p > 0.05 → la distribución se mantuvo estable
```

---

### 5. Monitoreo y data drift

**Ejemplo 14: Detectar drift en variables categóricas**

> ¿La distribución de `método_pago` cambió entre enero y marzo?

```python
# Unir datos de ambos períodos
datos_ambos = pd.concat([enero, marzo])
esperado, observado, resultados = chi2_independence(
    datos_ambos, x='período', y='método_pago'
)
# p < 0.05 → hay drift en esa variable categórica
```

**Ejemplo 15: Comparar distribuciones entre regiones**

> ¿El comportamiento de compra es consistente entre Buenos Aires y el resto del país?

```python
esperado, observado, resultados = chi2_independence(
    datos, x='region', y='compró'
)
# p < 0.05 → conviene modelar por separado o incluir región como feature
```

**Ejemplo 16: Validar estabilidad de un modelo en producción**

> ¿Las categorías predichas por el modelo en enero son iguales a las de marzo?

```python
# Crear tabla de contingencia
tabla = pd.crosstab(predicciones_enero, predicciones_marzo)
chi2, p, dof, expected = chi2_contingency(tabla)
# p > 0.05 → el modelo es estable en sus predicciones
# p < 0.05 → posiblemente hay concept drift
```

---

### 6. A/B Testing

**Ejemplo 17: Comparar dos versiones de una página**

> El equipo de marketing lanzó una nueva versión del sitio. ¿La tasa de conversión cambió?

```python
esperado, observado, resultados = chi2_independence(
    datos, x='versión', y='conversión'
)
# p < 0.05 → la nueva versión tiene diferente tasa de conversión
# Revisar las frecuencias observadas para ver en qué dirección
```

**Ejemplo 18: Evaluar dos estrategias de pricing**

> ¿La estrategia de precio dinámico genera una distribución distinta de categorías de productos vendidos?

```python
esperado, observado, resultados = chi2_independence(
    datos, x='estrategia_precio', y='categoría_producto'
)
```

**Ejemplo 19: Comparar dos algoritmos de recomendación**

> ¿Los usuarios del algoritmo nuevo hacen clics en categorías de productos distintas?

```python
esperado, observado, resultados = chi2_independence(
    datos, x='algoritmo', y='categoría_click'
)
# p < 0.05 → los algoritmos generan patrones de clics distintos
```

---

### 7. Validación de supuestos en modelos

**Ejemplo 20: Verificar independencia de residuos categóricos**

> En un modelo de clasificación, ¿los errores del modelo son independientes de una variable sensible?

```python
# Crear variable de error: 1 si predijo mal, 0 si acertó
datos['error'] = (datos['predicción'] != datos['real']).astype(int)

esperado, observado, resultados = chi2_independence(
    datos, x='género', y='error'
)
# p > 0.05 → el error es independiente del género (bueno)
# p < 0.05 → hay sesgo: el modelo falla más para un género
```

---

## Interpretación de resultados

### Valores que devuelve la prueba (pingouin)

| Salida | Qué significa |
|---|---|
| **chi2** | Estadística Chi-cuadrado. Valores mayores indican mayor diferencia entre observado y esperado |
| **pval** | Probabilidad de observar esta diferencia si H₀ fuera cierta (independencia) |
| **dof** | Grados de libertad = (filas - 1) × (columnas - 1) |
| **cramer** | Tamaño del efecto (V de Cramer). Rango: 0 a 1 |
| **power** | Potencia post-hoc de la prueba |

### Reglas de decisión para p-valor

```
p < 0.01  → evidencia muy fuerte contra H₀ (asociación fuerte)
p < 0.05  → evidencia fuerte contra H₀ (estándar)
p < 0.10  → evidencia débil, posible tendencia
p ≥ 0.10  → sin evidencia suficiente
```

### Reglas de decisión para V de Cramer (tamaño del efecto)

```
|V| < 0.1   → efecto despreciable
|V| ≥ 0.1   → efecto pequeño
|V| ≥ 0.3   → efecto mediano
|V| ≥ 0.5   → efecto grande
```

### Relación entre Cramer's V y omega (ω)

Si necesitás reportar omega en lugar de Cramer:

$$\omega = V\sqrt{k-1}$$

donde `k` es el mínimo entre filas y columnas de la tabla de contingencia.

**Importante:** un p < 0.05 con Cramer's V = 0.05 significa que hay una diferencia estadísticamente significativa pero **prácticamente irrelevante**. Siempre reportá el tamaño del efecto junto al p-valor.

---

## Errores comunes

### 1. Usar Chi-cuadrado con datos numéricos continuos

```python
# MAL: comparar edades entre grupos
chi2_independence(datos, x='grupo', y='edad')  # ❌ edad es numérica continua

# BIEN: si querés comparar numéricos, usar Mann-Whitney o t-test
from pingouin import mwu
mwu(x=grupo_A['edad'], y=grupo_B['edad'])
```

### 2. Concluir sobre la dirección de la relación

```python
# MAL: "el grupo A tiene más compras que el grupo B"
# BIEN: "hay una asociación estadísticamente significativa entre el grupo y la compra"
# Para ver la dirección, revisar las frecuencias observadas manualmente
```

Chi-cuadrado solo indica si hay asociación, no la dirección.

### 3. Ignorar el tamaño del efecto

```python
# Resultado: p = 0.001, Cramer's V = 0.03
# MAL: "hay una asociación altamente significativa"
# BIEN: "hay una asociación estadísticamente significativa pero con un efecto muy pequeño (V = 0.03)"
```

### 4. Usar Chi-cuadrado cuando los esperados son muy bajos

```python
# MAL: tabla con celdas esperadas < 5
# BIEN: usar test exacto de Fisher
from scipy.stats import fisher_exact
odds_ratio, p = fisher_exact(tabla_2x2)
```

**Regla:** si más del 20% de las celdas esperadas tienen frecuencia < 5, usar Fisher.

### 5. Olvidar que Chi-cuadrado no asume causalidad

```python
# MAL: "el color del botón CAUSA más compras"
# BIEN: "hay una asociación entre el color del botón y la compra"
# Para causalidad se necesita experimento controlado + randomización
```

### 6. Confundir independencia con homogeneidad

- **Prueba de independencia:** ¿dos variables están relacionadas? (una sola muestra)
- **Prueba de homogeneidad:** ¿las distribuciones de una variable son iguales en diferentes poblaciones? (múltiples muestras)

Ambas usan Chi-cuadrado, pero la interpretación difiere.

---

## Resumen: ¿cuándo usar cada prueba?

```
¿Tenés variables categóricas?
│
├── ¿Dos variables categóricas?
│   ├── ¿Muestra grande (esperados ≥ 5)? → Chi-cuadrado
│   └── ¿Muestra chica (esperados < 5)? → Fisher exacto
│
├── ¿Una categórica + una numérica?
│   ├── ¿2 grupos, normales? → t-test independiente
│   ├── ¿2 grupos, NO normales? → Mann-Whitney U
│   └── ¿≥3 grupos? → Kruskal-Wallis
│
└── ¿Una categórica contra distribución teórica?
    └── Chi-cuadrado de bondad de ajuste
```

---

## Código rápido de referencia

```python
from pingouin import chi2_independence
from scipy.stats import fisher_exact
import pandas as pd

# ═══════════════════════════════════════════
# Chi-cuadrado de independencia (muestra grande)
# ═══════════════════════════════════════════

esperado, observado, resultados = chi2_independence(
    datos, x='variable_cat_1', y='variable_cat_2'
)

# Ver resultados principales
print(f"χ²: {resultados['chi2'].values[0]:.4f}")
print(f"p: {resultados['pval'].values[0]:.6f}")
print(f"V de Cramer: {resultados['cramer'].values[0]:.4f}")
print(f"Potencia: {resultados['power'].values[0]:.4f}")

# Ver tablas
print("\n--- Esperado ---")
print(esperado)
print("\n--- Observado ---")
print(observado)

# ═══════════════════════════════════════════
# Decisión
# ═══════════════════════════════════════════

alpha = 0.05
if resultados['pval'].values[0] < alpha:
    print("\n✅ Rechazar H₀ → hay asociación significativa")
    
    # Tamaño del efecto
    v = resultados['cramer'].values[0]
    if v < 0.1:
        print("   Efecto: despreciable")
    elif v < 0.3:
        print("   Efecto: pequeño")
    elif v < 0.5:
        print("   Efecto: mediano")
    else:
        print("   Efecto: grande")
else:
    print("\n❌ No rechazar H₀ → no hay asociación significativa")

# ═══════════════════════════════════════════
# Fisher exacto (muestra chica)
# ═══════════════════════════════════════════

# Para tablas 2x2 con frecuencias bajas
tabla = pd.crosstab(datos['var_1'], datos['var_2'])
odds_ratio, p_fisher = fisher_exact(tabla)
print(f"\nFisher exacto - p: {p_fisher:.6f}")
```

---

## Tamaño de muestra mínimo

Para que Chi-cuadrado sea válido:

| Condición | Mínimo |
|---|---|
| Todas las celdas esperadas ≥ 5 | n ≥ 30 (regla general) |
| 80% de celdas esperadas ≥ 5 | Aceptar con precaución |
| Alguna celda esperada < 5 | Usar **Fisher exacto** |

**Power analysis** (para calcular tamaño de muestra antes del experimento):

```python
from statsmodels.stats.power import GofChisquarePower
import math

w = 0.4       # Tamaño del efecto (0.1=pequeño, 0.3=mediano, 0.5=grande)
alpha = 0.05  # Nivel de significancia
power = 0.9   # Potencia deseada
df = (R-1) * (C-1)  # Grados de libertad de la tabla

analisis = GofChisquarePower()
n = analisis.solve_power(
    effect_size=w,
    alpha=alpha,
    power=power,
    n_bins=df + 1
)
n = math.ceil(n)
print(f"Tamaño total de muestra: {n}")
print(f"Por grupo: {n // 2}")
```
