# Preguntas de Entrevista: Estadística No Paramétrica

## Nivel Básico - Conceptos Fundamentales

### P1: ¿Qué es una prueba no paramétrica y cuándo la usás?

**Respuesta esperada:**
Una prueba que NO asume una distribución específica (como la normal). La usás cuando:
- Los datos NO son normales
- Tenés escalas ordinales
- Tenés outliers que no podés eliminar
- El tamaño de muestra es muy pequeño

---

### P2: Nombrá 3 pruebas no paramétricas y su equivalente paramétrico.

| No paramétrica | Paramétrica |
|----------------|-------------|
| Wilcoxon (una muestra) | t-test una muestra |
| U de Mann-Whitney | t-test muestras independientes |
| Wilcoxon pareado | t-test pareado |
| Kruskal-Wallis | ANOVA de un factor |
| Spearman | Pearson |

---

### P3: ¿Por qué no podemos usar t-test siempre?

**Respuesta esperada:**
El t-test asume:
1. Normalidad de los datos
2. Homogeneidad de varianzas (varianzas iguales)
3. Datos de escala interval o razón

Si se viola la normalidad, el t-test puede dar falsos positivos o falsos negativos.

---

## Nivel Intermedio - Decisión de Pruebas

### P4: Tenés 3 grupos de usuarios (free, basic, premium) y querés saber si el tiempo de uso difiere entre ellos. ¿Qué prueba usás?

**Respuesta:** Kruskal-Wallis

**¿Por qué?**
- 3+ grupos independientes
- Comparar una variable numérica entre grupos
- No paramétrico porque no sabés si es normal

---

### P5: Después de Kruskal-Wallis, el p-valor es 0.03. ¿Qué hacés ahora?

**Respuesta:** Dunn's test (post-hoc)

**¿Por qué?**
- Kruskal-Wallis solo te dice "hay diferencia en AL MENOS un par"
- NO te dice EN QUÉ pares hay diferencia
- Dunn's test identifica qué grupos específicamente difieren

---

### P6: Tenés datos de antes y después de un tratamiento en 20 pacientes. ¿Qué prueba usás?

**Respuesta:** Wilcoxon pareado

**¿Por qué?**
- Mismos sujetos medidos dos veces (datos pareados/dependientes)
- No paramétrico porque puede no ser normal

---

### P7: Tenés dos grupos independientes (hombres vs mujeres) y una variable ordinal (satisfacción: 1-5). ¿Qué prueba usás?

**Respuesta:** U de Mann-Whitney

**¿Por qué?**
- 2 grupos independientes
- Variable ordinal (no se puede usar t-test)
- Mann-Whitney compara rangos, no valores

---

### P8: ¿Cuál es la diferencia entre Wilcoxon pareado y U de Mann-Whitney?

| Aspecto | Wilcoxon pareado | Mann-Whitney |
|---------|------------------|--------------|
| Tipo de datos | Pareados (mismos sujetos) | Independientes |
| Ejemplo | Antes/despues del mismo paciente | Grupo A vs Grupo B |
| Qué compara | Diferencias dentro del par | Rangos entre grupos |

---

## Nivel Avanzado - Escenarios Complex

### P9: Tenés 4 versiones de un algoritmo y 10 datasets. Querés saber si el rendimiento difiere. Kruskal-Wallis da p=0.02. Hacés Dunn's test y encontrás que solo A vs B y A vs C difieren. ¿Qué concluís?

**Respuesta:**
- El algoritmo A tiene rendimiento significativamente diferente a B y C
- Pero NO hay diferencia significativa entre B, C, y D
- Podés concluir que A es distinto, pero no podés decir que B es mejor que C

---

### P10: ¿Por qué no hacés Dunn's test directamente sin Kruskal-Wallis previo?

**Respuesta:**
- Dunn's test es POST-HOC, designed para explorar QUÉ pares difieren
- Si lo hacés sin Kruskal-Wallis, estás haciendo múltiples comparaciones sin control
- Aumentás enormemente la probabilidad de falsos positivos (error tipo I)

---

### P11: Tenés datos con outliers extremos. ¿Por qué Spearman es mejor que Pearson?

**Respuesta:**
- Pearson usa valores originales → los outliers "tiran" de la correlación
- Spearman usa RANGOS → los outliers no distorsionan tanto
- Ejemplo: si tenés salarios de $30K-$50K y uno de $500K, Pearson se va a romper

---

### P12: ¿Cuándo Spearman te da valor bajo pero la relación SÍ existe?

**Respuesta:**
Cuando la relación NO es monotónica. Ejemplo:
- Relación en forma de U (先 baja, luego sube)
- Spearman ve que no siempre sube o siempre baja → ρ bajo
- Pero SÍ hay relación, solo no es monotónica

---

### P13: ¿Por qué en Dunn's test necesitás corregir por múltiples comparaciones?

**Respuesta:**
- Cada comparación tiene 5% de probabilidad de error tipo I
- Si hacés 6 comparaciones (3 grupos): probabilidad acumulada = 1 - (0.95)^6 = 26.5%
- Las correcciones (Bonferroni, Benjamini-Hochberg) reducen esta probabilidad

---

### P14: Compará Bonferroni y Benjamini-Hochberg. ¿Cuál usás en cada caso?

| Aspecto | Bonferroni | Benjamini-Hochberg |
|---------|------------|-------------------|
| Conservadora | Sí | No |
| Falsos positivos | Menos | Más |
| Falsos negativos | Más | Menos |
| Cuándo usar | Investigación médica, ensayos clínicos | Ciencia de datos, exploración |

---

## Nivel Experto - Trampas y Matices

### P15: ¿Qué pasa si usás Spearman cuando la relación ES lineal y no hay outliers?

**Respuesta:**
- Spearman funciona, pero pierde POTENCIA
- Pearson detectaría la correlación con menos datos
- Spearman es "más seguro" pero menos eficiente en este caso

---

### P16: ¿Podés usar ANOVA en vez de Kruskal-Wallis si los datos no son normales?

**Respuesta:**
- Técnicamente sí, pero NO es recomendado
- ANOVA es robusto a leves violaciones de normalidad
- Pero con outliers o distribuciones muy sesgadas, Kruskal-Wallis es más confiable
- Regla práctica: si hay duda, usá Kruskal-Wallis

---

### P17: ¿Qué problema tiene hacer 10 pruebas t-test para comparar 10 grupos en vez de ANOVA/Kruskal-Wallis?

**Respuesta:**
- Probabilidad acumulada de error tipo I = 1 - (0.95)^10 = 40%
- Es decir, 40% de probabilidad de encontrar una diferencia falsa
- Por eso se usan pruebas como Kruskal-Wallis que comparan todo de una

---

### P18: En un EDA, ¿cuándo deberías SUSPEAR de una variable por tener outliers?

**Respuesta:**
NUNCA automáticamente. Primero:
1. Investigá POR QUÉ es outlier (¿error de datos? ¿caso real?)
2. Si es error → corregilo o eliminás ese registro
3. Si es real → considerá Spearman en vez de eliminarlo
4. Solo eliminá si es error de medición y no podés corregir

---

### P19: ¿Qué relación hay entre Spearman y regresión lineal?

**Respuesta:**
- Spearman = Pearson pero sobre RANGOS
- Si rankeás ambos valores y hacés regresión lineal, estás haciendo regresión sobre rangos
- La pendiente de esa regresión se relaciona con ρ (Spearman), no con r (Pearson)

---

### P20: ¿Cuándo Spearman Y Pearson dan valores muy distintos?

**Respuesta:**
Cuando:
1. Hay outliers fuertes (Spearman da más alto, Pearson más bajo)
2. La relación es monotónica no lineal (Spearman detecta, Pearson subestima)
3. Hay una mezcla de relaciones (algunos puntos lineales, otros no)

---

## Preguntas de Código

### P21: ¿Qué código usás para hacer Kruskal-Wallis en Python?

```python
import pingouin as pg

resultado = pg.kruskal(data=datos, dv='variable_dependiente', between='grupo')
print(resultado)
```

---

### P22: ¿Cómo hacés Dunn's test con corrección de Bonferroni?

```python
from scikit_posthocs import posthoc_dunn

posthoc = posthoc_dunn(
    datos_largo,
    val_col='valor',
    group_col='grupo',
    p_adjust='bonferroni'
)
print(posthoc)
```

---

### P23: ¿Cómo calculás Spearman con pingouin y obtenés el IC 95%?

```python
from pingouin import corr

resultados = corr(
    x=datos['x'],
    y=datos['y'],
    alternative='two-sided',
    method='spearman'
)
print(resultados)
```

---

## Preguntas Situacionales (Entrevista Real)

### P24: "Tenés un dataset con 50 variables. ¿Cómo decidís qué pruebas estadísticas correr?"

**Respuesta:**
1. Identificar tipo de cada variable (numérica, ordinal, categórica)
2. Visualizar distribuciones (histogramas, boxplots)
3. Detectar outliers
4. Para pares de variables numéricas: Spearman (más seguro)
5. Para comparar grupos: Kruskal-Wallis → Dunn's test
6. Para antes/despues: Wilcoxon pareado

---

### P25: "Un stakeholder te pregunta '¿por qué usaste Spearman y no Pearson?'. ¿Qué le decís?"

**Respuesta:**
"Spearman es más robusto a outliers y no requiere que la relación sea lineal. En este dataset, los outliers de [variable X] distorsionarían la correlación de Pearson. Spearman detecta la tendencia real sin que los valores extremos afecten el resultado."

---

### P26: "¿Qué harías si Kruskal-Wallis da p=0.06 (no significativo), pero ves diferencias obvias en el gráfico?"

**Respuesta:**
1. Verificar tamaño de muestra (¿es muy pequeño?)
2. Calcular el poder estadístico
3. Considerar si la diferencia práctica es importante aunque no sea estadísticamente significativa
4. Reportar: "No hay evidencia suficiente, pero se observa una tendencia que vale la pena investigar con más datos"

---

### P27: "¿Qué pruebas usarías para evaluar si un modelo de ML es significativamente mejor que otro?"

**Respuesta:**
1. Obtener métricas (accuracy, F1, etc.) en K folds o múltiples datasets
2. Spearman para ver si el ranking de modelos es consistente
3. Si tenés 3+ modelos: Kruskal-Wallis en las métricas
4. Dunn's test para ver qué pares de modelos difieren

---

## Preguntas de Verdadero/Falso

### P28: "Si p-value > 0.05, no hay relación entre las variables"

**FALSO.** p > 0.05 significa que no hay evidencia suficiente para concluir que SÍ hay relación. Puede haber una relación real pero el test no la detectó (error tipo II).

---

### P29: "Spearman siempre es mejor que Pearson"

**FALSO.** Spearman es más seguro pero pierde potencia en relaciones lineales sin outliers. Si sabés que la relación es lineal y limpia, Pearson es más eficiente.

---

### P30: "Dunn's test te dice cuál grupo es el mejor"

**FALSO.** Dunn's test solo te dice qué pares difieren. No te dice cuál es "mejor" o "peor". Solo identifica diferencias estadísticas.

---

## Respuestas Rápidas (Flashcards)

| Pregunta | Respuesta |
|----------|-----------|
| ¿Qué prueba para 2 grupos independientes? | Mann-Whitney U |
| ¿Qué prueba para 2 grupos pareados? | Wilcoxon pareado |
| ¿Qué prueba para 3+ grupos? | Kruskal-Wallis |
| ¿Qué prueba después de Kruskal-Wallis? | Dunn's test |
| ¿Qué correlación para outliers? | Spearman |
| ¿Qué correlación para ordinales? | Spearman |
| ¿Qué correlación para lineal limpia? | Pearson |
| ¿Qué corrección es más conservadora? | Bonferroni |
| ¿Qué corrección es más potente? | Benjamini-Hochberg |
