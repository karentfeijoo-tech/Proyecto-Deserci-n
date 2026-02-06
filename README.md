· Predicción de Deserción Estudiantil

Proyecto de Ciencia de Datos e Inteligencia Artificial enfocado en predecir la probabilidad de deserción de estudiantes universitarios utilizando **Regresión Logística** y una aplicación interactiva en **Streamlit**.


---

· Objetivo

Identificar estudiantes con **alto riesgo de deserción académica** a partir de su desempeño histórico, permitiendo una intervención temprana por parte de la institución.

---

· Modelo Utilizado

- **Regresión Logística**
- Escalado de variables con `StandardScaler`
- Balanceo de clases con `class_weight='balanced'`

Este modelo permite interpretar fácilmente la influencia de cada variable en la probabilidad de deserción.

---

· Variables Utilizadas

| Variable | Descripción |
|--------|------------|
| `prom_global` | Promedio académico del estudiante |
| `repeticiones` | Número de materias repetidas |
| `carga_academica` | Número de materias cursadas en el periodo |
| `DESERTA` | Variable objetivo (1 = deserta, 0 = no deserta) |

> La variable **asistencia** fue descartada por alta correlación con el promedio académico, evitando multicolinealidad.









