import pandas as pd 
df= pd.read_excel('REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO.xlsx')
print(df.head())        # Primeras filas
#tranforme el promedio de str a flotante
df['PROMEDIO'] = df['PROMEDIO'].str.replace(',', '.').astype('float64')

#agrupacion de los estudiantes por periodo, media del promedio, media del estado, media de las repeticiones de materia
#media de la asistencia, cuantas materias por estudiante, maximo nivel 

df_agg = df.groupby(['ESTUDIANTE', 'PERIODO']).agg(
    prom_global = ('PROMEDIO', 'mean'),
    porc_reprob = ('ESTADO', lambda x: (x == 'REPROBADO').mean()),
    repeticiones = ('NO. VEZ', lambda x: (x > 1).sum()),
    asist_media = ('ASISTENCIA', 'mean'),
    carga_academica = ('COD_MATERIA', 'count'),
    nivel = ('NIVEL', 'max')
).reset_index()

#Cree una columna para que no hayan espacios en el periodo
print(df_agg.head())
df_agg['PERIODO_ORD'] = (
    df_agg['PERIODO']
    .str.replace(' ', '')
)
#Ordena las filas por orden cronologico estudiante y periodo ordinario
df_agg = df_agg.sort_values(['ESTUDIANTE', 'PERIODO_ORD'])

#Aqui detectamos el ultimo periodo de cada estudiante
df_agg['DESERTA'] = (
    df_agg
    .groupby('ESTUDIANTE')['PERIODO_ORD']
    .shift(-1)
    .isna()
    .astype(int)
)

#Toma el valor maximo de periodo_ord, o sea el ultimo periodo que existe en los datos 
ultimo_periodo = df_agg['PERIODO_ORD'].max()
#Se quedan solo las filas que seamn distintas de ese ultimo periodo
#se eliminan las filas del ultimo periodo 
df_agg = df_agg[df_agg['PERIODO_ORD'] != ultimo_periodo]


#Deserta es numerico 0 o 1 
#.map(...) reemplaza cada valor segun el diccionario y guardamos el resultado en la nueva columna llamada disercion
df_agg['DESERCION_TEXT'] = df_agg['DESERTA'].map({
    0: 'NO DESERTA',
    1: 'DESERTA'
})

print(df_agg['DESERCION_TEXT'].value_counts())
print(df_agg['DESERCION_TEXT'].value_counts(normalize=True))



#verificar si los desertores tienen un promedio menor 
umbral = 7.0

desertores_bajo_promedio = df_agg[
    (df_agg['DESERTA'] == 1) &
    (df_agg['prom_global'] < umbral)
]

cantidad = len(desertores_bajo_promedio)
print(f"Desertores con promedio < {umbral}: {cantidad}")


#ANALIS 
#PROMEDIO ACADEMICO POR GRUPO
print(df_agg.groupby('DESERTA')['prom_global'].mean())

#ASISTENCIA POR GRUPO
print(df_agg.groupby('DESERTA')['asist_media'].mean())

#REPETICIONES POR GRUPO
print(df_agg.groupby('DESERTA')['repeticiones'].mean())

#estadisticas descriprivas por grupo 
print(df_agg[['prom_global','asist_media','repeticiones','carga_academica','nivel']].describe())


#crear variable de inasistencia
df_agg['inasistencia'] = 1 - df_agg['asist_media']

#PREPROCESAMIENTO

#definimos las variables x y y 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Variable objetivo
# Variables
X = df_agg[['prom_global', 'repeticiones', 'carga_academica']]
y = df_agg['DESERTA']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
from sklearn.linear_model import LogisticRegression

modelo = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

modelo.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

y_pred = modelo.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(
    y_test,
    y_pred,
    target_names=['NO DESERTA','DESERTA']
))

coeficientes = pd.DataFrame({
    'Variable': X.columns,
    'Coeficiente': modelo.coef_[0]
}).sort_values(by='Coeficiente', ascending=False)

print(coeficientes)

