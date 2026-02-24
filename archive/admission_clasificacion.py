"""
=============================================================
 ACTIVIDAD: Árbol de Clasificación - Graduate Admissions
 Alumno: [Tu nombre]
 Dataset: Admission_Predict_Ver1_1.csv
=============================================================
"""

# ── LIBRERÍAS ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Necesario en entornos sin pantalla (VS Code remoto / script)
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── CARGA DE DATOS ─────────────────────────────────────────
# Usamos la versión ampliada (500 registros)
df = pd.read_csv('Admission_Predict_Ver1.1.csv')

# Limpiar nombres de columnas (algunos tienen espacios al final)
df.columns = df.columns.str.strip()
print("Columnas:", list(df.columns))
print(f"\nForma del dataset: {df.shape}")

# ── ANÁLISIS DESCRIPTIVO ───────────────────────────────────
print("\n=== Estadísticas descriptivas ===")
print(df.describe().round(2))

print("\n=== Valores nulos por columna ===")
print(df.isnull().sum())

print("\n=== Tipos de datos ===")
print(df.dtypes)

# ── CREACIÓN DE LA VARIABLE RESPUESTA BINARIA ──────────────
# "yes" si Chance of Admit >= 0.6, "no" en caso contrario
df['Admit'] = df['Chance of Admit'].apply(lambda x: 'yes' if x >= 0.6 else 'no')
print("\n=== Distribución de la variable Admit ===")
print(df['Admit'].value_counts())
print(df['Admit'].value_counts(normalize=True).round(3))

# ── ANÁLISIS DESCRIPTIVO GRÁFICO ──────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Análisis Exploratorio - Graduate Admissions', fontsize=14, fontweight='bold')

# Distribución de Chance of Admit
axes[0, 0].hist(df['Chance of Admit'], bins=30, color='steelblue', edgecolor='white')
axes[0, 0].axvline(0.6, color='red', linestyle='--', label='Umbral 0.6')
axes[0, 0].set_title('Distribución de Chance of Admit')
axes[0, 0].set_xlabel('Chance of Admit')
axes[0, 0].legend()

# Conteo de clases
counts = df['Admit'].value_counts()
axes[0, 1].bar(counts.index, counts.values, color=['#e74c3c', '#2ecc71'])
axes[0, 1].set_title('Distribución de la Clase (Admit)')
axes[0, 1].set_ylabel('Frecuencia')
for i, v in enumerate(counts.values):
    axes[0, 1].text(i, v + 3, str(v), ha='center', fontweight='bold')

# CGPA vs Chance of Admit
axes[1, 0].scatter(df['CGPA'], df['Chance of Admit'],
                   c=df['Admit'].map({'yes': '#2ecc71', 'no': '#e74c3c'}),
                   alpha=0.5, s=30)
axes[1, 0].axhline(0.6, color='black', linestyle='--', linewidth=0.8)
axes[1, 0].set_title('CGPA vs Chance of Admit')
axes[1, 0].set_xlabel('CGPA')
axes[1, 0].set_ylabel('Chance of Admit')

# GRE Score vs Chance of Admit
axes[1, 1].scatter(df['GRE Score'], df['Chance of Admit'],
                   c=df['Admit'].map({'yes': '#2ecc71', 'no': '#e74c3c'}),
                   alpha=0.5, s=30)
axes[1, 1].axhline(0.6, color='black', linestyle='--', linewidth=0.8)
axes[1, 1].set_title('GRE Score vs Chance of Admit')
axes[1, 1].set_xlabel('GRE Score')

plt.tight_layout()
plt.savefig('01_analisis_exploratorio.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[Gráfico guardado: 01_analisis_exploratorio.png]")

# Matriz de correlación (solo variables numéricas)
num_cols = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research', 'Chance of Admit']
plt.figure(figsize=(9, 7))
sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=0.5)
plt.title('Matriz de Correlación', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('02_correlacion.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Gráfico guardado: 02_correlacion.png]")

# ── PREPARACIÓN DE DATOS ───────────────────────────────────
# Eliminar columnas irrelevantes
df.drop(columns=['Serial No.', 'Chance of Admit'], inplace=True)

# Codificar variable respuesta: yes=1, no=0
le = LabelEncoder()
df['Admit_bin'] = le.fit_transform(df['Admit'])  # no=0, yes=1

# Variables predictoras (X) y respuesta (y)
feature_cols = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']
X = df[feature_cols]
y = df['Admit_bin']

print(f"\n=== Preparación de datos ===")
print(f"X shape: {X.shape}")
print(f"y distribución: {dict(pd.Series(y).value_counts())}")

# ── DIVISIÓN TRAIN / TEST (70% / 30%) ─────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]} registros | Test: {X_test.shape[0]} registros")

# ── MODELO 1: ÁRBOL DE DECISIÓN ───────────────────────────
print("\n" + "="*55)
print("  MODELO 1: Árbol de Decisión (DecisionTreeClassifier)")
print("="*55)

dt = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,           # Limitar profundidad para evitar sobreajuste
    min_samples_leaf=10,   # Mínimo 10 muestras por hoja
    random_state=42
)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]

acc_dt = accuracy_score(y_test, y_pred_dt)
auc_dt = roc_auc_score(y_test, y_prob_dt)

print(f"\nAccuracy: {acc_dt:.4f}")
print(f"AUC-ROC:  {auc_dt:.4f}")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_dt, target_names=['no', 'yes']))

# Visualizar el árbol
plt.figure(figsize=(18, 8))
plot_tree(dt, feature_names=feature_cols, class_names=['no', 'yes'],
          filled=True, rounded=True, fontsize=9)
plt.title('Árbol de Decisión (max_depth=4)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('03_arbol_decision.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Gráfico guardado: 03_arbol_decision.png]")

# Importancia de variables
feat_imp_dt = pd.Series(dt.feature_importances_, index=feature_cols).sort_values(ascending=True)
plt.figure(figsize=(8, 5))
feat_imp_dt.plot(kind='barh', color='steelblue')
plt.title('Importancia de Variables - Árbol de Decisión', fontweight='bold')
plt.xlabel('Importancia (Gini)')
plt.tight_layout()
plt.savefig('04_importancia_dt.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Gráfico guardado: 04_importancia_dt.png]")

# Reglas del árbol en texto
print("\n=== Reglas del Árbol (formato texto) ===")
print(export_text(dt, feature_names=feature_cols))

# ── MODELO 2: RANDOM FOREST ───────────────────────────────
print("\n" + "="*55)
print("  MODELO 2: Random Forest")
print("="*55)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)

print(f"\nAccuracy: {acc_rf:.4f}")
print(f"AUC-ROC:  {auc_rf:.4f}")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_rf, target_names=['no', 'yes']))

# Importancia de variables RF
feat_imp_rf = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
plt.figure(figsize=(8, 5))
feat_imp_rf.plot(kind='barh', color='darkorange')
plt.title('Importancia de Variables - Random Forest', fontweight='bold')
plt.xlabel('Importancia (Mean Decrease Impurity)')
plt.tight_layout()
plt.savefig('05_importancia_rf.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Gráfico guardado: 05_importancia_rf.png]")

# ── MATRICES DE CONFUSIÓN ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, y_pred, title in zip(
        axes,
        [y_pred_dt, y_pred_rf],
        ['Árbol de Decisión', 'Random Forest']):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['no', 'yes'], yticklabels=['no', 'yes'])
    ax.set_title(f'Matriz de Confusión\n{title}', fontweight='bold')
    ax.set_ylabel('Real')
    ax.set_xlabel('Predicho')

plt.tight_layout()
plt.savefig('06_matrices_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Gráfico guardado: 06_matrices_confusion.png]")

# ── CURVA ROC ─────────────────────────────────────────────
plt.figure(figsize=(7, 5))
for y_prob, label, color in [
        (y_prob_dt, f'Árbol de Decisión (AUC={auc_dt:.3f})', 'steelblue'),
        (y_prob_rf, f'Random Forest (AUC={auc_rf:.3f})', 'darkorange')]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=label, lw=2, color=color)

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Aleatorio (AUC=0.5)')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Comparación de Modelos', fontweight='bold')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('07_curva_roc.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Gráfico guardado: 07_curva_roc.png]")

# ── VALIDACIÓN CRUZADA ────────────────────────────────────
print("\n=== Validación Cruzada (5-fold) ===")
cv_dt = cross_val_score(dt, X, y, cv=5, scoring='accuracy')
cv_rf = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"Árbol de Decisión - Accuracy CV: {cv_dt.mean():.4f} ± {cv_dt.std():.4f}")
print(f"Random Forest     - Accuracy CV: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")

# ── RESUMEN COMPARATIVO ───────────────────────────────────
print("\n" + "="*55)
print("  RESUMEN COMPARATIVO DE MODELOS")
print("="*55)
resumen = pd.DataFrame({
    'Modelo':     ['Árbol de Decisión', 'Random Forest'],
    'Accuracy':   [round(acc_dt, 4), round(acc_rf, 4)],
    'AUC-ROC':    [round(auc_dt, 4), round(auc_rf, 4)],
    'CV Acc (media)': [round(cv_dt.mean(), 4), round(cv_rf.mean(), 4)],
})
print(resumen.to_string(index=False))
print("\n¡Script completado exitosamente! Revisa los archivos PNG generados.")