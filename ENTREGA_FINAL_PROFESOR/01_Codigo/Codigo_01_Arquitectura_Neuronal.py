"""
FASE 3: Modelo Multi-scale CNN + BiLSTM + Attention
====================================================
Arquitectura:
  INPUT (60, 8)
    ↓
  [Branch CNN k=3]  [Branch CNN k=7]  [Branch CNN k=15]   ← Multi-escala
    ↓ Pool             ↓ Pool             ↓ Pool
    └──────────────────┴──────────────────┘
                      ↓ Concatenar
              [BiLSTM 64 units]
                      ↓
              [Attention (dot-product)]
                      ↓
              [Dropout 0.4]
              [Dense 64, ReLU]
              [Dropout 0.3]
              [Dense 10, Softmax]

Entrenamiento:
  - class_weights para compensar desbalanceo NONE
  - Early Stopping (patience=15) sobre val_loss
  - ReduceLROnPlateau (factor=0.5, patience=7)
  - Checkpoint del mejor modelo (val_accuracy)
  - Label Smoothing = 0.1 (reduce confianza excesiva)

Output:
  dataset_fase2_1h/
    modelo_fase3_best.keras
    historia_entrenamiento.json
    metricas_evaluacion.json
    confusion_matrix.csv
"""

import os
import json
import warnings
import numpy as np

os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
warnings.filterwarnings("ignore")

import keras
from keras import layers, Model, ops
from keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
import pandas as pd

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
DATA_DIR   = r"C:\Users\User\Desktop\VALIDAR HISTORICOS\dataset_fase2_1h"
MODEL_PATH = os.path.join(DATA_DIR, "modelo_fase3_best.keras")

TARGET_LEN = 60
N_FEATURES = 8
N_CLASSES  = 10

# Hiperparámetros del modelo
CNN_FILTERS     = 64
LSTM_UNITS      = 64
DENSE_UNITS     = 64
DROPOUT_LSTM    = 0.35
DROPOUT_DENSE1  = 0.40
DROPOUT_DENSE2  = 0.30
LABEL_SMOOTHING = 0.10

# Hiperparámetros de entrenamiento
EPOCHS      = 100
BATCH_SIZE  = 64
LR_INITIAL  = 1e-3
LR_MIN      = 1e-5
ES_PATIENCE = 15
LR_PATIENCE = 7
LR_FACTOR   = 0.5

SEP = "=" * 70

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

# ─── 1. CARGA DE DATOS ───────────────────────────────────────────────────────
section("1. CARGA DE DATOS")

def load_split(name):
    path = os.path.join(DATA_DIR, f"{name}.npz")
    data = np.load(path)
    X, y = data["X"], data["y"]
    print(f"  {name}: X={X.shape}, y={y.shape}")
    return X, y

X_train, y_train = load_split("train")
X_val,   y_val   = load_split("val")
X_test,  y_test  = load_split("test")

with open(os.path.join(DATA_DIR, "class_weights.json")) as f:
    class_weights_raw = json.load(f)
class_weights = {int(k): v for k, v in class_weights_raw.items()}

with open(os.path.join(DATA_DIR, "label_map.json"), encoding="utf-8") as f:
    lmap = json.load(f)
idx2label = {int(k): v for k, v in lmap["idx2label"].items()}
label_names_ordered = [idx2label[i] for i in range(N_CLASSES)]

print(f"\nClass weights:")
for idx, w in class_weights.items():
    print(f"  {idx2label[idx]:<6}: {w:.4f}")

# ─── 2. CONSTRUCCIÓN DEL MODELO ──────────────────────────────────────────────
section("2. CONSTRUCCION DEL MODELO: Multi-scale CNN + BiLSTM + Attention")

def attention_block(x):
    """
    Soft self-attention sobre la dimensión temporal.
    Genera scores de atención por timestep y devuelve
    la suma ponderada (context vector).
    """
    # x: (batch, timesteps, features)
    score = layers.Dense(1, use_bias=False)(x)       # (batch, timesteps, 1)
    score = layers.Flatten()(score)                   # (batch, timesteps)
    alpha = layers.Softmax(name="attention_weights")(score)  # (batch, timesteps)
    alpha_exp = layers.Reshape((x.shape[1], 1))(alpha)       # (batch, timesteps, 1)
    context = layers.Multiply()([x, alpha_exp])               # (batch, timesteps, features)
    context = ops.sum(context, axis=1)  # (batch, features)
    return context, alpha

def build_model(target_len=TARGET_LEN, n_features=N_FEATURES, n_classes=N_CLASSES):
    inp = keras.Input(shape=(target_len, n_features), name="input")

    # ─ Rama CNN multi-escala ────────────────────────────────────────────────
    branches = []
    for k in [3, 7, 15]:
        x = layers.Conv1D(
            filters=CNN_FILTERS, kernel_size=k,
            padding="same", activation="relu",
            name=f"conv_k{k}"
        )(inp)
        x = layers.BatchNormalization(name=f"bn_k{k}")(x)
        x = layers.Conv1D(
            filters=CNN_FILTERS // 2, kernel_size=k,
            padding="same", activation="relu",
            name=f"conv2_k{k}"
        )(x)
        x = layers.BatchNormalization(name=f"bn2_k{k}")(x)
        branches.append(x)

    # Concatenar ramas → (batch, 60, 96)
    merged = layers.Concatenate(axis=-1, name="merge_branches")(branches)
    merged = layers.Dropout(0.20, name="drop_cnn")(merged)

    # ─ BiLSTM ───────────────────────────────────────────────────────────────
    bilstm_out = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, return_sequences=True, dropout=DROPOUT_LSTM),
        name="bilstm"
    )(merged)  # (batch, 60, 128)

    # ─ Attention ────────────────────────────────────────────────────────────
    context, alpha = attention_block(bilstm_out)  # (batch, 128)

    # ─ Clasificador ─────────────────────────────────────────────────────────
    x = layers.Dropout(DROPOUT_DENSE1, name="drop1")(context)
    x = layers.Dense(DENSE_UNITS, activation="relu", name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(DROPOUT_DENSE2, name="drop2")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inp, outputs=out, name="MultiCNN_BiLSTM_Attention")
    return model

model = build_model()
model.summary()

# Contar parámetros
total_params = model.count_params()
print(f"\nParametros totales: {total_params:,}")

# ─── 3. COMPILACIÓN ──────────────────────────────────────────────────────────
section("3. COMPILACION")

loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_INITIAL),
    loss=loss_fn,
    metrics=["accuracy",
             keras.metrics.Precision(name="precision"),
             keras.metrics.Recall(name="recall")]
)

print(f"  Loss:      CategoricalCrossentropy (label_smoothing={LABEL_SMOOTHING})")
print(f"  Optimizer: Adam (lr={LR_INITIAL})")
print(f"  Metrics:   accuracy, precision, recall")

# ─── 4. PREPARAR DATOS PARA KERAS ────────────────────────────────────────────
section("4. PREPARAR DATOS (one-hot + class_weight dict)")

y_train_oh = keras.utils.to_categorical(y_train, num_classes=N_CLASSES)
y_val_oh   = keras.utils.to_categorical(y_val,   num_classes=N_CLASSES)
y_test_oh  = keras.utils.to_categorical(y_test,  num_classes=N_CLASSES)

print(f"X_train: {X_train.shape}, y_train_oh: {y_train_oh.shape}")
print(f"X_val:   {X_val.shape},   y_val_oh:   {y_val_oh.shape}")
print(f"X_test:  {X_test.shape},  y_test_oh:  {y_test_oh.shape}")

# ─── 5. CALLBACKS ────────────────────────────────────────────────────────────
section("5. CALLBACKS")

callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=ES_PATIENCE,
        restore_best_weights=True,
        verbose=1,
        mode="max"
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=LR_MIN,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
        mode="max"
    )
]
print(f"  EarlyStopping:     patience={ES_PATIENCE}, monitor=val_accuracy")
print(f"  ReduceLROnPlateau: patience={LR_PATIENCE}, factor={LR_FACTOR}")
print(f"  ModelCheckpoint:   {MODEL_PATH}")

# ─── 6. ENTRENAMIENTO ────────────────────────────────────────────────────────
section("6. ENTRENAMIENTO")
print(f"  Epochs:      {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR_INITIAL}")
print(f"  Train size:  {len(X_train)}  |  Val size: {len(X_val)}\n")

history = model.fit(
    X_train, y_train_oh,
    validation_data=(X_val, y_val_oh),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Guardar historia
hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(os.path.join(DATA_DIR, "historia_entrenamiento.json"), "w") as f:
    json.dump(hist_dict, f, indent=2)
print(f"\nHistoria guardada.")

# ─── 7. EVALUACIÓN EN VALIDACIÓN ─────────────────────────────────────────────
section("7. EVALUACION EN VALIDACION")

y_val_pred_proba = model.predict(X_val, verbose=0)
y_val_pred = np.argmax(y_val_pred_proba, axis=1)

acc_val = accuracy_score(y_val, y_val_pred)
f1_val  = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
print(f"Val Accuracy (macro): {acc_val*100:.2f}%")
print(f"Val F1 Score (macro): {f1_val*100:.2f}%")

print("\nClassification Report (VAL):")
print(classification_report(
    y_val, y_val_pred,
    target_names=label_names_ordered,
    zero_division=0
))

# Matriz de confusión en val
cm_val = confusion_matrix(y_val, y_val_pred)
pd.DataFrame(
    cm_val,
    index=label_names_ordered,
    columns=label_names_ordered
).to_csv(os.path.join(DATA_DIR, "confusion_matrix_val.csv"))
print(f"\nMatriz de confusion VAL guardada.")

# ─── 8. EVALUACIÓN EN TEST (OOS) ─────────────────────────────────────────────
section("8. EVALUACION EN TEST (OOS — datos futuros no vistos)")

y_test_pred_proba = model.predict(X_test, verbose=0)
y_test_pred = np.argmax(y_test_pred_proba, axis=1)

acc_test = accuracy_score(y_test, y_test_pred)
f1_test  = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
print(f"TEST Accuracy (macro): {acc_test*100:.2f}%")
print(f"TEST F1 Score (macro): {f1_test*100:.2f}%")

print("\nClassification Report (TEST OOS):")
print(classification_report(
    y_test, y_test_pred,
    target_names=label_names_ordered,
    zero_division=0
))

# Matriz de confusión en test
cm_test = confusion_matrix(y_test, y_test_pred)
pd.DataFrame(
    cm_test,
    index=label_names_ordered,
    columns=label_names_ordered
).to_csv(os.path.join(DATA_DIR, "confusion_matrix_test.csv"))

# Distribución de confianza media por clase
print("\nConfianza media de prediccion por clase (TEST):")
print(f"{'Clase':<8} {'Confianza media':>15}")
print("-" * 26)
for i in range(N_CLASSES):
    mask = y_test_pred == i
    if mask.sum() > 0:
        conf_mean = y_test_pred_proba[mask, i].mean()
        print(f"  {idx2label[i]:<6}  {conf_mean*100:>12.1f}%")

# ─── 9. GUARDAR MÉTRICAS ─────────────────────────────────────────────────────
section("9. GUARDAR METRICAS FINALES")

epochs_done = len(history.history["val_accuracy"])
best_val_acc = max(history.history["val_accuracy"])
best_epoch   = int(np.argmax(history.history["val_accuracy"])) + 1

metrics = {
    "best_epoch":        best_epoch,
    "epochs_trained":    epochs_done,
    "val_accuracy":      round(acc_val * 100, 2),
    "val_f1_macro":      round(f1_val * 100, 2),
    "test_accuracy":     round(acc_test * 100, 2),
    "test_f1_macro":     round(f1_test * 100, 2),
    "best_val_accuracy_epoch": round(best_val_acc * 100, 2),
    "model_path":        MODEL_PATH,
    "total_params":      total_params,
    "hyperparams": {
        "cnn_filters":      CNN_FILTERS,
        "lstm_units":       LSTM_UNITS,
        "dense_units":      DENSE_UNITS,
        "dropout_lstm":     DROPOUT_LSTM,
        "dropout_dense1":   DROPOUT_DENSE1,
        "dropout_dense2":   DROPOUT_DENSE2,
        "label_smoothing":  LABEL_SMOOTHING,
        "batch_size":       BATCH_SIZE,
        "lr_initial":       LR_INITIAL,
    }
}

with open(os.path.join(DATA_DIR, "metricas_evaluacion.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# ─── RESUMEN FINAL ────────────────────────────────────────────────────────────
section("RESUMEN FINAL FASE 3")
print(f"""
Modelo: Multi-scale CNN (k=3,7,15) + BiLSTM + Attention
Parametros: {total_params:,}

Resultados:
  Mejor epoch:    {best_epoch} / {epochs_done}

  VAL:
    Accuracy: {acc_val*100:.2f}%
    F1 macro: {f1_val*100:.2f}%

  TEST (OOS — sep2024/2025):
    Accuracy: {acc_test*100:.2f}%
    F1 macro: {f1_test*100:.2f}%

Archivos generados:
  modelo_fase3_best.keras     → modelo Keras listo para inferencia
  historia_entrenamiento.json → curvas de loss/acc por epoch
  confusion_matrix_val.csv    → matriz de confusion en validacion
  confusion_matrix_test.csv   → matriz de confusion en test OOS
  metricas_evaluacion.json    → resumen completo de metricas

Proximos pasos (Fase 4):
  1. Analizar la matriz de confusion (pares DT/TC, DS/SC, CA/REC)
  2. Temperature Scaling para calibrar probabilidades
  3. Early Detection: re-entrenar con ventanas parciales (50%, 65%, 80%)
""")
print(SEP)
print("  FIN DEL REPORTE — FASE 3 COMPLETADA")
print(SEP)
