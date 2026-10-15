# ============================================================
# common_utils.py
# Shared pipeline, callbacks, evaluation, and training loop.
# ============================================================

import os, sys, time, random, gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve, auc, cohen_kappa_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel("ERROR")

# ========== CONFIG ==========
SEED = 42
DATASET_PATH = "processed_data"          # change if needed
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 7                          # adjust if you have more/less
RESULTS_DIR = "results"
MODELS_DIR = "models"
LABEL_SMOOTHING = 0.1
CACHE_IN_MEMORY = True

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def setup_environment():
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) detected: {len(gpus)}")
        # Enable mixed precision only if GPU is available
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled (float16).")
    else:
        print("\nWARNING: No GPU detected. Training will be slow.")
        # Use float32 on CPU to avoid op compatibility issues
        tf.keras.mixed_precision.set_global_policy("float32")
        print("Using float32 precision (CPU).")

def load_and_split_data():
    print("\nScanning dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Folder '{DATASET_PATH}' not found!"); sys.exit(1)
    all_dirs = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    class_names = []
    for d in all_dirs:
        class_path = os.path.join(DATASET_PATH, d)
        if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(class_path)):
            class_names.append(d)
    print("Classes found:", class_names)
    filepaths, labels = [], []
    for idx, name in enumerate(class_names):
        class_path = os.path.join(DATASET_PATH, name)
        imgs = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"  {name}: {len(imgs)}")
        for f in imgs:
            filepaths.append(os.path.join(class_path, f)); labels.append(idx)
    filepaths = np.array(filepaths); labels = np.array(labels)
    print(f"Total: {len(filepaths)} images")
    X_temp, X_test, y_temp, y_test = train_test_split(filepaths, labels, test_size=0.15, stratify=labels, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, stratify=y_temp, random_state=SEED)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, class_names

def build_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.15),
        layers.RandomTranslation(0.15, 0.15),
        layers.RandomBrightness(0.15),
    ], name="aug")

def make_dataset(filepaths, labels, preprocess_fn, augment, training, batch_size=None, cache_path=None):
    batch_size = batch_size or BATCH_SIZE
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    def _decode(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, IMAGE_SIZE)
        # Always cast to float32 – the model will handle precision internally
        return tf.cast(img, tf.float32), label
    ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    if CACHE_IN_MEMORY: ds = ds.cache(cache_path) if cache_path else ds.cache()
    if training: ds = ds.shuffle(len(filepaths), seed=SEED, reshuffle_each_iteration=True)
    def _aug_prep(img, label):
        if training and augment: 
            img = augment(img, training=True)
        return preprocess_fn(img), label
    ds = ds.map(_aug_prep, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds

def get_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weights = np.sqrt(weights)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    print("Class weights:", cw)
    return cw

# ---------- FOCAL LOSS (critical for >80%) ----------
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=NUM_CLASSES)
        ce = tf.keras.backend.categorical_crossentropy(y_true_one_hot, y_pred, from_logits=False)
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        return -alpha * tf.pow(1 - p_t, gamma) * tf.math.log(p_t + tf.keras.backend.epsilon())
    return loss_fn

class MacroF1Callback(callbacks.Callback):
    def __init__(self, val_ds, y_val):
        super().__init__(); self.val_ds = val_ds; self.y_val = y_val
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_pred = np.argmax(self.model.predict(self.val_ds, verbose=0), axis=1)
        macro_f1 = f1_score(self.y_val, y_pred, average="macro")
        logs["val_macro_f1"] = macro_f1
        print(f" - val_macro_f1: {macro_f1:.4f}")

def get_callbacks(model_type, phase, val_ds, y_val, initial_threshold=None):
    checkpoint_path = os.path.join(MODELS_DIR, f"{model_type}_best.keras")
    csv_path = os.path.join(RESULTS_DIR, f"{model_type}_phase{phase}_log.csv")
    es_patience, rlr_patience = (5,3) if phase==1 else (4,2)
    ckpt = dict(filepath=checkpoint_path, monitor="val_macro_f1", mode="max", save_best_only=True, verbose=1)
    if initial_threshold: ckpt["initial_value_threshold"] = initial_threshold
    return [
        MacroF1Callback(val_ds, y_val),
        callbacks.ModelCheckpoint(**ckpt),
        callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=es_patience, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=rlr_patience, min_lr=1e-7, verbose=1),
        callbacks.CSVLogger(csv_path),
    ], checkpoint_path

def classifier_head(base_model):
    return models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32"),
    ])

def compile_model(model, lr=0.001, loss=None):
    if loss is None: loss = focal_loss()
    model.compile(optimizer=optimizers.Adam(lr), loss=loss, metrics=["accuracy"], steps_per_execution=32)
    return model

def evaluate_model(model, test_ds, y_test, class_names, model_type):
    print(f"\n===== EVALUATING {model_type.upper()} =====")
    y_pred_proba = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    acc = float(np.mean(y_pred == y_test))
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    kappa = cohen_kappa_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    # Save confusion matrix
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"CM - {model_type.upper()}")
    plt.colorbar()
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center", color="white" if cm[i,j]>cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_type}_cm.png"), dpi=150)
    plt.close()
    # Save classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(RESULTS_DIR, f"{model_type}_report.csv"))
    # ROC AUC
    y_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))
    try: macro_auc = roc_auc_score(y_bin, y_pred_proba, average="macro", multi_class="ovr")
    except: macro_auc = 0.0
    print(f"Test Acc: {acc*100:.2f}% | Macro F1: {macro_f1*100:.2f}% | AUC: {macro_auc:.4f} | Kappa: {kappa:.4f}")
    return {"test_accuracy": acc, "macro_f1": macro_f1, "macro_auc": macro_auc, "cohen_kappa": kappa}

def append_summary(row):
    path = os.path.join(RESULTS_DIR, "comparison.csv")
    df = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[old["model"] != row["model"]]
        df = pd.concat([old, df], ignore_index=True)
    df = df.sort_values("macro_f1", ascending=False)
    df.to_csv(path, index=False)
    print("\n===== MODEL COMPARISON =====")
    print(df.to_string(index=False))

def run_full_pipeline(model_type, create_model_fn, preprocess_fn, head_epochs, fine_tune_epochs=0, unfreeze_n=0, head_lr=0.001, custom_loss=None):
    start = time.time()
    setup_environment()
    print(f"\n===== TRAINING {model_type.upper()} =====")
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_and_split_data()
    cw = get_class_weights(y_train)
    aug = build_augmentation()
    train_ds = make_dataset(X_train, y_train, preprocess_fn, aug, True)
    val_ds = make_dataset(X_val, y_val, preprocess_fn, None, False)
    model, base = create_model_fn()
    if custom_loss: compile_model(model, head_lr, loss=custom_loss)
    cb1, ckpt = get_callbacks(model_type, 1, val_ds, y_val)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs, callbacks=cb1, class_weight=cw, verbose=1)
    best_f1 = max(h1.history.get("val_macro_f1", [0.0]))
    h2 = None
    if base and fine_tune_epochs>0:
        print(f"\n===== FINE-TUNING {model_type.upper()} =====")
        base.trainable = True
        for layer in base.layers[:-unfreeze_n]: layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization): layer.trainable = False
        compile_model(model, 1e-5, loss=custom_loss if custom_loss else None)
        cb2, _ = get_callbacks(model_type, 2, val_ds, y_val, initial_threshold=best_f1)
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs, callbacks=cb2, class_weight=cw, verbose=1)
    if os.path.exists(ckpt): model.load_weights(ckpt)
    test_ds = make_dataset(X_test, y_test, preprocess_fn, None, False)
    metrics = evaluate_model(model, test_ds, y_test, class_names, model_type)
    save_path = os.path.join(MODELS_DIR, f"{model_type}_trained.keras")
    model.save(save_path)
    row = {"model": model_type, "test_accuracy": metrics["test_accuracy"], "macro_f1": metrics["macro_f1"],
           "macro_auc": metrics["macro_auc"], "cohen_kappa": metrics["cohen_kappa"],
           "train_time_min": (time.time()-start)/60.0, "model_path": save_path}
    append_summary(row)
    print(f"\n{model_type.upper()} DONE! Time: {(time.time()-start)/60:.2f} min")
    del model; gc.collect(); tf.keras.backend.clear_session()
    return row

def show_results(model_type):
    print(f"\nResults for {model_type} saved in '{RESULTS_DIR}'")
    print(f"  - Confusion matrix: {model_type}_cm.png")
    print(f"  - Classification report: {model_type}_report.csv")
    print(f"  - Check '{RESULTS_DIR}/comparison.csv' for the leaderboard.")