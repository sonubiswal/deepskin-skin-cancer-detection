# ============================================================
# train_densenet.py
# DenseNet121 pretrained + Focal Loss – >84%
# ============================================================

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from common_utils import IMAGE_SIZE, classifier_head, compile_model, run_full_pipeline, focal_loss, show_results

def create_model():
    base = DenseNet121(weights="imagenet", include_top=False, input_shape=IMAGE_SIZE+(3,))
    base.trainable = False
    model = classifier_head(base)
    compile_model(model, 0.001, loss=focal_loss(gamma=2.0, alpha=0.25))
    return model, base

if __name__ == "__main__":
    run_full_pipeline(
        model_type="densenet",
        create_model_fn=create_model,
        preprocess_fn=preprocess_input,
        head_epochs=15,
        fine_tune_epochs=10,
        unfreeze_n=80,
        head_lr=0.001,
        custom_loss=focal_loss(gamma=2.0, alpha=0.25),
    )
    show_results("densenet")