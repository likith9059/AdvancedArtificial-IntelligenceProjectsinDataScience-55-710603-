"""Week 6 fine-tuning utilities."""
from tensorflow import keras

def unfreeze_last_backbone_layers(model, backbone_name, trainable_fraction=0.20):
    backbone = model.get_layer(backbone_name)
    backbone.trainable = True
    fine_tune_at = int(len(backbone.layers) * (1 - trainable_fraction))
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    return backbone, fine_tune_at, sum(1 for layer in backbone.layers if layer.trainable)

def compile_for_fine_tuning(model, learning_rate=1e-5):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="binary_crossentropy", metrics=["accuracy", keras.metrics.Precision(name="defect_precision"), keras.metrics.Recall(name="defect_recall"), keras.metrics.AUC(name="roc_auc"), keras.metrics.AUC(name="pr_auc", curve="PR")])
    return model
