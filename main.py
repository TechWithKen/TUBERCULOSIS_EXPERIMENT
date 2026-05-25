from TUBERCULOSIS_EXPERIMENT.tbmodel.dataloading import load_data

from TUBERCULOSIS_EXPERIMENT.tbmodel.data_cleaning import (
    clean_data,
    apply_label
)

from TUBERCULOSIS_EXPERIMENT.tbmodel.feature_engineering import (
    split_features,
    get_feature_groups
)

from TUBERCULOSIS_EXPERIMENT.tbmodel.data_preprocessing import (
    build_preprocessor,
    encode_target
)

from TUBERCULOSIS_EXPERIMENT.tbmodel.train import (
    get_model_registry,
    run_batch_grid_search,
    get_trained_models
)

from TUBERCULOSIS_EXPERIMENT.tbmodel.metrics import (
    evaluate_all_models
)

import joblib
import pandas as pd

def run_training_pipeline():

    print(" Training started...\n")


    raw = load_data()

    cleaned = clean_data(
        apply_label(raw)
    )


    split = split_features(cleaned)

    X_train = split["X_train"]
    X_test = split["X_test"]

    y_train = split["y_train"]
    y_test = split["y_test"]

  
    y_train_enc, encoder = encode_target(y_train)

    y_test_enc = encoder.transform(y_test)


    categorical, numerical = get_feature_groups(X_train)

    preprocessor = build_preprocessor(
        categorical,
        numerical
    )


    model_registry = get_model_registry()

    print("🔍 Running hyperparameter tuning...\n")

    grid_results = run_batch_grid_search(
        preprocessor,
        X_train,
        y_train_enc,
        model_registry
    )



    print("\n🧠 Training best models...\n")

    trained_models = get_trained_models(
        grid_results,
        X_train,
        y_train_enc
    )

  

    print("\n📊 Evaluating models...\n")

    results_df = evaluate_all_models(
        trained_models,
        X_test,
        y_test_enc
    )

    print(results_df)


    best_model_name = (
        results_df
        .sort_values("f1", ascending=False)
        .iloc[0]["model"]
    )

    best_model = trained_models[best_model_name]

    print(f"\n🏆 Best Model: {best_model_name}")


    joblib.dump(best_model, "best_model.pkl")

    joblib.dump(encoder, "encoder.pkl")

    print("\n✅ Model artifacts saved")

    print("\n✅ Training completed")

    return {
        "results": results_df,
        "best_model_name": best_model_name,
        "best_model": best_model
    }

def run_inference_pipeline():
    pass

if __name__ == "__main__":
    run_training_pipeline()