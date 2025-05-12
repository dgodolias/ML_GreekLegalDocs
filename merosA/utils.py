import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb # Added for Scikit-learn XGBoost comparison if needed in future



def preprocess_data(processor, x_train, y_train, x_val, y_val, x_test, y_test):

    binary_vectors_train = processor.map_reviews_to_binary_vectors(x_train)
    binary_vectors_val   = processor.map_reviews_to_binary_vectors(x_val)
    binary_vectors_test  = processor.map_reviews_to_binary_vectors(x_test)

    binary_vectors_train = np.array(binary_vectors_train)
    binary_vectors_val   = np.array(binary_vectors_val)
    binary_vectors_test  = np.array(binary_vectors_test)
    
    y_train = np.array(y_train).flatten()
    y_val   = np.array(y_val).flatten()
    y_test  = np.array(y_test).flatten()

    if binary_vectors_train.ndim == 1:
        binary_vectors_train = np.expand_dims(binary_vectors_train, axis=0)
    if binary_vectors_val.ndim == 1:
        binary_vectors_val = np.expand_dims(binary_vectors_val, axis=0)
    if binary_vectors_test.ndim == 1:
        binary_vectors_test = np.expand_dims(binary_vectors_test, axis=0)
        

    top_k = int(processor.vocab_size / 10)
    top_word_indices = processor.calculate_information_gain(binary_vectors_train, y_train, top_k=top_k)
    

    binary_vectors_train = binary_vectors_train[:, top_word_indices]
    binary_vectors_val   = binary_vectors_val[:,   top_word_indices]
    binary_vectors_test  = binary_vectors_test[:,  top_word_indices]
    

    return binary_vectors_train, y_train, \
           binary_vectors_val,   y_val,   \
           binary_vectors_test,  y_test




def runMyAdaBoost(adaboost, 
                  X_train, y_train_transformed, 
                  X_val,   y_val_transformed, 
                  X_test,  y_test_transformed):

    print("Training Custom AdaBoost...")
    adaboost.fit(X_train, y_train_transformed)
    
    y_pred_train = adaboost.predict(X_train)
    y_pred_val   = adaboost.predict(X_val)
    y_pred_test  = adaboost.predict(X_test)

    y_pred_train = np.where(y_pred_train == -1, 0, 1)
    y_pred_val   = np.where(y_pred_val   == -1, 0, 1)
    y_pred_test  = np.where(y_pred_test  == -1, 0, 1)
    
    print("y TRAIN SUM", np.sum(y_pred_train))
    print("y VAL SUM", np.sum(y_pred_val))
    print("y TEST SUM", np.sum(y_pred_test))
    
    return y_pred_train, y_pred_val, y_pred_test


def runMyXGBoost(xgboost_model,
                 X_train, y_train,
                 X_val,   y_val,
                 X_test,  y_test):
    """
    Trains the XGBoost model and returns predictions.
    Assumes y_train, y_val, y_test are already in the correct format (0 or 1).
    """
    print("\nTraining Custom XGBoost...")
    xgboost_model.fit(X_train, y_train)

    print("Predicting with Custom XGBoost...")
    y_pred_train = xgboost_model.predict(X_train)
    y_pred_val   = xgboost_model.predict(X_val)
    y_pred_test  = xgboost_model.predict(X_test)

    # Predictions from our XGBoost class are already 0 or 1 due to (preds > 0.5).astype(int)
    print(f"y TRAIN SUM (XGBoost): {np.sum(y_pred_train)}")
    print(f"y VAL SUM (XGBoost): {np.sum(y_pred_val)}")
    print(f"y TEST SUM (XGBoost): {np.sum(y_pred_test)}")

    return y_pred_train, y_pred_val, y_pred_test


def runMyModel_scratch(model,
                       X_train, y_train,
                       X_val,   y_val,
                       X_test,  y_test,
                       model_name="Custom Model"):
    """
    Trains a generic from-scratch model and returns predictions.
    Assumes y_train, y_val, y_test are already in the correct format (0 or 1).
    The model should have fit() and predict() methods.
    """
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)

    print(f"Predicting with {model_name}...")
    y_pred_train = model.predict(X_train)
    y_pred_val   = model.predict(X_val)
    y_pred_test  = model.predict(X_test)

    print(f"y TRAIN SUM ({model_name}): {np.sum(y_pred_train)}")
    print(f"y VAL SUM ({model_name}): {np.sum(y_pred_val)}")
    print(f"y TEST SUM ({model_name}): {np.sum(y_pred_test)}")
    
    compute_and_print_metrics(y_train, y_pred_train, dataset_description="TRAINING data", model_name=model_name)
    compute_and_print_metrics(y_val, y_pred_val, dataset_description="VALIDATION data", model_name=model_name)


    return y_pred_train, y_pred_val, y_pred_test



def runScikitAdaBoost(X_train, y_train, X_test, y_test, n_estimators=100):

    scikit_precisions = []
    scikit_recalls = []
    scikit_f1_scores = []
    scikit_accuracies = []

    scikit_micro_precisions = []
    scikit_micro_recalls = []
    scikit_micro_f1_scores = []

    scikit_macro_precisions = []
    scikit_macro_recalls = []
    scikit_macro_f1_scores = []
    
    scikit_iterations = list(range(n_estimators // 10, n_estimators + 1, n_estimators // 10))

    for iters in scikit_iterations:
        sk_model = AdaBoostClassifier(n_estimators=iters)
        sk_model.fit(X_train, y_train)
        y_pred_sk = sk_model.predict(X_test)
        
        # Positive (class=1) metrics
        precision, recall, f1 = compute_metrics(y_test, y_pred_sk)
        
        # Overall accuracy
        accuracy = compute_accuracy(y_test, y_pred_sk)
        
        # Micro / Macro averages
        metrics_matrix = compute_metrics_matrix(y_test, y_pred_sk)
        # [3, :] -> Micro-average row; [4, :] -> Macro-average row
        micro_precision = float(metrics_matrix[3, 1])
        micro_recall    = float(metrics_matrix[3, 2])
        micro_f1        = float(metrics_matrix[3, 3])
        macro_precision = float(metrics_matrix[4, 1])
        macro_recall    = float(metrics_matrix[4, 2])
        macro_f1        = float(metrics_matrix[4, 3])
        
        # Append all metrics
        scikit_precisions.append(precision)
        scikit_recalls.append(recall)
        scikit_f1_scores.append(f1)
        scikit_accuracies.append(accuracy)

        scikit_micro_precisions.append(micro_precision)
        scikit_micro_recalls.append(micro_recall)
        scikit_micro_f1_scores.append(micro_f1)

        scikit_macro_precisions.append(macro_precision)
        scikit_macro_recalls.append(macro_recall)
        scikit_macro_f1_scores.append(macro_f1)
        
        # Print them
        print(f"Scikit n_estimators {iters}: "
              f"Accuracy={accuracy:.4f}, "
              f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, "
              f"Micro-Precision={micro_precision:.4f}, Micro-Recall={micro_recall:.4f}, Micro-F1={micro_f1:.4f}, "
              f"Macro-Precision={macro_precision:.4f}, Macro-Recall={macro_recall:.4f}, Macro-F1={macro_f1:.4f}")
    
    return (scikit_iterations,
            scikit_accuracies,
            scikit_precisions,
            scikit_recalls,
            scikit_f1_scores,
            scikit_micro_precisions,
            scikit_micro_recalls,
            scikit_micro_f1_scores,
            scikit_macro_precisions,
            scikit_macro_recalls,
            scikit_macro_f1_scores)



def plot_adaboost_comparison_metrics(custom_adaboost, 
                                     X_train, y_train, 
                                     X_val, y_val, 
                                     X_test, y_test, 
                                     interval=None):

    n_clf = len(custom_adaboost.clfs)
    print(f"Total number of stumps: {n_clf}")
    if interval is None:
        interval = max(1, n_clf // 10)
    
    custom_iterations = []
    custom_train_acc = []
    custom_train_prec = []
    custom_train_rec = []
    custom_train_f1 = []
    custom_train_micro_prec = []
    custom_train_micro_rec = []
    custom_train_micro_f1 = []
    custom_train_macro_prec = []
    custom_train_macro_rec = []
    custom_train_macro_f1 = []
    
    custom_val_acc = []
    custom_val_prec = []
    custom_val_rec = []
    custom_val_f1 = []
    custom_val_micro_prec = []
    custom_val_micro_rec = []
    custom_val_micro_f1 = []
    custom_val_macro_prec = []
    custom_val_macro_rec = []
    custom_val_macro_f1 = []
    
    custom_test_final = None
    for i in range(interval, n_clf + 1, interval):
        print(f"\n=== CUSTOM AdaBoost n_estimators={i} ===")
        ens_train = np.zeros(X_train.shape[0])
        for clf in custom_adaboost.clfs[:i]:
            ens_train += clf.alpha * clf.predict(X_train)
        y_pred_train = np.sign(ens_train)
        y_pred_train = np.where(y_pred_train == -1, 0, 1)
        _print_table_iteration("TRAIN", y_train, y_pred_train)
        
        ens_val = np.zeros(X_val.shape[0])
        for clf in custom_adaboost.clfs[:i]:
            ens_val += clf.alpha * clf.predict(X_val)
        y_pred_val = np.sign(ens_val)
        y_pred_val = np.where(y_pred_val == -1, 0, 1)
        _print_table_iteration("VAL", y_val, y_pred_val)
        
        ens_test = np.zeros(X_test.shape[0])
        for clf in custom_adaboost.clfs[:i]:
            ens_test += clf.alpha * clf.predict(X_test)
        y_pred_test = np.sign(ens_test)
        y_pred_test = np.where(y_pred_test == -1, 0, 1)
        if i == n_clf:
            _print_table_iteration("TEST", y_test, y_pred_test)
            custom_test_final = y_pred_test
        
        custom_iterations.append(i)
        acc_tr = compute_accuracy(y_train, y_pred_train)
        p_tr, r_tr, f1_tr = compute_metrics(y_train, y_pred_train)
        mat_tr = compute_metrics_matrix(y_train, y_pred_train)
        custom_train_acc.append(acc_tr)
        custom_train_prec.append(p_tr)
        custom_train_rec.append(r_tr)
        custom_train_f1.append(f1_tr)
        custom_train_micro_prec.append(float(mat_tr[3,1]))
        custom_train_micro_rec.append(float(mat_tr[3,2]))
        custom_train_micro_f1.append(float(mat_tr[3,3]))
        custom_train_macro_prec.append(float(mat_tr[4,1]))
        custom_train_macro_rec.append(float(mat_tr[4,2]))
        custom_train_macro_f1.append(float(mat_tr[4,3]))
        
        acc_val = compute_accuracy(y_val, y_pred_val)
        p_val, r_val, f1_val = compute_metrics(y_val, y_pred_val)
        mat_val = compute_metrics_matrix(y_val, y_pred_val)
        custom_val_acc.append(acc_val)
        custom_val_prec.append(p_val)
        custom_val_rec.append(r_val)
        custom_val_f1.append(f1_val)
        custom_val_micro_prec.append(float(mat_val[3,1]))
        custom_val_micro_rec.append(float(mat_val[3,2]))
        custom_val_micro_f1.append(float(mat_val[3,3]))
        custom_val_macro_prec.append(float(mat_val[4,1]))
        custom_val_macro_rec.append(float(mat_val[4,2]))
        custom_val_macro_f1.append(float(mat_val[4,3]))
    
    scikit_iterations = list(range(n_clf // 10, n_clf + 1, n_clf // 10))
    scikit_train_acc = []
    scikit_train_prec = []
    scikit_train_rec = []
    scikit_train_f1 = []
    scikit_train_micro_prec = []
    scikit_train_micro_rec = []
    scikit_train_micro_f1 = []
    scikit_train_macro_prec = []
    scikit_train_macro_rec = []
    scikit_train_macro_f1 = []
    
    scikit_val_acc = []
    scikit_val_prec = []
    scikit_val_rec = []
    scikit_val_f1 = []
    scikit_val_micro_prec = []
    scikit_val_micro_rec = []
    scikit_val_micro_f1 = []
    scikit_val_macro_prec = []
    scikit_val_macro_rec = []
    scikit_val_macro_f1 = []
    
    scikit_test_final = None
    for iters in scikit_iterations:
        print(f"\n=== Scikit AdaBoost n_estimators={iters} ===")
        sk_model = AdaBoostClassifier(n_estimators=iters)
        sk_model.fit(X_train, y_train)
        
        y_pred_train_sk = sk_model.predict(X_train)
        _print_table_iteration("TRAIN", y_train, y_pred_train_sk)
        y_pred_val_sk = sk_model.predict(X_val)
        _print_table_iteration("VAL", y_val, y_pred_val_sk)
        y_pred_test_sk = sk_model.predict(X_test)
        if iters == scikit_iterations[-1]:
            _print_table_iteration("TEST", y_test, y_pred_test_sk)
            scikit_test_final = y_pred_test_sk
        
        scikit_train_acc.append(compute_accuracy(y_train, y_pred_train_sk))
        p_tr_sk, r_tr_sk, f1_tr_sk = compute_metrics(y_train, y_pred_train_sk)
        mat_tr_sk = compute_metrics_matrix(y_train, y_pred_train_sk)
        scikit_train_prec.append(p_tr_sk)
        scikit_train_rec.append(r_tr_sk)
        scikit_train_f1.append(f1_tr_sk)
        scikit_train_micro_prec.append(float(mat_tr_sk[3,1]))
        scikit_train_micro_rec.append(float(mat_tr_sk[3,2]))
        scikit_train_micro_f1.append(float(mat_tr_sk[3,3]))
        scikit_train_macro_prec.append(float(mat_tr_sk[4,1]))
        scikit_train_macro_rec.append(float(mat_tr_sk[4,2]))
        scikit_train_macro_f1.append(float(mat_tr_sk[4,3]))
        
        scikit_val_acc.append(compute_accuracy(y_val, y_pred_val_sk))
        p_val_sk, r_val_sk, f1_val_sk = compute_metrics(y_val, y_pred_val_sk)
        mat_val_sk = compute_metrics_matrix(y_val, y_pred_val_sk)
        scikit_val_prec.append(p_val_sk)
        scikit_val_rec.append(r_val_sk)
        scikit_val_f1.append(f1_val_sk)
        scikit_val_micro_prec.append(float(mat_val_sk[3,1]))
        scikit_val_micro_rec.append(float(mat_val_sk[3,2]))
        scikit_val_micro_f1.append(float(mat_val_sk[3,3]))
        scikit_val_macro_prec.append(float(mat_val_sk[4,1]))
        scikit_val_macro_rec.append(float(mat_val_sk[4,2]))
        scikit_val_macro_f1.append(float(mat_val_sk[4,3]))

    fig, axs = plt.subplots(2, 2, figsize=(14,10))
    fig.suptitle("Train Metrics: Custom vs. Scikit AdaBoost")
    ax = axs[0,0]
    ax.plot(custom_iterations, custom_train_acc, marker='o', label='Custom', color='blue')
    ax.plot(scikit_iterations, scikit_train_acc, marker='x', linestyle='--', label='Scikit', color='blue')
    ax.set_title("Accuracy")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)
    
    ax = axs[0,1]
    ax.plot(custom_iterations, custom_train_prec, marker='o', label='Precision', color='red')
    ax.plot(custom_iterations, custom_train_micro_prec, marker='^', label='Micro', color='orange')
    ax.plot(custom_iterations, custom_train_macro_prec, marker='s', label='Macro', color='brown')
    ax.plot(scikit_iterations, scikit_train_prec, marker='x', linestyle='--', label='Precision (Scikit)', color='red')
    ax.plot(scikit_iterations, scikit_train_micro_prec, marker='v', linestyle='--', label='Micro (Scikit)', color='orange')
    ax.plot(scikit_iterations, scikit_train_macro_prec, marker='d', linestyle='--', label='Macro (Scikit)', color='brown')
    ax.set_title("Precision")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("Precision")
    ax.legend(fontsize='small')
    ax.grid(True)
    
    ax = axs[1,0]
    ax.plot(custom_iterations, custom_train_rec, marker='o', label='Recall', color='green')
    ax.plot(custom_iterations, custom_train_micro_rec, marker='^', label='Micro', color='lime')
    ax.plot(custom_iterations, custom_train_macro_rec, marker='s', label='Macro', color='darkgreen')
    ax.plot(scikit_iterations, scikit_train_rec, marker='x', linestyle='--', label='Recall (Scikit)', color='green')
    ax.plot(scikit_iterations, scikit_train_micro_rec, marker='v', linestyle='--', label='Micro (Scikit)', color='lime')
    ax.plot(scikit_iterations, scikit_train_macro_rec, marker='d', linestyle='--', label='Macro (Scikit)', color='darkgreen')
    ax.set_title("Recall")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("Recall")
    ax.legend(fontsize='small')
    ax.grid(True)
    
    ax = axs[1,1]
    ax.plot(custom_iterations, custom_train_f1, marker='o', label='F1', color='purple')
    ax.plot(custom_iterations, custom_train_micro_f1, marker='^', label='Micro', color='magenta')
    ax.plot(custom_iterations, custom_train_macro_f1, marker='s', label='Macro', color='violet')
    ax.plot(scikit_iterations, scikit_train_f1, marker='x', linestyle='--', label='F1 (Scikit)', color='purple')
    ax.plot(scikit_iterations, scikit_train_micro_f1, marker='v', linestyle='--', label='Micro (Scikit)', color='magenta')
    ax.plot(scikit_iterations, scikit_train_macro_f1, marker='d', linestyle='--', label='Macro (Scikit)', color='violet')
    ax.set_title("F1")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("F1")
    ax.legend(fontsize='small')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(14,10))
    fig.suptitle("Validation Metrics: Custom vs. Scikit AdaBoost")
    ax = axs[0,0]
    ax.plot(custom_iterations, custom_val_acc, marker='o', label='Custom', color='blue')
    ax.plot(scikit_iterations, scikit_val_acc, marker='x', linestyle='--', label='Scikit', color='blue')
    ax.set_title("Accuracy")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)
    
    ax = axs[0,1]
    ax.plot(custom_iterations, custom_val_prec, marker='o', label='Precision', color='red')
    ax.plot(custom_iterations, custom_val_micro_prec, marker='^', label='Micro', color='orange')
    ax.plot(custom_iterations, custom_val_macro_prec, marker='s', label='Macro', color='brown')
    ax.plot(scikit_iterations, scikit_val_prec, marker='x', linestyle='--', label='Precision (Scikit)', color='red')
    ax.plot(scikit_iterations, scikit_val_micro_prec, marker='v', linestyle='--', label='Micro (Scikit)', color='orange')
    ax.plot(scikit_iterations, scikit_val_macro_prec, marker='d', linestyle='--', label='Macro (Scikit)', color='brown')
    ax.set_title("Precision")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("Precision")
    ax.legend(fontsize='small')
    ax.grid(True)
    
    ax = axs[1,0]
    ax.plot(custom_iterations, custom_val_rec, marker='o', label='Recall', color='green')
    ax.plot(custom_iterations, custom_val_micro_rec, marker='^', label='Micro', color='lime')
    ax.plot(custom_iterations, custom_val_macro_rec, marker='s', label='Macro', color='darkgreen')
    ax.plot(scikit_iterations, scikit_val_rec, marker='x', linestyle='--', label='Recall (Scikit)', color='green')
    ax.plot(scikit_iterations, scikit_val_micro_rec, marker='v', linestyle='--', label='Micro (Scikit)', color='lime')
    ax.plot(scikit_iterations, scikit_val_macro_rec, marker='d', linestyle='--', label='Macro (Scikit)', color='darkgreen')
    ax.set_title("Recall")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("Recall")
    ax.legend(fontsize='small')
    ax.grid(True)
    
    ax = axs[1,1]
    ax.plot(custom_iterations, custom_val_f1, marker='o', label='F1', color='purple')
    ax.plot(custom_iterations, custom_val_micro_f1, marker='^', label='Micro', color='magenta')
    ax.plot(custom_iterations, custom_val_macro_f1, marker='s', label='Macro', color='violet')
    ax.plot(scikit_iterations, scikit_val_f1, marker='x', linestyle='--', label='F1 (Scikit)', color='purple')
    ax.plot(scikit_iterations, scikit_val_micro_f1, marker='v', linestyle='--', label='Micro (Scikit)', color='magenta')
    ax.plot(scikit_iterations, scikit_val_macro_f1, marker='d', linestyle='--', label='Macro (Scikit)', color='violet')
    ax.set_title("F1")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("F1")
    ax.legend(fontsize='small')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_logregr_comparison_metrics(custom_iteration_data, scikit_iteration_data):

    custom_iteration_data = sorted(custom_iteration_data, key=lambda d: d["iteration"])
    c_iters = []
    # Train
    c_train_acc  = []
    c_train_prec = []
    c_train_rec  = []
    c_train_f1   = []
    
    c_train_mi_prec = []
    c_train_mi_rec  = []
    c_train_mi_f1   = []
    c_train_ma_prec = []
    c_train_ma_rec  = []
    c_train_ma_f1   = []

    # Val
    c_val_acc  = []
    c_val_prec = []
    c_val_rec  = []
    c_val_f1   = []
    c_val_mi_prec = []
    c_val_mi_rec  = []
    c_val_mi_f1   = []
    c_val_ma_prec = []
    c_val_ma_rec  = []
    c_val_ma_f1   = []

    for entry in custom_iteration_data:
        i = entry["iteration"]
        c_iters.append(i)

        train_acc = entry["train_accuracy"]
        train_mat = entry["train_matrix"]

        c_train_acc.append(train_acc)
        # parse class=1
        c_train_prec.append(float(train_mat[2,1]))  # row=2 => Class 1, col=1 => Precision
        c_train_rec.append(float(train_mat[2,2]))
        c_train_f1.append(float(train_mat[2,3]))
        # parse micro => row=3
        c_train_mi_prec.append(float(train_mat[3,1]))
        c_train_mi_rec.append(float(train_mat[3,2]))
        c_train_mi_f1.append(float(train_mat[3,3]))
        # parse macro => row=4
        c_train_ma_prec.append(float(train_mat[4,1]))
        c_train_ma_rec.append(float(train_mat[4,2]))
        c_train_ma_f1.append(float(train_mat[4,3]))

        # Val
        val_acc = entry["val_accuracy"]
        val_mat = entry["val_matrix"]
        c_val_acc.append(val_acc)
        c_val_prec.append(float(val_mat[2,1]))
        c_val_rec.append(float(val_mat[2,2]))
        c_val_f1.append(float(val_mat[2,3]))
        c_val_mi_prec.append(float(val_mat[3,1]))
        c_val_mi_rec.append(float(val_mat[3,2]))
        c_val_mi_f1.append(float(val_mat[3,3]))
        c_val_ma_prec.append(float(val_mat[4,1]))
        c_val_ma_rec.append(float(val_mat[4,2]))
        c_val_ma_f1.append(float(val_mat[4,3]))

    # 2) Parse SCIKIT data
    scikit_iteration_data = sorted(scikit_iteration_data, key=lambda d: d["iteration"])
    s_iters = []
    s_train_acc  = []
    s_train_prec = []
    s_train_rec  = []
    s_train_f1   = []
    s_train_mi_prec = []
    s_train_mi_rec  = []
    s_train_mi_f1   = []
    s_train_ma_prec = []
    s_train_ma_rec  = []
    s_train_ma_f1   = []

    s_val_acc  = []
    s_val_prec = []
    s_val_rec  = []
    s_val_f1   = []
    s_val_mi_prec = []
    s_val_mi_rec  = []
    s_val_mi_f1   = []
    s_val_ma_prec = []
    s_val_ma_rec  = []
    s_val_ma_f1   = []

    for entry in scikit_iteration_data:
        i = entry["iteration"]
        s_iters.append(i)

        # Train
        train_acc = entry["train_accuracy"]
        train_mat = entry["train_matrix"]
        s_train_acc.append(train_acc)
        s_train_prec.append(float(train_mat[2,1]))
        s_train_rec.append(float(train_mat[2,2]))
        s_train_f1.append(float(train_mat[2,3]))
        s_train_mi_prec.append(float(train_mat[3,1]))
        s_train_mi_rec.append(float(train_mat[3,2]))
        s_train_mi_f1.append(float(train_mat[3,3]))
        s_train_ma_prec.append(float(train_mat[4,1]))
        s_train_ma_rec.append(float(train_mat[4,2]))
        s_train_ma_f1.append(float(train_mat[4,3]))

        # Val
        val_acc = entry["val_accuracy"]
        val_mat = entry["val_matrix"]
        s_val_acc.append(val_acc)
        s_val_prec.append(float(val_mat[2,1]))
        s_val_rec.append(float(val_mat[2,2]))
        s_val_f1.append(float(val_mat[2,3]))
        s_val_mi_prec.append(float(val_mat[3,1]))
        s_val_mi_rec.append(float(val_mat[3,2]))
        s_val_mi_f1.append(float(val_mat[3,3]))
        s_val_ma_prec.append(float(val_mat[4,1]))
        s_val_ma_rec.append(float(val_mat[4,2]))
        s_val_ma_f1.append(float(val_mat[4,3]))


    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("TRAIN Metrics: Custom vs. Scikit Logistic Regression")

    # Accuracy
    ax = axs[0,0]
    ax.plot(c_iters, c_train_acc, marker='o', label='Custom', color='blue')
    ax.plot(s_iters, s_train_acc, marker='x', linestyle='--', label='Scikit', color='blue')
    ax.set_title("Accuracy")
    ax.set_xlabel("Iteration (maybe 100,200,... )")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)

    # Precision: class=1, micro, macro
    ax = axs[0,1]
    ax.plot(c_iters, c_train_prec, marker='o', label='Class=1', color='red')
    ax.plot(c_iters, c_train_mi_prec, marker='^', label='Micro', color='orange')
    ax.plot(c_iters, c_train_ma_prec, marker='s', label='Macro', color='brown')
    ax.plot(s_iters, s_train_prec, marker='x', linestyle='--', label='Class=1 (Scikit)', color='red')
    ax.plot(s_iters, s_train_mi_prec, marker='v', linestyle='--', label='Micro (Scikit)', color='orange')
    ax.plot(s_iters, s_train_ma_prec, marker='d', linestyle='--', label='Macro (Scikit)', color='brown')
    ax.set_title("Precision")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Precision")
    ax.legend(fontsize='small')
    ax.grid(True)

    # Recall
    ax = axs[1,0]
    ax.plot(c_iters, c_train_rec, marker='o', label='Class=1', color='green')
    ax.plot(c_iters, c_train_mi_rec, marker='^', label='Micro', color='lime')
    ax.plot(c_iters, c_train_ma_rec, marker='s', label='Macro', color='darkgreen')
    ax.plot(s_iters, s_train_rec, marker='x', linestyle='--', label='Class=1 (Scikit)', color='green')
    ax.plot(s_iters, s_train_mi_rec, marker='v', linestyle='--', label='Micro (Scikit)', color='lime')
    ax.plot(s_iters, s_train_ma_rec, marker='d', linestyle='--', label='Macro (Scikit)', color='darkgreen')
    ax.set_title("Recall")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Recall")
    ax.legend(fontsize='small')
    ax.grid(True)

    # F1
    ax = axs[1,1]
    ax.plot(c_iters, c_train_f1, marker='o', label='Class=1', color='purple')
    ax.plot(c_iters, c_train_mi_f1, marker='^', label='Micro', color='magenta')
    ax.plot(c_iters, c_train_ma_f1, marker='s', label='Macro', color='violet')
    ax.plot(s_iters, s_train_f1, marker='x', linestyle='--', label='Class=1 (Scikit)', color='purple')
    ax.plot(s_iters, s_train_mi_f1, marker='v', linestyle='--', label='Micro (Scikit)', color='magenta')
    ax.plot(s_iters, s_train_ma_f1, marker='d', linestyle='--', label='Macro (Scikit)', color='violet')
    ax.set_title("F1")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("F1")
    ax.legend(fontsize='small')
    ax.grid(True)

    plt.tight_layout()
    plt.show()

    # ====== VAL =======
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("VAL Metrics: Custom vs. Scikit Logistic Regression")

    # Accuracy
    ax = axs[0,0]
    ax.plot(c_iters, c_val_acc, marker='o', label='Custom', color='blue')
    ax.plot(s_iters, s_val_acc, marker='x', linestyle='--', label='Scikit', color='blue')
    ax.set_title("Accuracy")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)

    # Precision
    ax = axs[0,1]
    ax.plot(c_iters, c_val_prec, marker='o', label='Class=1', color='red')
    ax.plot(c_iters, c_val_mi_prec, marker='^', label='Micro', color='orange')
    ax.plot(c_iters, c_val_ma_prec, marker='s', label='Macro', color='brown')
    ax.plot(s_iters, s_val_prec, marker='x', linestyle='--', label='Class=1 (Scikit)', color='red')
    ax.plot(s_iters, s_val_mi_prec, marker='v', linestyle='--', label='Micro (Scikit)', color='orange')
    ax.plot(s_iters, s_val_ma_prec, marker='d', linestyle='--', label='Macro (Scikit)', color='brown')
    ax.set_title("Precision")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Precision")
    ax.legend(fontsize='small')
    ax.grid(True)

    # Recall
    ax = axs[1,0]
    ax.plot(c_iters, c_val_rec, marker='o', label='Class=1', color='green')
    ax.plot(c_iters, c_val_mi_rec, marker='^', label='Micro', color='lime')
    ax.plot(c_iters, c_val_ma_rec, marker='s', label='Macro', color='darkgreen')
    ax.plot(s_iters, s_val_rec, marker='x', linestyle='--', label='Class=1 (Scikit)', color='green')
    ax.plot(s_iters, s_val_mi_rec, marker='v', linestyle='--', label='Micro (Scikit)', color='lime')
    ax.plot(s_iters, s_val_ma_rec, marker='d', linestyle='--', label='Macro (Scikit)', color='darkgreen')
    ax.set_title("Recall")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Recall")
    ax.legend(fontsize='small')
    ax.grid(True)

    # F1
    ax = axs[1,1]
    ax.plot(c_iters, c_val_f1, marker='o', label='Class=1', color='purple')
    ax.plot(c_iters, c_val_mi_f1, marker='^', label='Micro', color='magenta')
    ax.plot(c_iters, c_val_ma_f1, marker='s', label='Macro', color='violet')
    ax.plot(s_iters, s_val_f1, marker='x', linestyle='--', label='Class=1 (Scikit)', color='purple')
    ax.plot(s_iters, s_val_mi_f1, marker='v', linestyle='--', label='Micro (Scikit)', color='magenta')
    ax.plot(s_iters, s_val_ma_f1, marker='d', linestyle='--', label='Macro (Scikit)', color='violet')
    ax.set_title("F1")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("F1")
    ax.legend(fontsize='small')
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def runMyLogisticRegression(logRegr,
                            X_train, y_train,
                            X_val,   y_val,
                            X_test,  y_test,
                            batch_size=256):
    print("Training Logistic Regression (Custom)...")
    
    iteration_data = logRegr.fit(X_train, y_train, X_val, y_val, batch_size=batch_size)

    y_pred_train = logRegr.predict(X_train)
    y_pred_val   = logRegr.predict(X_val)
    y_pred_test  = logRegr.predict(X_test)

    print("\n==== Final metrics table for CUSTOM Logistic Regression ====")

    _print_table_iteration("TEST", y_test, y_pred_test)


    return y_pred_train, y_pred_val, y_pred_test, iteration_data



def runScikitLogRegr(X_train, y_train,
                     X_val,   y_val,
                     X_test,  y_test,
                     max_iter=1000, step=100):

    from sklearn.linear_model import LogisticRegression as ScikitLogReg

    iters_list = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1_scores = []
    test_micro_precisions = []
    test_micro_recalls = []
    test_macro_precisions = []
    test_macro_recalls = []
    
    y_pred_test_final = None

    scikit_iteration_data = []

    for i in range(step, max_iter + 1, step):
        sk_model = ScikitLogReg(max_iter=i, solver='saga', penalty='l2')
        sk_model.fit(X_train, y_train)

        # Predictions on TRAIN
        y_pred_train = sk_model.predict(X_train)
        print(f"\n=== ScikitLogRegr max_iter={i} ===")
        _print_table_iteration("TRAIN", y_train, y_pred_train)

        # Predictions on VAL
        y_pred_val   = sk_model.predict(X_val)
        _print_table_iteration("VAL",   y_val,   y_pred_val)

        # Predictions on TEST
        y_pred_test  = sk_model.predict(X_test)

        # Compute train metrics for iteration_data
        train_acc = compute_accuracy(y_train, y_pred_train)
        train_matrix = compute_metrics_matrix(y_train, y_pred_train)

        # Compute val metrics for iteration_data
        val_acc = compute_accuracy(y_val, y_pred_val)
        val_matrix = compute_metrics_matrix(y_val, y_pred_val)

        # Compute test metrics (used for final plotting arrays AND iteration_data)
        acc_test = compute_accuracy(y_test, y_pred_test)
        p_test, r_test, f1_test = compute_metrics(y_test, y_pred_test)
        test_matrix = compute_metrics_matrix(y_test, y_pred_test)

        # Collect partial-model test metrics for final arrays
        iters_list.append(i)
        test_accuracies.append(acc_test)
        test_precisions.append(p_test)
        test_recalls.append(r_test)
        test_f1_scores.append(f1_test)

        # row 3 = Micro, row 4 = Macro (columns => [Category,Prec,Recall,F1])
        micro_p_test = float(test_matrix[3, 1])
        micro_r_test = float(test_matrix[3, 2])
        macro_p_test = float(test_matrix[4, 1])
        macro_r_test = float(test_matrix[4, 2])

        test_micro_precisions.append(micro_p_test)
        test_micro_recalls.append(micro_r_test)
        test_macro_precisions.append(macro_p_test)
        test_macro_recalls.append(macro_r_test)

        # Keep track of final iteration's test predictions
        y_pred_test_final = y_pred_test

        # Also store everything in iteration_data
        scikit_iteration_data.append({
            "iteration": i,
            "train_accuracy": train_acc,
            "train_matrix":  train_matrix,
            "val_accuracy":   val_acc,
            "val_matrix":    val_matrix,
            "test_accuracy": acc_test,
            "test_matrix":   test_matrix
        })

    # Print final fancy table for test set
    if y_pred_test_final is not None:
        print("\n==== Final metrics table for SCIKIT Logistic Regression ====")
        _print_table_iteration("TEST", y_test, y_pred_test_final)

    return (
        iters_list,
        test_accuracies, test_precisions, test_recalls, test_f1_scores,
        test_micro_precisions, test_micro_recalls,
        test_macro_precisions, test_macro_recalls,
        y_pred_test_final
    ), scikit_iteration_data


def _print_table_iteration(set_name, y_true, y_pred):
    """
    Prints the fancy table of metrics (class 0, class 1, micro, macro)
    plus accuracy for the given (y_true, y_pred).
    """
    acc = compute_accuracy(y_true, y_pred)
    matrix = compute_metrics_matrix(y_true, y_pred)

    print(f"==== Metrics on {set_name} set ====")
    print(f"Accuracy = {acc:.4f}")
    print(f"Model Evaluation Metrics ({set_name} Set):")
    print("+" + "-" * 50 + "+")
    for row in matrix:
        print("| {:<12} | {:<12} | {:<12} | {:<12} |".format(*row))
    print("+" + "-" * 50 + "+\n")

def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def compute_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

def compute_metrics_matrix(y_true, y_pred):
    TP1 = np.sum((y_true == 1) & (y_pred == 1))
    FP1 = np.sum((y_true == 0) & (y_pred == 1))
    FN1 = np.sum((y_true == 1) & (y_pred == 0))
    precision_1 = TP1 / (TP1 + FP1) if (TP1 + FP1) > 0 else 0
    recall_1 = TP1 / (TP1 + FN1) if (TP1 + FN1) > 0 else 0
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0

    TP0 = np.sum((y_true == 0) & (y_pred == 0))
    FP0 = np.sum((y_true == 1) & (y_pred == 0))
    FN0 = np.sum((y_true == 0) & (y_pred == 1))
    precision_0 = TP0 / (TP0 + FP0) if (TP0 + FP0) > 0 else 0
    recall_0 = TP0 / (TP0 + FN0) if (TP0 + FN0) > 0 else 0
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0

    TP = TP1 + TP0
    FP = FP1 + FP0
    FN = FN1 + FN0
    precision_micro = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_micro = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0

    precision_macro = (precision_1 + precision_0) / 2
    recall_macro = (recall_1 + recall_0) / 2
    f1_macro = (f1_1 + f1_0) / 2

    return np.array([
        ["Category",      "Precision",       "Recall",         "F1-Score"],
        ["Class 0",       round(precision_0, 4), round(recall_0, 4), round(f1_0, 4)],
        ["Class 1",       round(precision_1, 4), round(recall_1, 4), round(f1_1, 4)],
        ["Micro-Average", round(precision_micro,4), round(recall_micro,4), round(f1_micro,4)],
        ["Macro-Average", round(precision_macro,4), round(recall_macro,4), round(f1_macro,4)],
    ])

def compute_and_print_metrics(y_true, y_pred, dataset_description="Data", model_name="Model"):
    """
    Computes and prints accuracy and a detailed metrics table for a given dataset.
    dataset_description: A string like "TRAINING data", "VALIDATION data", "TEST data".
    model_name: The name of the model being evaluated.
    """
    # The _print_table_iteration function uses its first argument as the set_name in its output.
    # The model_name here provides context for the overarching print statement.
    print(f"\n==== Metrics for {model_name} on {dataset_description} ====")
    _print_table_iteration(dataset_description, y_true, y_pred)



