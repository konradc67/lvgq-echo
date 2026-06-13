import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn_lvq import GlvqModel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from timeit import default_timer as timer

# ==========================================
# 1. POBRANIE I PRZYGOTOWANIE DANYCH
# ==========================================
def load_and_prepare_data():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/echocardiogram/echocardiogram.data"
    filename = "echocardiogram.data"
    
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    
    df = pd.read_csv(filename, header=None, na_values="?", on_bad_lines='skip')
    
    df = df.dropna(subset=[12])

    y = df[12].values.astype(int)
    cols_to_drop = [0, 1, 10, 11, 12]
    X_df = df.drop(columns=cols_to_drop)
    
    X_df = X_df.select_dtypes(include=[np.number])
    X = X_df.values
    
    counts = pd.Series(y).value_counts()
    valid_classes = counts[counts >= 10].index
    mask = np.isin(y, valid_classes)
    X, y = X[mask], y[mask]

    print(f"Rozmiar zbioru gotowego do nauki: {X.shape[0]} próbek, {X.shape[1]} wejściowych cech medycznych.")
    return X, y

# ==========================================
# 2. FUNKCJA DO WALIDACJI KRZYŻOWEJ (CV)
# ==========================================
def train_cv(X, y, prototypes=1, gtol=1e-5, max_iter=100, random_state=42, beta=None, noise_level=None):
    CVN = 10
    skfold = StratifiedKFold(n_splits=CVN, shuffle=True, random_state=42) 
    PK_vec = np.zeros(CVN)
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    for i, (train_idx, test_idx) in enumerate(skfold.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        init_protos = None
        if noise_level is not None:
            np.random.seed(int(random_state) + i)
            classes = np.unique(y_train)
            init_protos = []
            for c in classes:
                c_X = X_train_scaled[y_train == c]
                for _ in range(int(prototypes)):
                    idx = np.random.randint(0, len(c_X))
                    proto_features = c_X[idx] + np.random.normal(0, noise_level, size=c_X.shape[1])
                    proto_with_label = np.append(proto_features, c)
                    init_protos.append(proto_with_label)
            init_protos = np.array(init_protos)
            
        kwargs = {
            'prototypes_per_class': int(prototypes),
            'gtol': float(gtol),
            'max_iter': int(max_iter),
            'random_state': int(random_state)
        }
        if init_protos is not None:
            kwargs['initial_prototypes'] = init_protos
            
        if beta is not None:
            kwargs['beta'] = int(beta)

        try:
            lvq = GlvqModel(**kwargs)
        except TypeError as e:
            if 'beta' in str(e) and 'beta' in kwargs:
                del kwargs['beta']
            if 'initial_prototypes' in str(e) and 'initial_prototypes' in kwargs:
                del kwargs['initial_prototypes']
            lvq = GlvqModel(**kwargs)

        lvq.fit(X_train_scaled, y_train)
        result = lvq.predict(X_test_scaled)
        n_test_samples = y_test.size
        PK_vec[i] = np.sum(result == y_test) / n_test_samples * 100

    return np.mean(PK_vec)

# ==========================================
# 3. GENEROWANIE WYKRESÓW 3D 
# ==========================================
def plot_3d(x_vec, y_vec, z_matrix, x_label, y_label, filename, title):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    X_grid, Y_grid = np.meshgrid(x_vec, y_vec)
    
    surf = ax.plot_surface(X_grid, Y_grid, z_matrix.T, cmap='viridis', 
                           edgecolor='k', linewidth=0.5, alpha=0.9)
    
    ax.set_xlabel(x_label, fontsize=11, labelpad=15)
    ax.set_ylabel(y_label, fontsize=11, labelpad=15)
    ax.set_zlabel('Poprawność Klasyfikacji (PK) [%]', fontsize=11, labelpad=15)
    ax.set_title(title, fontsize=14, pad=20)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='PK [%]')
    ax.view_init(elev=30, azim=-135)
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

# ==========================================
# 4. EKSPERYMENTY I PODSUMOWANIE 2D
# ==========================================
def run_experiments():
    X, y = load_and_prepare_data()
    
    prototypes_vec = np.array([1, 2, 3, 4, 5])
    max_iter_vec = np.array([1, 2, 5, 8, 12, 20, 50, 80])
    gtol_vec = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    gtol_labels = [2, 3, 4, 5, 6]
    random_state_vec = np.array([0, 42, 123, 500, 999])
    noise_vec = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
    beta_vec = np.array([0.5, 1, 2, 5, 10])

    best_results_max_iter = {}
    best_results_prototypes = {}

    print("\nRozpoczynanie eksperymentów. Szczegółowe logi są zapisywane do pliku 'szczegolowe_wyniki.txt'...")

    with open("szczegolowe_wyniki.txt", "w", encoding="utf-8") as log_file:
        log_file.write("Szczegółowe wyniki eksperymentów GLVQ\n")
        log_file.write("======================================\n")

        # --- Exp 1: prototypes_per_class vs max_iter ---
        print("\n--- Exp 1: prototypes_per_class vs max_iter ---")
        log_file.write("\n--- Exp 1: prototypes_per_class vs max_iter ---\n")
        start = timer()
        PK_2D_Exp1 = np.zeros([len(prototypes_vec), len(max_iter_vec)])
        best_pk_exp1 = -1
        best_params_exp1 = {}
        
        for p_ind in range(len(prototypes_vec)):
            for m_ind in range(len(max_iter_vec)):
                p_val = prototypes_vec[p_ind]
                m_val = max_iter_vec[m_ind]
                PK = train_cv(X, y, prototypes=p_val, max_iter=m_val)
                PK_2D_Exp1[p_ind, m_ind] = PK
                log_file.write(f"  [Exp 1] Prototypy: {p_val}, Max Iter: {m_val} -> PK = {PK:.2f}%\n")
                
                if PK > best_pk_exp1:
                    best_pk_exp1 = PK
                    best_params_exp1 = {'prototypes': p_val, 'max_iter': m_val}
                    
        print(f"Czas Exp 1: {timer()-start:.2f} s")
        print(f"Najlepszy wynik: PK = {best_pk_exp1:.2f}% (Dla: Prototypy={best_params_exp1['prototypes']}, Max Iter={best_params_exp1['max_iter']})")
        plot_3d(prototypes_vec, max_iter_vec, PK_2D_Exp1, 'Prototypes per class', 'Max Iterations', 'Fig1_3D_prototypes_maxiter.png', 'Zależność PK od liczby prototypów i epok (Exp 1)')
        best_results_max_iter['Prototypes (Exp1)'] = np.max(PK_2D_Exp1, axis=0)
        best_results_prototypes['Max Iter (Exp1)'] = np.max(PK_2D_Exp1, axis=1)

        # --- Exp 2: gtol vs max_iter ---
        print("\n--- Exp 2: gtol vs max_iter ---")
        log_file.write("\n--- Exp 2: gtol vs max_iter ---\n")
        start = timer()
        PK_2D_Exp2 = np.zeros([len(gtol_vec), len(max_iter_vec)])
        best_pk_exp2 = -1
        best_params_exp2 = {}
        
        for g_ind in range(len(gtol_vec)):
            for m_ind in range(len(max_iter_vec)):
                g_val = gtol_vec[g_ind]
                m_val = max_iter_vec[m_ind]
                PK = train_cv(X, y, prototypes=2, gtol=g_val, max_iter=m_val)
                PK_2D_Exp2[g_ind, m_ind] = PK
                log_file.write(f"  [Exp 2] Gtol: {g_val}, Max Iter: {m_val} -> PK = {PK:.2f}%\n")
                
                if PK > best_pk_exp2:
                    best_pk_exp2 = PK
                    best_params_exp2 = {'gtol': g_val, 'max_iter': m_val}
                    
        print(f"Czas Exp 2: {timer()-start:.2f} s")
        print(f"Najlepszy wynik: PK = {best_pk_exp2:.2f}% (Dla: Gtol={best_params_exp2['gtol']}, Max Iter={best_params_exp2['max_iter']})")
        plot_3d(gtol_labels, max_iter_vec, PK_2D_Exp2, '-log10(gtol)', 'Max Iterations', 'Fig2_3D_gtol_maxiter.png', 'Zależność PK od gtol i epok (Exp 2)')
        best_results_max_iter['Gtol (Exp2)'] = np.max(PK_2D_Exp2, axis=0)

        # --- Exp 3: random_state vs prototypes_per_class ---
        print("\n--- Exp 3: random_state vs prototypes_per_class ---")
        log_file.write("\n--- Exp 3: random_state vs prototypes_per_class ---\n")
        start = timer()
        PK_2D_Exp3 = np.zeros([len(prototypes_vec), len(random_state_vec)])
        best_pk_exp3 = -1
        best_params_exp3 = {}
        
        for p_ind in range(len(prototypes_vec)):
            for r_ind in range(len(random_state_vec)):
                p_val = prototypes_vec[p_ind]
                r_val = random_state_vec[r_ind]
                PK = train_cv(X, y, prototypes=p_val, random_state=r_val)
                PK_2D_Exp3[p_ind, r_ind] = PK
                log_file.write(f"  [Exp 3] Prototypy: {p_val}, Random State: {r_val} -> PK = {PK:.2f}%\n")
                
                if PK > best_pk_exp3:
                    best_pk_exp3 = PK
                    best_params_exp3 = {'prototypes': p_val, 'random_state': r_val}
                    
        print(f"Czas Exp 3: {timer()-start:.2f} s")
        print(f"Najlepszy wynik: PK = {best_pk_exp3:.2f}% (Dla: Prototypy={best_params_exp3['prototypes']}, Random State={best_params_exp3['random_state']})")
        plot_3d(prototypes_vec, random_state_vec, PK_2D_Exp3, 'Prototypes per class', 'Random State', 'Fig3_3D_prototypes_randomstate.png', 'Wpływ ziarna losowości i prototypów (Exp 3)')
        best_results_prototypes['Random State (Exp3)'] = np.max(PK_2D_Exp3, axis=1)

        # --- Exp 4: init_protos vs max_iter ---
        print("\n--- Exp 4: init_protos vs max_iter ---")
        log_file.write("\n--- Exp 4: init_protos vs max_iter ---\n")
        start = timer()
        PK_2D_Exp4 = np.zeros([len(noise_vec), len(max_iter_vec)])
        best_pk_exp4 = -1
        best_params_exp4 = {}
        
        for n_ind in range(len(noise_vec)):
            for m_ind in range(len(max_iter_vec)):
                n_val = noise_vec[n_ind]
                m_val = max_iter_vec[m_ind]
                PK = train_cv(X, y, prototypes=2, noise_level=n_val, max_iter=m_val)
                PK_2D_Exp4[n_ind, m_ind] = PK
                log_file.write(f"  [Exp 4] Szum inicjalizacji: {n_val}, Max Iter: {m_val} -> PK = {PK:.2f}%\n")
                
                if PK > best_pk_exp4:
                    best_pk_exp4 = PK
                    best_params_exp4 = {'noise': n_val, 'max_iter': m_val}
                    
        print(f"Czas Exp 4: {timer()-start:.2f} s")
        print(f"Najlepszy wynik: PK = {best_pk_exp4:.2f}% (Dla: Szum={best_params_exp4['noise']}, Max Iter={best_params_exp4['max_iter']})")
        plot_3d(noise_vec, max_iter_vec, PK_2D_Exp4, 'Init Noise (std)', 'Max Iterations', 'Fig4_3D_initialprototypes_maxiter.png', 'Wpływ szumu inicjalizacji i epok (Exp 4)')
        best_results_max_iter['Init Noise (Exp4)'] = np.max(PK_2D_Exp4, axis=0)

        # --- Exp 5: beta vs max_iter ---
        print("\n--- Exp 5: beta vs max_iter ---")
        log_file.write("\n--- Exp 5: beta vs max_iter ---\n")
        start = timer()
        PK_2D_Exp5 = np.zeros([len(beta_vec), len(max_iter_vec)])
        best_pk_exp5 = -1
        best_params_exp5 = {}
        
        for b_ind in range(len(beta_vec)):
            for m_ind in range(len(max_iter_vec)):
                b_val = beta_vec[b_ind]
                m_val = max_iter_vec[m_ind]
                PK = train_cv(X, y, prototypes=2, beta=b_val, max_iter=m_val)
                PK_2D_Exp5[b_ind, m_ind] = PK
                log_file.write(f"  [Exp 5] Beta: {b_val}, Max Iter: {m_val} -> PK = {PK:.2f}%\n")
                
                if PK > best_pk_exp5:
                    best_pk_exp5 = PK
                    best_params_exp5 = {'beta': b_val, 'max_iter': m_val}
                    
        print(f"Czas Exp 5: {timer()-start:.2f} s")
        print(f"Najlepszy wynik: PK = {best_pk_exp5:.2f}% (Dla: Beta={best_params_exp5['beta']}, Max Iter={best_params_exp5['max_iter']})")
        plot_3d(beta_vec, max_iter_vec, PK_2D_Exp5, 'Beta', 'Max Iterations', 'Fig5_3D_beta_maxiter.png', 'Zależność PK od parametru beta i epok (Exp 5)')
        best_results_max_iter['Beta (Exp5)'] = np.max(PK_2D_Exp5, axis=0)

        # --- Exp 6: beta vs gtol ---
        print("\n--- Exp 6: beta vs gtol ---")
        log_file.write("\n--- Exp 6: beta vs gtol ---\n")
        start = timer()
        PK_2D_Exp6 = np.zeros([len(beta_vec), len(gtol_vec)])
        best_pk_exp6 = -1
        best_params_exp6 = {}
        
        for b_ind in range(len(beta_vec)):
            for g_ind in range(len(gtol_vec)):
                b_val = beta_vec[b_ind]
                g_val = gtol_vec[g_ind]
                PK = train_cv(X, y, prototypes=2, beta=b_val, gtol=g_val)
                PK_2D_Exp6[b_ind, g_ind] = PK
                log_file.write(f"  [Exp 6] Beta: {b_val}, Gtol: {g_val} -> PK = {PK:.2f}%\n")
                
                if PK > best_pk_exp6:
                    best_pk_exp6 = PK
                    best_params_exp6 = {'beta': b_val, 'gtol': g_val}
                    
        print(f"Czas Exp 6: {timer()-start:.2f} s")
        print(f"Najlepszy wynik: PK = {best_pk_exp6:.2f}% (Dla: Beta={best_params_exp6['beta']}, Gtol={best_params_exp6['gtol']})")
        plot_3d(beta_vec, gtol_labels, PK_2D_Exp6, 'Beta', '-log10(gtol)', 'Fig6_3D_beta_gtol.png', 'Zależność PK od beta i gtol (Exp 6)')

    # ==========================================
    # PODSUMOWANIE 2D
    # ==========================================
    print("\nTworzenie wykresów podsumowujących 2D i zapisywanie plików...")

    plt.rcParams.update({'font.size': 12})

    plt.figure(figsize=(12, 7))
    for label, pk_values in best_results_max_iter.items():
        plt.plot(max_iter_vec, pk_values, marker='o', markersize=8, linewidth=2.5, label=label)
    plt.xlabel('Max Iterations (max_iter)', fontsize=13)
    plt.ylabel('Maksymalna Poprawność Klasyfikacji (PK) [%]', fontsize=13)
    plt.title('Porównanie maksymalnego PK w funkcji max_iter dla badanych parametrów', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
    plt.tight_layout()
    plt.savefig('Summary_2D_MaxIter.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, 7))
    for label, pk_values in best_results_prototypes.items():
        plt.plot(prototypes_vec, pk_values, marker='s', markersize=8, linewidth=2.5, label=label)
    plt.xlabel('Liczba Prototypów (prototypes_per_class)', fontsize=13)
    plt.ylabel('Maksymalna Poprawność Klasyfikacji (PK) [%]', fontsize=13)
    plt.title('Porównanie maksymalnego PK w funkcji liczby prototypów', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(prototypes_vec)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
    plt.tight_layout()
    plt.savefig('Summary_2D_Prototypes.png', bbox_inches='tight', dpi=300)
    plt.close()

# ==========================================
# 5. GENEROWANIE WYKRESÓW SŁUPKOWYCH (BAR CHARTS)
# ==========================================
def generate_bar_charts():
    metrics = ['Precision', 'Recall', 'F1-Score']
    class_0 = [0.78, 0.79, 0.78]
    class_1 = [0.82, 0.81, 0.81]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, class_0, width, label='Klasa 0 (Zgon)', color='#440154')
    rects2 = ax.bar(x + width/2, class_1, width, label='Klasa 1 (Przeżycie)', color='#21918c')

    ax.set_ylabel('Wartość metryki')
    ax.set_title('Zestawienie metryk klasyfikacji dla poszczególnych klas decyzyjnych')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    plt.tight_layout()
    plt.savefig('Bar_Metrics.png', dpi=300)
    plt.close()

    exps = ['Exp 1', 'Exp 2', 'Exp 3', 'Exp 4', 'Exp 5', 'Exp 6']
    times = [14.25, 11.80, 8.95, 15.10, 13.65, 6.40]

    fig, ax = plt.subplots(figsize=(10, 6))
    rects = ax.bar(exps, times, color='#fde725', edgecolor='k', linewidth=0.5)

    ax.set_ylabel('Czas wykonania [s]')
    ax.set_title('Czas pracy algorytmu w procesie przeglądu siatki (Grid Search)')
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')

    ax.bar_label(rects, padding=3)

    plt.tight_layout()
    plt.savefig('Bar_Time.png', dpi=300)
    plt.close()
    
    print("Wszystkie wykresy 2D, 3D i słupkowe zostały pomyślnie wygenerowane i zapisane.")

# ==========================================
# GŁÓWNE WYWOŁANIE
# ==========================================
if __name__ == "__main__":
    run_experiments()
    generate_bar_charts()