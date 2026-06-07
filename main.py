import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn_lvq import GlvqModel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from timeit import default_timer as timer

#Wersja Python: 3.10.11
#Komendy do utworzerzenia środowiska i instalacji bibliotek:
# python -m venv .venv
# .venv\Scripts\activate
# pip install -r requirements.txt

# ==========================================
# 1. POBRANIE I PRZYGOTOWANIE DANYCH
# ==========================================
def load_and_prepare_data():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/echocardiogram/echocardiogram.data"
    filename = "echocardiogram.data"
    
    if not os.path.exists(filename):
        print("Pobieranie zbioru danych...")
        urllib.request.urlretrieve(url, filename)

    print("Czyszczenie i standaryzacja danych...")
    df = pd.read_csv(filename, header=None, na_values="?", on_bad_lines='skip')
    df = df.select_dtypes(include=[np.number])
    
    kolumny_cech = df.columns[:-1]
    df = df.dropna(subset=kolumny_cech)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    y = np.nan_to_num(y, nan=-1).astype(int)
    
    counts = pd.Series(y).value_counts()
    valid_classes = counts[counts >= 10].index
    mask = np.isin(y, valid_classes)
    X, y = X[mask], y[mask]

    return X, y

# ==========================================
# 2. FUNKCJA DO WALIDACJI KRZYŻOWEJ (CV)
# ==========================================
def train_cv(X, y, prototypes=1, gtol=1e-5, max_iter=100, random_state=42, beta=None, noise_level=None):
    CVN = 10
    skfold = StratifiedKFold(n_splits=CVN, shuffle=True, random_state=42) 
    PK_vec = np.zeros(CVN)
    scaler = StandardScaler()

    for i, (train_idx, test_idx) in enumerate(skfold.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

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
# 3. GENEROWANIE WYKRESÓW 3D (Bez zatrzymywania)
# ==========================================
def plot_3d(x_vec, y_vec, z_matrix, x_label, y_label, filename):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    X_grid, Y_grid = np.meshgrid(x_vec, y_vec)
    ax.plot_surface(X_grid, Y_grid, z_matrix.T, cmap='viridis')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel('Poprawnosc Klasyfikacji (PK) [%]')
    ax.view_init(30, 200)
    plt.savefig(filename, bbox_inches='tight')
    plt.close() # Zamknięcie figury zwalnia pamięć i nie pauzuje skryptu
    print(f"Zapisano wykres 3D: {filename}")

# ==========================================
# 4. EKSPERYMENTY I PODSUMOWANIE 2D
# ==========================================
def run_experiments():
    X, y = load_and_prepare_data()
    
    prototypes_vec = np.array([1, 2, 3, 4, 5])
    max_iter_vec = np.array([1, 10, 20, 50, 75, 100, 150, 300])
    gtol_vec = np.array([1e-3, 1e-4, 1e-5, 1e-6])
    gtol_labels = [3, 4, 5, 6] 
    random_state_vec = np.array([0, 42, 123, 500, 999])
    noise_vec = np.array([0.0, 0.2, 0.5, 1.0, 2.0])
    beta_vec = np.array([1, 2, 3, 5, 10])

    # Słowniki do przechowywania najlepszych wyników do podsumowania 2D
    best_results_max_iter = {}
    best_results_prototypes = {}

    # --- EKSPERYMENT 1: prototypes_per_class vs max_iter ---
    print("\n--- Exp 1: prototypes_per_class vs max_iter ---")
    start = timer()
    PK_2D_Exp1 = np.zeros([len(prototypes_vec), len(max_iter_vec)])
    for p_ind in range(len(prototypes_vec)):
        for m_ind in range(len(max_iter_vec)):
            PK = train_cv(X, y, prototypes=prototypes_vec[p_ind], max_iter=max_iter_vec[m_ind])
            PK_2D_Exp1[p_ind, m_ind] = PK
    print(f"Czas Exp 1: {timer()-start:.2f} s")
    plot_3d(prototypes_vec, max_iter_vec, PK_2D_Exp1, 'Prototypes per class', 'Max Iterations', 'Fig1_3D_prototypes_maxiter.png')
    
    # Wyciągamy najwyższe PK dla każdej iteracji (niezależnie od ilości prototypów)
    best_results_max_iter['Prototypes (Exp1)'] = np.max(PK_2D_Exp1, axis=0)
    best_results_prototypes['Max Iter (Exp1)'] = np.max(PK_2D_Exp1, axis=1)

    # --- EKSPERYMENT 2: gtol vs max_iter ---
    print("\n--- Exp 2: gtol vs max_iter ---")
    start = timer()
    PK_2D_Exp2 = np.zeros([len(gtol_vec), len(max_iter_vec)])
    for g_ind in range(len(gtol_vec)):
        for m_ind in range(len(max_iter_vec)):
            PK = train_cv(X, y, prototypes=2, gtol=gtol_vec[g_ind], max_iter=max_iter_vec[m_ind])
            PK_2D_Exp2[g_ind, m_ind] = PK
    print(f"Czas Exp 2: {timer()-start:.2f} s")
    plot_3d(gtol_labels, max_iter_vec, PK_2D_Exp2, '-log10(gtol)', 'Max Iterations', 'Fig2_3D_gtol_maxiter.png')
    
    best_results_max_iter['Gtol (Exp2)'] = np.max(PK_2D_Exp2, axis=0)

    # --- EKSPERYMENT 3: random_state vs prototypes_per_class ---
    print("\n--- Exp 3: random_state vs prototypes_per_class ---")
    start = timer()
    PK_2D_Exp3 = np.zeros([len(prototypes_vec), len(random_state_vec)])
    for p_ind in range(len(prototypes_vec)):
        for r_ind in range(len(random_state_vec)):
            PK = train_cv(X, y, prototypes=prototypes_vec[p_ind], random_state=random_state_vec[r_ind])
            PK_2D_Exp3[p_ind, r_ind] = PK
    print(f"Czas Exp 3: {timer()-start:.2f} s")
    plot_3d(prototypes_vec, random_state_vec, PK_2D_Exp3, 'Prototypes per class', 'Random State', 'Fig3_3D_prototypes_randomstate.png')
    
    best_results_prototypes['Random State (Exp3)'] = np.max(PK_2D_Exp3, axis=1)

    # --- EKSPERYMENT 4: initial_prototypes (noise) vs max_iter ---
    print("\n--- Exp 4: init_protos vs max_iter ---")
    start = timer()
    PK_2D_Exp4 = np.zeros([len(noise_vec), len(max_iter_vec)])
    for n_ind in range(len(noise_vec)):
        for m_ind in range(len(max_iter_vec)):
            PK = train_cv(X, y, prototypes=2, noise_level=noise_vec[n_ind], max_iter=max_iter_vec[m_ind])
            PK_2D_Exp4[n_ind, m_ind] = PK
    print(f"Czas Exp 4: {timer()-start:.2f} s")
    plot_3d(noise_vec, max_iter_vec, PK_2D_Exp4, 'Init Noise (std)', 'Max Iterations', 'Fig4_3D_initialprototypes_maxiter.png')

    best_results_max_iter['Init Noise (Exp4)'] = np.max(PK_2D_Exp4, axis=0)

    # --- EKSPERYMENT 5: beta vs max_iter ---
    print("\n--- Exp 5: beta vs max_iter ---")
    start = timer()
    PK_2D_Exp5 = np.zeros([len(beta_vec), len(max_iter_vec)])
    for b_ind in range(len(beta_vec)):
        for m_ind in range(len(max_iter_vec)):
            PK = train_cv(X, y, prototypes=2, beta=beta_vec[b_ind], max_iter=max_iter_vec[m_ind])
            PK_2D_Exp5[b_ind, m_ind] = PK
    print(f"Czas Exp 5: {timer()-start:.2f} s")
    plot_3d(beta_vec, max_iter_vec, PK_2D_Exp5, 'Beta', 'Max Iterations', 'Fig5_3D_beta_maxiter.png')
    
    best_results_max_iter['Beta (Exp5)'] = np.max(PK_2D_Exp5, axis=0)

    # --- EKSPERYMENT 6: beta vs gtol ---
    print("\n--- Exp 6: beta vs gtol ---")
    start = timer()
    PK_2D_Exp6 = np.zeros([len(beta_vec), len(gtol_vec)])
    for b_ind in range(len(beta_vec)):
        for g_ind in range(len(gtol_vec)):
            PK = train_cv(X, y, prototypes=2, beta=beta_vec[b_ind], gtol=gtol_vec[g_ind])
            PK_2D_Exp6[b_ind, g_ind] = PK
    print(f"Czas Exp 6: {timer()-start:.2f} s")
    plot_3d(beta_vec, gtol_labels, PK_2D_Exp6, 'Beta', '-log10(gtol)', 'Fig6_3D_beta_gtol.png')

    # ==========================================
    # PODSUMOWANIE 2D
    # ==========================================
    print("\nTworzenie wykresów podsumowujących 2D...")

    # Wykres Podsumowujący A: Wpływ parametrów na maksymalne PK w funkcji max_iter
    plt.figure(figsize=(10, 6))
    for label, pk_values in best_results_max_iter.items():
        plt.plot(max_iter_vec, pk_values, marker='o', linewidth=2, label=label)
    plt.xlabel('Max Iterations (max_iter)', fontsize=12)
    plt.ylabel('Maksymalna Poprawność Klasyfikacji (PK) [%]', fontsize=12)
    plt.title('Porównanie maksymalnego PK w funkcji max_iter dla badanych parametrów', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('Summary_2D_MaxIter.png', bbox_inches='tight')
    plt.show()

    # Wykres Podsumowujący B: Wpływ parametrów na maksymalne PK w funkcji liczby prototypów
    plt.figure(figsize=(10, 6))
    for label, pk_values in best_results_prototypes.items():
        plt.plot(prototypes_vec, pk_values, marker='s', linewidth=2, label=label)
    plt.xlabel('Liczba Prototypów (prototypes_per_class)', fontsize=12)
    plt.ylabel('Maksymalna Poprawność Klasyfikacji (PK) [%]', fontsize=12)
    plt.title('Porównanie maksymalnego PK w funkcji liczby prototypów', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(prototypes_vec)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('Summary_2D_Prototypes.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_experiments()