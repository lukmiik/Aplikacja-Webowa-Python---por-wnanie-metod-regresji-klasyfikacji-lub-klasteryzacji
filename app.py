import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

st.set_page_config(
    page_title="Porównanie Metod ML",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Aplikacja Webowa - Porównanie Metod Machine Learning")
st.markdown("Autorzy: Łukasz Łukaszewski, Paweł Kwieciński")

st.sidebar.header("Konfiguracja")
analysis_type = st.sidebar.selectbox(
    "Wybierz typ analizy:",
    ["Klasyfikacja", "Regresja", "Klasteryzacja"]
)

@st.cache_data
def generate_synthetic_data(data_type, n_samples=1000):
    """
    Generuje syntetyczne dane dla różnych typów problemów ML
    Źródło: scikit-learn documentation
    """
    if data_type == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=3,
            random_state=42
        )
        return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])]), pd.Series(y, name='target')

    elif data_type == "regression":
        X, y = make_regression(
            n_samples=n_samples,
            n_features=8,
            n_informative=5,
            noise=0.1,
            random_state=42
        )
        return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])]), pd.Series(y, name='target')

    elif data_type == "clustering":
        X, y = make_blobs(
            n_samples=n_samples,
            centers=4,
            n_features=5,
            cluster_std=1.0,
            random_state=42
        )
        return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])]), pd.Series(y, name='true_cluster')

def introduce_missing_values(df, missing_ratio=0.1):
    """
    Wprowadza losowe braki w danych
    """
    df_missing = df.copy()
    n_missing = int(len(df) * missing_ratio)

    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            missing_idx = np.random.choice(df.index, size=n_missing, replace=False)
            df_missing.loc[missing_idx, col] = np.nan

    return df_missing

def clean_data(df, unique_key=""):
    """
    Czyści dane - uzupełnia braki metodami statystycznymi
    """
    st.subheader("🧹 Czyszczenie danych")

    missing_count = df.isnull().sum().sum()
    st.write(f"Znaleziono {missing_count} brakujących wartości")

    if missing_count > 0:
        impute_method = st.selectbox(
            "Wybierz metodę uzupełniania braków:",
            ["mean", "median", "most_frequent", "forward_fill"],
            key=f"impute_method_{unique_key}"  # Unikalny klucz
        )

        df_cleaned = df.copy()

        if impute_method == "forward_fill":
            df_cleaned = df_cleaned.fillna(method='ffill').fillna(method='bfill')
        else:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                imputer = SimpleImputer(strategy=impute_method)
                df_cleaned[numeric_columns] = imputer.fit_transform(df_cleaned[numeric_columns])

        st.success(f"Dane zostały wyczyszczone metodą: {impute_method}")
        return df_cleaned
    else:
        st.success("Brak braków w danych!")
        return df

def classification_analysis():
    st.header("📊 Analiza Klasyfikacji")

    datasets = {}
    for i in range(3):
        st.subheader(f"Zbiór danych {i+1}")
        n_samples = st.slider(f"Liczba próbek (zbiór {i+1}):", 500, 2000, 1000, key=f"samples_{i}")
        X, y = generate_synthetic_data("classification", n_samples)

        X_missing = introduce_missing_values(X, 0.15)
        st.write("Dane przed czyszczeniem:")
        st.write(f"Braki w danych: {X_missing.isnull().sum().sum()}")

        X_clean = clean_data(X_missing, unique_key=f"classification_{i}")
        datasets[f"dataset_{i+1}"] = (X_clean, y)

    classifiers = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
    }

    st.subheader("🔍 Porównanie Algorytmów")
    results = []

    for dataset_name, (X, y) in datasets.items():
        for clf_name, clf in classifiers.items():
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            results.append({
                'Zbiór danych': dataset_name,
                'Algorytm': clf_name,
                'Dokładność': accuracy
            })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    results_pivot = results_df.pivot(index='Algorytm', columns='Zbiór danych', values='Dokładność')
    sns.heatmap(results_pivot, annot=True, cmap='viridis', ax=ax)
    plt.title('Porównanie dokładności algorytmów klasyfikacji')
    st.pyplot(fig)

def regression_analysis():
    st.header("📈 Analiza Regresji")

    datasets = {}
    for i in range(3):
        st.subheader(f"Zbiór danych {i+1}")
        n_samples = st.slider(f"Liczba próbek (zbiór {i+1}):", 500, 2000, 1000, key=f"reg_samples_{i}")
        X, y = generate_synthetic_data("regression", n_samples)

        X_missing = introduce_missing_values(X, 0.1)
        st.write("Dane przed czyszczeniem:")
        st.write(f"Braki w danych: {X_missing.isnull().sum().sum()}")

        X_clean = clean_data(X_missing, unique_key=f"regression_{i}")
        datasets[f"dataset_{i+1}"] = (X_clean, y)

    regressors = {
        "Random Forest": RandomForestRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "SVR": SVR()
    }

    st.subheader("🔍 Porównanie Algorytmów")
    results = []

    for dataset_name, (X, y) in datasets.items():
        for reg_name, reg in regressors.items():
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            reg.fit(X_train_scaled, y_train)
            y_pred = reg.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)

            results.append({
                'Zbiór danych': dataset_name,
                'Algorytm': reg_name,
                'MSE': mse
            })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    results_pivot = results_df.pivot(index='Algorytm', columns='Zbiór danych', values='MSE')
    sns.heatmap(results_pivot, annot=True, cmap='viridis_r', ax=ax)
    plt.title('Porównanie MSE algorytmów regresji (niższe = lepsze)')
    st.pyplot(fig)

def clustering_analysis():
    st.header("🎯 Analiza Klasteryzacji")

    datasets = {}
    for i in range(3):
        st.subheader(f"Zbiór danych {i+1}")
        n_samples = st.slider(f"Liczba próbek (zbiór {i+1}):", 500, 2000, 1000, key=f"clust_samples_{i}")
        X, y_true = generate_synthetic_data("clustering", n_samples)

        X_missing = introduce_missing_values(X, 0.08)
        st.write("Dane przed czyszczeniem:")
        st.write(f"Braki w danych: {X_missing.isnull().sum().sum()}")

        X_clean = clean_data(X_missing, unique_key=f"clustering_{i}")
        datasets[f"dataset_{i+1}"] = (X_clean, y_true)

    clusterers = {
        "K-Means": KMeans(n_clusters=4, random_state=42),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
        "Agglomerative": AgglomerativeClustering(n_clusters=4)
    }

    st.subheader("🔍 Porównanie Algorytmów")
    results = []

    for dataset_name, (X, y_true) in datasets.items():
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for clust_name, clusterer in clusterers.items():
            cluster_labels = clusterer.fit_predict(X_scaled)

            if len(set(cluster_labels)) > 1:
                silhouette = silhouette_score(X_scaled, cluster_labels)
            else:
                silhouette = -1

            results.append({
                'Zbiór danych': dataset_name,
                'Algorytm': clust_name,
                'Silhouette Score': silhouette
            })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    results_pivot = results_df.pivot(index='Algorytm', columns='Zbiór danych', values='Silhouette Score')
    sns.heatmap(results_pivot, annot=True, cmap='viridis', ax=ax)
    plt.title('Porównanie Silhouette Score algorytmów klasteryzacji (wyższe = lepsze)')
    st.pyplot(fig)

if analysis_type == "Klasyfikacja":
    classification_analysis()
elif analysis_type == "Regresja":
    regression_analysis()
elif analysis_type == "Klasteryzacja":
    clustering_analysis()

st.markdown("---")
st.markdown("""
**Źródła wykorzystane w projekcie:**
- scikit-learn: Pedregosa et al., JMLR 12, pp. 2825-2830, 2011
- Streamlit: https://streamlit.io/
- Pandas: McKinney, W. (2010). Data Structures for Statistical Computing in Python
- Matplotlib/Seaborn: Hunter, J. D. (2007). Matplotlib: A 2D graphics environment
""")
