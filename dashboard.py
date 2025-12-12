import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(
    page_title="Movie Rating Models Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎬 Movie Rating Prediction Models Dashboard")

# Load data and models
@st.cache_resource
def load_data_and_models():
    """Load the dataset and all cached models"""
    df = pd.read_csv("final_movie_table.csv")
    
    # Preprocess data
    df_nz = df[df['budget'] > 0].copy()
    df_nz['log_budget'] = np.log1p(df_nz['budget'])
    df_nz['log_revenue'] = np.log1p(df_nz['revenue'])
    df_nz['log_vote_count'] = np.log1p(df_nz['vote_count'])
    df_nz['log_user_rating_count'] = np.log1p(df_nz['user_rating_count'])
    df_nz['log_keyword_count'] = np.log1p(df_nz['keyword_count'])
    
    numeric_vars = [
        'vote_average',
        'log_budget',
        'log_revenue',
        'log_vote_count',
        'log_user_rating_count',
        'log_keyword_count',
        'runtime'
    ]
    df_num = df_nz[numeric_vars].dropna()
    
    # Load artifact directory
    artifact_dir = Path("artifacts")
    
    # Load models
    models = {}
    model_files = [
        'scaler', 'pca_full', 'pca_three', 
        'kmeans_base', 'kmeans_pca', 'kmeans_tuned',
        'knn_final', 'mlp_final',
        'ols_full', 'ols_numeric', 'ols_reduced'
    ]
    
    for model_name in model_files:
        try:
            models[model_name] = joblib.load(artifact_dir / f"{model_name}.joblib")
        except FileNotFoundError:
            st.warning(f"Model {model_name} not found in artifacts")
    
    with open(artifact_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    return df_num, models, metadata

df_num, models, metadata = load_data_and_models()

# Prepare data for predictions
X = df_num.drop(columns=['vote_average']).copy()
y = df_num['vote_average'].copy()

scaler = models['scaler']
X_scaled = scaler.transform(X)

# Create tabs for different models
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "ℹ️ About",
    "📊 Linear Regression",
    "🔍 KNN Regression", 
    "📈 K-Means Clustering",
    "🔢 PCA Analysis",
    "🧠 MLP Neural Network",
    "⚖️ Model Comparison",
    "📋 Data Exploration"
])

# ==================== ABOUT TAB ====================
with tab0:
    st.header("About This Project")
    
    st.markdown("""
    ### Movie Rating Prediction Models Dashboard
    
    This interactive dashboard presents a comprehensive analysis of various machine learning models 
    for predicting movie ratings based on engagement metrics, financial data, and content features.
    """)
    
    # Group Members Section
    st.subheader("👥 Group Members")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Member 1**
        - Name: [Your Name]
        - Role: [Your Role]
        """)
    
    with col2:
        st.markdown("""
        **Member 2**
        - Name: [Your Name]
        - Role: [Your Role]
        """)
    
    with col3:
        st.markdown("""
        **Member 3**
        - Name: [Your Name]
        - Role: [Your Role]
        """)
    
    st.markdown("---")
    
    # Dataset Information
    st.subheader("📊 Dataset Information")
    
    st.markdown("""
    **Dataset:** The Movies Dataset
    
    **Source:** [Kaggle - The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
    
    **Description:** This dataset contains metadata for over 45,000 movies from the Full MovieLens Dataset, 
    including information on cast, crew, plot keywords, budget, revenue, posters, release dates, languages, 
    production companies, and countries.
    
    **Features Used:**
    - `vote_average`: Target variable - average rating from users
    - `log_budget`: Log-transformed movie budget
    - `log_revenue`: Log-transformed movie revenue
    - `log_vote_count`: Log-transformed number of votes
    - `log_user_rating_count`: Log-transformed count of user ratings
    - `log_keyword_count`: Log-transformed number of keywords
    - `runtime`: Movie runtime in minutes
    """)
    
    st.markdown("---")
    
    # Project Overview
    st.subheader("🎯 Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Models Implemented:**
        - Linear Regression (OLS)
        - K-Nearest Neighbors (KNN)
        - K-Means Clustering
        - Principal Component Analysis (PCA)
        - Multi-Layer Perceptron (MLP)
        """)
    
    with col2:
        st.markdown("""
        **Key Findings:**
        - MLP achieves best predictive performance (RMSE ≈ 0.88)
        - Engagement metrics dominate predictions
        - 4 distinct movie archetypes identified via clustering
        - 83% of variance captured in 3 principal components
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🚀 How to Use This Dashboard
    
    1. **About** (Current Tab): Learn about the project, dataset, and group members
    2. **Linear Regression**: Explore OLS models with full, numeric, and reduced features
    3. **KNN Regression**: View k-nearest neighbors analysis with hyperparameter tuning
    4. **K-Means Clustering**: Examine unsupervised clustering of movies into archetypes
    5. **PCA Analysis**: Understand dimensionality reduction and feature relationships
    6. **MLP Neural Network**: Analyze the best-performing deep learning model
    7. **Model Comparison**: Compare all models side-by-side with performance metrics
    8. **Data Exploration**: Explore individual column distributions and statistics
    """)

# ==================== LINEAR REGRESSION ====================
with tab1:
    st.header("Linear Regression Models")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("OLS Full Model")
        ols_full = models['ols_full']
        st.metric("R-squared", f"{ols_full.rsquared:.4f}")
        with st.expander("View Model Summary"):
            st.text(str(ols_full.summary()))
    
    with col2:
        st.subheader("OLS Numeric Model")
        ols_numeric = models['ols_numeric']
        st.metric("R-squared", f"{ols_numeric.rsquared:.4f}")
        with st.expander("View Model Summary"):
            st.text(str(ols_numeric.summary()))
    
    with col3:
        st.subheader("OLS Reduced Model")
        ols_reduced = models['ols_reduced']
        st.metric("R-squared", f"{ols_reduced.rsquared:.4f}")
        with st.expander("View Model Summary"):
            st.text(str(ols_reduced.summary()))
    
    st.markdown("""
    **Key Insights:**
    - The full categorical model (R² ≈ 0.38) significantly outperforms numeric-only (R² ≈ 0.26)
    - Engagement metrics (vote_count, user_rating_count) are strongest predictors
    - Genre, language, and actor popularity add crucial interpretive power
    - Larger budgets/revenues surprisingly associated with slightly lower ratings
    """)

# ==================== KNN REGRESSION ====================
with tab2:
    st.header("K-Nearest Neighbors Regression")
    
    knn_final = models['knn_final']
    
    # Make predictions for comparison
    preds_knn = knn_final.predict(X_scaled)
    rmse_knn = np.sqrt(mean_squared_error(y, preds_knn))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("RMSE", f"{rmse_knn:.4f}")
    with col2:
        st.metric("K Neighbors", "18")
    with col3:
        st.metric("Distance Metric", "Manhattan")
    
    st.subheader("KNN Hyperparameter Tuning")
    
    # Prepare data for KNN tuning visualization
    X_for_tuning = df_num.drop(columns=['vote_average']).copy()
    y_for_tuning = df_num['vote_average'].copy()
    
    X_train_tuning, X_valid_tuning, y_train_tuning, y_valid_tuning = train_test_split(
        X_for_tuning, y_for_tuning, test_size=0.25, random_state=3001
    )
    
    scaler_tuning = StandardScaler()
    X_train_scaled_tuning = scaler_tuning.fit_transform(X_train_tuning)
    X_valid_scaled_tuning = scaler_tuning.transform(X_valid_tuning)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Tuned KNN: RMSE by k (Manhattan, Distance-Weighted)**")
        rmse_list_tuned = []
        for k in range(1, 51):
            model = KNeighborsRegressor(
                n_neighbors=k,
                weights='distance',
                metric='manhattan'
            )
            model.fit(X_train_scaled_tuning, y_train_tuning)
            preds = model.predict(X_valid_scaled_tuning)
            rmse = np.sqrt(mean_squared_error(y_valid_tuning, preds))
            rmse_list_tuned.append(rmse)
        
        fig_tuned = px.line(
            x=list(range(1, 51)),
            y=rmse_list_tuned,
            markers=True,
            labels={'x': 'k', 'y': 'Validation RMSE'},
            title='Tuned KNN Validation RMSE by k'
        )
        fig_tuned.update_traces(mode='lines+markers')
        st.plotly_chart(fig_tuned, use_container_width=True)
    
    with col2:
        # Feature importance visualization
        r = permutation_importance(knn_final, X_scaled, y, n_repeats=20, random_state=3001)
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': r.importances_mean
        }).sort_values('importance', ascending=True)
        
        fig_imp = px.bar(importance_df, x='importance', y='feature', 
                          orientation='h', title='Permutation Importance - Tuned KNN')
        st.plotly_chart(fig_imp, use_container_width=True)
    
    st.markdown("""
    **Model Configuration:**
    - n_neighbors: 18
    - weights: distance (closer neighbors weighted more heavily)
    - metric: manhattan (L1 distance)
    - Validation RMSE: ~0.95
    
    **Key Insights:**
    - KNN exploits local similarity in audience engagement patterns
    - Vote count and user rating count dominate predictions
    - Model improves on linear regression by capturing neighborhood structure
    - Manhattan distance with distance weighting balances bias and variance effectively
    """)

# ==================== K-MEANS CLUSTERING ====================
with tab3:
    st.header("K-Means Clustering Analysis")
    
    kmeans_base = models['kmeans_base']
    kmeans_tuned = models['kmeans_tuned']
    
    # Compute clusters
    clusters_base = kmeans_base.predict(X_scaled)
    clusters_tuned = kmeans_tuned.predict(X_scaled)
    
    st.subheader("Baseline K-Means (k=3) vs Tuned K-Means (k=4)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Baseline K-Means (k=3)**")
        st.metric("Inertia", f"{kmeans_base.inertia_:.2f}")
        unique, counts = np.unique(clusters_base, return_counts=True)
        cluster_dist_base = pd.DataFrame({
            'Cluster': unique,
            'Count': counts
        })
        st.dataframe(cluster_dist_base, use_container_width=True)
    
    with col2:
        st.write("**Tuned K-Means (k=4)**")
        st.metric("Inertia", f"{kmeans_tuned.inertia_:.2f}")
        unique, counts = np.unique(clusters_tuned, return_counts=True)
        cluster_dist_tuned = pd.DataFrame({
            'Cluster': unique,
            'Count': counts
        })
        st.dataframe(cluster_dist_tuned, use_container_width=True)
    
    # Cluster tuning analysis
    st.subheader("Cluster Tuning Analysis")
    k_values = range(2, 11)
    inertia_list = []
    silhouette_list = []
    
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=3001, n_init=20)
        labels = km.fit_predict(X_scaled)
        inertia_list.append(km.inertia_)
        silhouette_list.append(silhouette_score(X_scaled, labels))
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_inertia = px.line(x=list(k_values), y=inertia_list, markers=True, 
                             labels={'x': 'k', 'y': 'Inertia'}, 
                             title='Inertia by Number of Clusters')
        fig_inertia.update_traces(mode='lines+markers')
        st.plotly_chart(fig_inertia, use_container_width=True)
    
    with col2:
        fig_silhouette = px.line(x=list(k_values), y=silhouette_list, markers=True, 
                                labels={'x': 'k', 'y': 'Silhouette Score'}, 
                                title='Silhouette Score by Number of Clusters')
        fig_silhouette.update_traces(mode='lines+markers')
        st.plotly_chart(fig_silhouette, use_container_width=True)
    
    # Baseline cluster visualizations
    st.subheader("Baseline Clustering (k=3) Analysis")
    
    pca_2d = PCA(n_components=2)
    X_pca_base = pca_2d.fit_transform(X_scaled)
    
    df_plot_base = pd.DataFrame({
        'PC1': X_pca_base[:, 0],
        'PC2': X_pca_base[:, 1],
        'Cluster': clusters_base.astype(str)
    })
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_clusters_base = px.scatter(
            df_plot_base, x='PC1', y='PC2', color='Cluster',
            title='Baseline K-Means in PCA Space',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig_clusters_base.update_traces(marker=dict(size=5, opacity=0.6))
        st.plotly_chart(fig_clusters_base, use_container_width=True)
    
    with col2:
        df_clusters_base = df_num.copy()
        df_clusters_base['cluster'] = clusters_base
        
        fig_vote_base = px.box(
            df_clusters_base,
            x='cluster',
            y='vote_average',
            title='Vote Average by Baseline Cluster',
            labels={'cluster': 'Cluster', 'vote_average': 'Vote Average'}
        )
        st.plotly_chart(fig_vote_base, use_container_width=True)
    
    with col3:
        cluster_profiles_base = df_clusters_base.groupby('cluster').mean(numeric_only=True)
        cluster_profiles_melted_base = cluster_profiles_base.reset_index().melt(
            id_vars='cluster',
            var_name='feature',
            value_name='value'
        )
        
        fig_profiles_base = px.bar(
            cluster_profiles_melted_base,
            x='feature',
            y='value',
            color='cluster',
            barmode='group',
            title='Baseline Cluster Profiles'
        )
        fig_profiles_base.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_profiles_base, use_container_width=True)
    
    # Tuned cluster visualizations
    st.subheader("Tuned Clustering (k=4) Analysis")
    
    X_pca_tuned = pca_2d.transform(X_scaled)
    
    df_plot_tuned = pd.DataFrame({
        'PC1': X_pca_tuned[:, 0],
        'PC2': X_pca_tuned[:, 1],
        'Cluster': clusters_tuned.astype(str)
    })
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_clusters_tuned = px.scatter(
            df_plot_tuned, x='PC1', y='PC2', color='Cluster',
            title='Tuned K-Means (k=4) in PCA Space',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig_clusters_tuned.update_traces(marker=dict(size=5, opacity=0.6))
        st.plotly_chart(fig_clusters_tuned, use_container_width=True)
    
    with col2:
        df_clusters_tuned = df_num.copy()
        df_clusters_tuned['cluster'] = clusters_tuned
        
        fig_vote_tuned = px.box(
            df_clusters_tuned,
            x='cluster',
            y='vote_average',
            title='Vote Average by Tuned Cluster',
            labels={'cluster': 'Cluster', 'vote_average': 'Vote Average'}
        )
        st.plotly_chart(fig_vote_tuned, use_container_width=True)
    
    with col3:
        cluster_profiles_tuned = df_clusters_tuned.groupby('cluster').mean(numeric_only=True)
        cluster_profiles_melted_tuned = cluster_profiles_tuned.reset_index().melt(
            id_vars='cluster',
            var_name='feature',
            value_name='value'
        )
        
        fig_profiles_tuned = px.bar(
            cluster_profiles_melted_tuned,
            x='feature',
            y='value',
            color='cluster',
            barmode='group',
            title='Tuned Cluster Profiles'
        )
        fig_profiles_tuned.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_profiles_tuned, use_container_width=True)
    
    st.markdown("""
    **Why k=4?**
    - Silhouette score analysis favored k=4 over baseline k=3
    - Provides better separation and interpretability
    - Baseline clusters blended mid and low engagement movies
    - Tuned model resolves this by separating them
    """)

# ==================== PCA ANALYSIS ====================
with tab4:
    st.header("Principal Component Analysis")
    
    pca = models['pca_full']
    pca3 = models['pca_three']
    
    explained_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Scree Plot")
        fig_scree = px.bar(
            x=list(range(1, len(explained_var) + 1)),
            y=explained_var,
            labels={'x': 'Principal Component', 'y': 'Variance Explained'},
            title='Variance Explained by Component'
        )
        st.plotly_chart(fig_scree, use_container_width=True)
    
    with col2:
        st.subheader("Cumulative Variance")
        fig_cum = px.line(
            x=list(range(1, len(cum_var) + 1)),
            y=cum_var,
            markers=True,
            labels={'x': 'Principal Component', 'y': 'Cumulative Variance'},
            title='Cumulative Variance Explained'
        )
        fig_cum.add_hline(y=0.80, line_dash='dash', line_color='red', annotation_text='80% threshold')
        fig_cum.add_hline(y=0.83, line_dash='dash', line_color='green', annotation_text='83% (PC1-3)')
        st.plotly_chart(fig_cum, use_container_width=True)
    
    # PCA loadings
    st.subheader("PCA Loadings (First 3 Components)")
    loadings = pd.DataFrame(
        pca.components_.T[:, :3],
        columns=['PC1', 'PC2', 'PC3'],
        index=X.columns
    )
    
    loadings_melted = loadings.reset_index().melt(
        id_vars='index', var_name='Component', value_name='Loading'
    )
    fig_loadings = px.bar(
        loadings_melted, x='index', y='Loading', color='Component',
        barmode='group', title='Feature Contributions to Principal Components'
    )
    st.plotly_chart(fig_loadings, use_container_width=True)
    
    # 2D PCA projections
    st.subheader("2D PCA Projections")
    
    pca_2d = PCA(n_components=2)
    X_pca2d = pca_2d.fit_transform(X_scaled)
    
    col1, col2 = st.columns(2)
    
    with col1:
        df_pca2d = pd.DataFrame({
            'PC1': X_pca2d[:, 0],
            'PC2': X_pca2d[:, 1],
            'Vote Average': y.values
        })
        
        fig_2d = px.scatter(
            df_pca2d, x='PC1', y='PC2', color='Vote Average',
            title='Movies Projected onto PC1 and PC2',
            color_continuous_scale='viridis',
            opacity=0.6
        )
        fig_2d.update_traces(marker=dict(size=4))
        st.plotly_chart(fig_2d, use_container_width=True)
    
    with col2:
        clusters_tuned = models['kmeans_tuned'].predict(X_scaled)
        df_pca_clusters = pd.DataFrame({
            'PC1': X_pca2d[:, 0],
            'PC2': X_pca2d[:, 1],
            'Cluster': clusters_tuned.astype(str)
        })
        
        fig_clusters_2d = px.scatter(
            df_pca_clusters, x='PC1', y='PC2', color='Cluster',
            title='Tuned K-Means Clusters on PC1-PC2',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig_clusters_2d.update_traces(marker=dict(size=5, opacity=0.6))
        st.plotly_chart(fig_clusters_2d, use_container_width=True)
    
    # 3D PCA projections
    st.subheader("3D PCA Projections")
    
    X_pca3d = pca3.transform(X_scaled)
    
    col1, col2 = st.columns(2)
    
    with col1:
        df_pca3d = pd.DataFrame({
            'PC1': X_pca3d[:, 0],
            'PC2': X_pca3d[:, 1],
            'PC3': X_pca3d[:, 2],
            'Vote Average': y.values
        })
        
        fig_3d = px.scatter_3d(
            df_pca3d, x='PC1', y='PC2', z='PC3', color='Vote Average',
            title='3D PCA Projection Colored by Vote Average',
            color_continuous_scale='viridis'
        )
        fig_3d.update_traces(marker=dict(size=2, opacity=0.6))
        st.plotly_chart(fig_3d, use_container_width=True)
    
    with col2:
        clusters_pca = models['kmeans_pca'].predict(X_pca3d)
        df_pca3d_clusters = pd.DataFrame({
            'PC1': X_pca3d[:, 0],
            'PC2': X_pca3d[:, 1],
            'PC3': X_pca3d[:, 2],
            'Cluster': clusters_pca.astype(str)
        })
        
        fig_3d_clusters = px.scatter_3d(
            df_pca3d_clusters, x='PC1', y='PC2', z='PC3', color='Cluster',
            title='Tuned K-Means Clusters in 3D PCA Space',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig_3d_clusters.update_traces(marker=dict(size=3, opacity=0.6))
        st.plotly_chart(fig_3d_clusters, use_container_width=True)
    
    # K-Means on PCs directly
    st.subheader("K-Means Clustering on First 3 PCs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        df_pc_clusters_2d = pd.DataFrame({
            'PC1': X_pca3d[:, 0],
            'PC2': X_pca3d[:, 1],
            'Cluster': clusters_pca.astype(str)
        })
        
        fig_pc_2d = px.scatter(
            df_pc_clusters_2d, x='PC1', y='PC2', color='Cluster',
            title='K-Means on PCs: PC1 vs PC2',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig_pc_2d.update_traces(marker=dict(size=5, opacity=0.6))
        st.plotly_chart(fig_pc_2d, use_container_width=True)
    
    with col2:
        fig_pc_3d = px.scatter_3d(
            df_pca3d_clusters, x='PC1', y='PC2', z='PC3', color='Cluster',
            title='K-Means Clustering Using First 3 PCs (3D)',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig_pc_3d.update_traces(marker=dict(size=3, opacity=0.6))
        st.plotly_chart(fig_pc_3d, use_container_width=True)
    
    st.markdown(f"""
    **PCA Summary:**
    - **PC1 ({explained_var[0]:.1%} variance)**: Engagement & Financial Scale
      - Driven by: vote_count, user_rating_count, budget, revenue, runtime
    - **PC2 ({explained_var[1]:.1%} variance)**: Runtime & Content Richness
      - Driven by: runtime, keyword_count
    - **PC3 ({explained_var[2]:.1%} variance)**: Engagement Pattern Variation
      - Driven by: smaller differences in voting behavior
    
    **Key Insight:** First 3 PCs capture **{cum_var[2]:.1%}** of total variance
    → Dimensionality reduction is highly effective for this dataset
    """)

# ==================== MLP NEURAL NETWORK ====================
with tab5:
    st.header("Multi-Layer Perceptron (MLP) Regression")
    
    mlp_final = models['mlp_final']
    
    # Make predictions
    preds_mlp = mlp_final.predict(X_scaled)
    rmse_mlp = np.sqrt(mean_squared_error(y, preds_mlp))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RMSE", f"{rmse_mlp:.4f}")
    with col2:
        st.metric("Hidden Layers", "(64,)")
    with col3:
        st.metric("Activation", "ReLU")
    with col4:
        st.metric("Solver", "Adam")
    
    # Training loss curve
    st.subheader("Training Dynamics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_loss = px.line(
            x=list(range(len(mlp_final.loss_curve_))),
            y=mlp_final.loss_curve_,
            markers=True,
            labels={'x': 'Iteration', 'y': 'Training Loss'},
            title='MLP Training Convergence'
        )
        fig_loss.update_traces(mode='lines+markers')
        st.plotly_chart(fig_loss, use_container_width=True)
    
    with col2:
        # Feature importance
        r_mlp = permutation_importance(mlp_final, X_scaled, y, n_repeats=20, random_state=3001)
        importance_mlp = pd.DataFrame({
            'feature': X.columns,
            'importance': r_mlp.importances_mean
        }).sort_values('importance', ascending=True)
        
        fig_mlp_imp = px.bar(
            importance_mlp, x='importance', y='feature',
            orientation='h', title='Permutation Importance - Final MLP'
        )
        st.plotly_chart(fig_mlp_imp, use_container_width=True)
    
    # Prediction analysis
    st.subheader("Prediction Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        df_scatter = pd.DataFrame({
            'Predicted': preds_mlp,
            'Actual': y.values
        })
        fig_scatter = px.scatter(
            df_scatter, x='Actual', y='Predicted',
            title='Predicted vs Actual Values',
            opacity=0.5
        )
        # Add diagonal line
        min_val = min(df_scatter['Actual'].min(), df_scatter['Predicted'].min())
        max_val = max(df_scatter['Actual'].max(), df_scatter['Predicted'].max())
        fig_scatter.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                      mode='lines', name='Perfect Prediction',
                      line=dict(dash='dash', color='red'))
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        residuals = y.values - preds_mlp
        fig_residuals = px.histogram(
            {'Residuals': residuals},
            nbins=50,
            title='Distribution of Residuals'
        )
        st.plotly_chart(fig_residuals, use_container_width=True)
    
    with col3:
        # Residuals vs predicted
        fig_res_scatter = px.scatter(
            x=preds_mlp, y=residuals,
            title='Residuals vs Fitted Values',
            labels={'x': 'Fitted Values', 'y': 'Residuals'},
            opacity=0.5
        )
        fig_res_scatter.add_hline(y=0, line_dash='dash', line_color='red')
        st.plotly_chart(fig_res_scatter, use_container_width=True)
    
    st.markdown("""
    **Model Architecture:**
    - Input Layer: 7 numeric features
    - Hidden Layer: 64 neurons with ReLU activation
    - Output Layer: 1 (vote_average prediction)
    - Regularization (Alpha): 0.0001
    - Optimizer: Adam with adaptive learning rate
    
    **Performance:**
    - Training RMSE: ~0.846
    - Validation RMSE: ~0.882
    - Small generalization gap indicates good balance
    
    **Key Findings:**
    - MLP captures smooth nonlinear relationships between features
    - Log vote count overwhelmingly dominant predictor
    - Outperforms both linear regression and KNN
    - Training converges smoothly without oscillation
    """)

# ==================== MODEL COMPARISON ====================
with tab6:
    st.header("Model Comparison & Benchmarking")
    
    # Prepare comparison data
    ols_full = models['ols_full']
    ols_numeric = models['ols_numeric']
    ols_reduced = models['ols_reduced']
    knn_final = models['knn_final']
    mlp_final = models['mlp_final']
    
    # Calculate metrics for all models
    preds_ols_full = ols_full.fittedvalues
    preds_ols_numeric = ols_numeric.fittedvalues
    preds_ols_reduced = ols_reduced.fittedvalues
    preds_knn = knn_final.predict(X_scaled)
    preds_mlp = mlp_final.predict(X_scaled)
    
    comparison_data = {
        'Model': [
            'OLS Full',
            'OLS Numeric',
            'OLS Reduced',
            'KNN (k=18)',
            'MLP (64)'
        ],
        'RMSE': [
            np.sqrt(mean_squared_error(ols_full.model.endog, preds_ols_full)),
            np.sqrt(mean_squared_error(ols_numeric.model.endog, preds_ols_numeric)),
            np.sqrt(mean_squared_error(ols_reduced.model.endog, preds_ols_reduced)),
            np.sqrt(mean_squared_error(y, preds_knn)),
            np.sqrt(mean_squared_error(y, preds_mlp))
        ],
        'R-squared': [
            ols_full.rsquared,
            ols_numeric.rsquared,
            ols_reduced.rsquared,
            1 - (mean_squared_error(y, preds_knn) / np.var(y)),
            1 - (mean_squared_error(y, preds_mlp) / np.var(y))
        ],
        'Type': [
            'Linear',
            'Linear',
            'Linear',
            'Non-parametric',
            'Neural Network'
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display comparison table
    st.subheader("Performance Metrics")
    st.dataframe(
        comparison_df.style.format({
            'RMSE': '{:.4f}',
            'R-squared': '{:.4f}'
        }),
        use_container_width=True
    )
    
    # Visualize comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig_rmse = px.bar(
            comparison_df, x='Model', y='RMSE',
            color='Type', title='Model RMSE Comparison',
            color_discrete_map={
                'Linear': '#1f77b4',
                'Non-parametric': '#ff7f0e',
                'Neural Network': '#2ca02c'
            }
        )
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    with col2:
        fig_r2 = px.bar(
            comparison_df, x='Model', y='R-squared',
            color='Type', title='Model R-squared Comparison',
            color_discrete_map={
                'Linear': '#1f77b4',
                'Non-parametric': '#ff7f0e',
                'Neural Network': '#2ca02c'
            }
        )
        st.plotly_chart(fig_r2, use_container_width=True)
    
    # Prediction comparison scatter
    st.subheader("Prediction Distributions")
    
    prediction_comparison = pd.DataFrame({
        'Actual': np.concatenate([y.values]*3),
        'Predicted': np.concatenate([
            preds_knn, preds_mlp, preds_ols_numeric
        ]),
        'Model': ['KNN']*len(y) + ['MLP']*len(y) + ['OLS Numeric']*len(y)
    })
    
    fig_compare = px.scatter(
        prediction_comparison, x='Actual', y='Predicted', color='Model',
        title='Predictions Across Models',
        opacity=0.6
    )
    st.plotly_chart(fig_compare, use_container_width=True)
    
    # Summary insights
    st.markdown("""
    ## Model Comparison Summary
    
    ### Linear Regression Models
    - **OLS Full** (R² ≈ 0.38): Best linear model with categorical features
    - **OLS Numeric** (R² ≈ 0.26): Numeric-only baseline
    - **OLS Reduced** (R² ≈ 0.37): Simplified version of full model
    - **Insight**: Categorical features crucial for linear interpretation
    
    ### K-Nearest Neighbors
    - **KNN (k=18)**: RMSE ≈ 0.95
    - **Insight**: Exploits local structure in engagement patterns
    - Moderate improvement over OLS models through distance weighting
    
    ### Multi-Layer Perceptron (BEST) ⭐
    - **MLP (64 hidden units)**: RMSE ≈ 0.88, R² ≈ 0.60
    - **Why it wins**:
      - Captures smooth nonlinear relationships
      - No feature engineering required
      - Best generalization performance
      - Stable training convergence
    
    ### Clustering Models (Unsupervised)
    - **K-Means**: Reveals 4 natural movie archetypes
    - **PCA**: Shows 83% variance captured in 3 dimensions
    - **Use**: Data exploration and segmentation, not prediction
    
    ### Recommendation
    **Use MLP for production predictions** - it balances accuracy, stability, and interpretability
    """)

# ==================== DATA EXPLORATION ====================
with tab7:
    st.header("Data Exploration & Column Statistics")
    
    st.markdown("**Select a column to view its distribution and statistics:**")
    
    # Column selector
    selected_column = st.selectbox(
        "Choose a column to analyze:",
        options=df_num.columns,
        index=0
    )
    
    # Get statistics for the selected column
    col_data = df_num[selected_column]
    
    # Create three columns for metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Mean", f"{col_data.mean():.4f}")
    with col2:
        st.metric("Median", f"{col_data.median():.4f}")
    with col3:
        st.metric("Std Dev", f"{col_data.std():.4f}")
    with col4:
        st.metric("Min", f"{col_data.min():.4f}")
    with col5:
        st.metric("Max", f"{col_data.max():.4f}")
    
    # Display detailed statistics
    st.subheader("Detailed Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stats_data = {
            'Statistic': [
                'Count',
                'Mean',
                'Median',
                'Mode',
                'Std Dev',
                'Variance',
                'Skewness',
                'Kurtosis',
                'Q1 (25%)',
                'Q3 (75%)',
                'IQR',
                'Min',
                'Max',
                'Range'
            ],
            'Value': [
                f"{col_data.count():.0f}",
                f"{col_data.mean():.4f}",
                f"{col_data.median():.4f}",
                f"{col_data.mode().values[0]:.4f}" if len(col_data.mode()) > 0 else "N/A",
                f"{col_data.std():.4f}",
                f"{col_data.var():.4f}",
                f"{col_data.skew():.4f}",
                f"{col_data.kurtosis():.4f}",
                f"{col_data.quantile(0.25):.4f}",
                f"{col_data.quantile(0.75):.4f}",
                f"{col_data.quantile(0.75) - col_data.quantile(0.25):.4f}",
                f"{col_data.min():.4f}",
                f"{col_data.max():.4f}",
                f"{col_data.max() - col_data.min():.4f}"
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.write("**Missing Values:**")
        missing_count = col_data.isnull().sum()
        missing_pct = (missing_count / len(col_data)) * 100
        
        missing_data = {
            'Metric': ['Missing Count', 'Missing %', 'Non-Missing Count'],
            'Value': [
                f"{missing_count}",
                f"{missing_pct:.2f}%",
                f"{col_data.count()}"
            ]
        }
        
        missing_df = pd.DataFrame(missing_data)
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    
    # Distribution visualizations
    st.subheader(f"Distribution of {selected_column}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram with KDE
        fig_hist = px.histogram(
            df_num,
            x=selected_column,
            nbins=50,
            title=f"Histogram of {selected_column}",
            marginal="box",
            labels={selected_column: selected_column}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box plot
        fig_box = px.box(
            df_num,
            y=selected_column,
            title=f"Box Plot of {selected_column}",
            labels={selected_column: selected_column}
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Q-Q Plot and correlation info
    col1, col2 = st.columns(2)
    
    with col1:
        # Violin plot
        fig_violin = px.violin(
            df_num,
            y=selected_column,
            box=True,
            title=f"Violin Plot of {selected_column}",
            labels={selected_column: selected_column}
        )
        st.plotly_chart(fig_violin, use_container_width=True)
    
    with col2:
        # Correlation with other columns
        st.write("**Correlation with Other Columns:**")
        correlations = df_num.corr()[selected_column].sort_values(ascending=False)
        
        # Create a correlation bar chart (excluding self-correlation)
        corr_for_plot = correlations[correlations.index != selected_column]
        
        fig_corr = px.bar(
            x=corr_for_plot.values,
            y=corr_for_plot.index,
            orientation='h',
            title=f"Correlations with {selected_column}",
            labels={'x': 'Correlation', 'y': 'Column'}
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Additional insights
    st.subheader("Distribution Insights")
    
    insights = []
    
    # Skewness interpretation
    skewness = col_data.skew()
    if abs(skewness) < 0.5:
        skew_text = "approximately symmetric"
    elif skewness > 0:
        skew_text = "right-skewed (positive skew)"
    else:
        skew_text = "left-skewed (negative skew)"
    insights.append(f"**Skewness ({skewness:.4f})**: The distribution is {skew_text}")
    
    # Kurtosis interpretation
    kurtosis = col_data.kurtosis()
    if abs(kurtosis) < 1:
        kurt_text = "normal-like tails"
    elif kurtosis > 1:
        kurt_text = "heavy tails (more outliers)"
    else:
        kurt_text = "light tails (fewer outliers)"
    insights.append(f"**Kurtosis ({kurtosis:.4f})**: The distribution has {kurt_text}")
    
    # Outlier detection using IQR method
    Q1 = col_data.quantile(0.25)
    Q3 = col_data.quantile(0.75)
    IQR = Q3 - Q1
    outlier_count = ((col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))).sum()
    outlier_pct = (outlier_count / len(col_data)) * 100
    insights.append(f"**Outliers (IQR method)**: {outlier_count} outliers ({outlier_pct:.2f}% of data)")
    
    for insight in insights:
        st.markdown(insight)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Movie Rating Prediction Models Dashboard | Models trained on scaled numeric features</p>
    <p>⚡ Cached models loaded from artifacts for instant performance</p>
</div>
""", unsafe_allow_html=True)
