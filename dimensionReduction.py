import umap.umap_ as umap
from sklearn.decomposition import PCA
from plots import plot_pca, plot_umap

def dimensionalityReduction(apples):
    # Apply PCA
    pca_result, pca_model = apply_pca(apples, n_components=2)

    # Plot PCA
    plot_pca(pca_result, apples)

    # Apply UMAP
    umap_result = apply_umap(apples)

    # Plot UMAP
    plot_umap(umap_result, apples['Quality'])

def apply_pca(dataset, n_components=None):
    # Apply PCA
    pca_model = PCA(n_components=n_components)
    pca_result = pca_model.fit_transform(dataset)
    return pca_result, pca_model

def apply_umap(data_standardized):
    # Apply UMAP
    reducer = umap.UMAP()
    umap_result = reducer.fit_transform(data_standardized)
    return umap_result