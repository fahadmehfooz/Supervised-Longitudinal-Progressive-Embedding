import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


def plot_subject_progression(df, model_name, y_label, figsize):
    """
    Plots the progression of subjects over age for a specified number of subjects.

    Parameters:
    df (DataFrame): DataFrame containing 'RID', 'AGE', and 'Pseudotime_Normalized' columns.
    num_subjects (int): The number of subjects (RIDs) to plot.
    model_name (str): Name of the model for the plot title.
    figsize (tuple): Figure size.

    Returns:
    tuple: Figure and Axes objects.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Getting the unique RIDs in the dataframe
    unique_rids = df['RID'].unique()

    num_subjects = df["RID"].nunique()

    # Looping through each RID and plot their progression
    for i, rid in enumerate(unique_rids[:num_subjects]):
        subset = df[df['RID'] == rid]

        # Checking if the RID has multiple points (progression)
        if len(subset) > 1:
            # Plot with lines connecting points
            ax.plot(subset['AGE'], subset['Pseudotime_Normalized'], 'r-', marker='x')

    ax.set_xlabel('Age')
    ax.set_ylabel(y_label)
    ax.set_title(f'{model_name}: Subjects progression over Age')

    plt.close(fig) 
    return fig, ax

def plot_embedding_results(image_paths, title, figsize):

    fig, axes = plt.subplots(1, len(image_paths), figsize=figsize)

    for ax, img_path in zip(axes, image_paths):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('off')  

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_progression_all_models(figures_and_axes):
    fig = plt.figure(figsize=(18, 12))

    ax1 = plt.subplot2grid((2, 6), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
    ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2)

    ax4 = plt.subplot2grid((2, 6), (1, 0), colspan=2)
    ax5 = plt.subplot2grid((2, 6), (1, 2), colspan=2)
    ax6 = plt.subplot2grid((2, 6), (1, 4), colspan=2)

    # Plot AE with Direction Loss
    ae_dl_fig, ae_dl_ax = figures_and_axes[0]
    for line in ae_dl_ax.lines:
        ax1.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), marker=line.get_marker())
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Pseudotime')
    ax1.set_title('Slope')

    # Plot AE without Direction Loss
    ae_ndl_fig, ae_ndl_ax = figures_and_axes[1] 
    for line in ae_ndl_ax.lines:
        ax2.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), marker=line.get_marker())
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Pseudotime')
    ax2.set_title('Slope Without Direction Loss')

    # Plot LNE
    ae_LNE_fig, ae_LNE_ax = figures_and_axes[2]
    for line in ae_LNE_ax.lines:
        ax3.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), marker=line.get_marker())
    ax3.set_xlabel('Age')
    ax3.set_ylabel('Pseudotime')
    ax3.set_title('LNE')

    # Plot Logistic Regression
    lr_fig, lr_ax = figures_and_axes[3]
    for line in lr_ax.lines:
        ax4.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), marker=line.get_marker())
    ax4.set_xlabel('Age')
    ax4.set_ylabel('Logit')
    ax4.set_title('Logistic Regression')

    # Plot Elastic Net
    en_fig, en_ax = figures_and_axes[4]
    for line in en_ax.lines:
        ax5.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), marker=line.get_marker())
    ax5.set_xlabel('Age')
    ax5.set_ylabel('Logit')
    ax5.set_title('Elastic Net')

    # Plot MLP
    mlp_fig, mlp_ax = figures_and_axes[5]
    for line in mlp_ax.lines:
        ax6.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), marker=line.get_marker())
    ax6.set_xlabel('Age')
    ax6.set_ylabel('Logit')
    ax6.set_title('MLP')

    fig.suptitle('Subject Progression over Age', fontsize=16, fontweight='bold', y=1.05, ha='center')

    plt.tight_layout()
    plt.show()


def plot_violation_metrics(results, thresholds, figsize=(15, 8)):
    fig, axs = plt.subplots(1, 2, figsize = figsize )

    initial_threshold_ratios = [metrics['vio_ratios'][0] for metrics in results.values()]
    models = list(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

    # Plot 1: Violation Ratio vs Threshold for each model
    for name, metrics in results.items():
        axs[0].plot(thresholds, metrics['vio_ratios'], label=name, marker='o')
    axs[0].set_title('Violation Ratio vs Threshold')
    axs[0].set_xlabel('Threshold')
    axs[0].set_ylabel('Violation Ratio')
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: Violation Gap vs Threshold for each model
    for name, metrics in results.items():
        axs[1].plot(thresholds, metrics['vio_gaps'], label=name, marker='o')
    axs[1].set_title('Violation Gap vs Threshold')
    axs[1].set_xlabel('Threshold')
    axs[1].set_ylabel('Violation Gap')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_umap_embeddings(data_2d_umap_train_dl, data_2d_umap_test_dl, y_train, y_test, title):
    """
    Plot UMAP embeddings for train and test datasets with subplots.

    Parameters:
    - data_2d_umap_train_dl: DataFrame or array-like, 2D UMAP embeddings for train set (2 columns).
    - data_2d_umap_test_dl: DataFrame or array-like, 2D UMAP embeddings for test set (2 columns).
    - y_train: Labels for the train set.
    - y_test: Labels for the test set.
    - title: Title for the entire plot.

    Returns:
    None
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    sns.scatterplot(
        x=data_2d_umap_train_dl["UMAP1"],
        y=data_2d_umap_train_dl["UMAP2"],
        hue=y_train,
        palette="viridis",
        ax=axes[0]
    )
    axes[0].set_title("Train Data")
    axes[0].set_xlabel("UMAP Dimension 1")
    axes[0].set_ylabel("UMAP Dimension 2")
    axes[0].legend(title="Labels", loc="best", fontsize="small")

    sns.scatterplot(
        x=data_2d_umap_test_dl["UMAP1"],
        y=data_2d_umap_test_dl["UMAP2"],
        hue=y_test,
        palette="viridis",
        ax=axes[1],
    )
    axes[1].set_title("Test Data")
    axes[1].set_xlabel("UMAP Dimension 1")
    axes[1].set_ylabel("UMAP Dimension 2")
    axes[1].legend(title="Labels", loc="best", fontsize="small")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.show()

