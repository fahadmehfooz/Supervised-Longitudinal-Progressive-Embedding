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

    # Bold labels and title
    ax.set_xlabel('Age', fontweight='bold')     
    ax.set_ylabel(y_label, fontweight='bold')     
    ax.set_title(f'{model_name}: Subjects progression over Age', fontweight='bold')      

    plt.close(fig)      

    return fig, ax

def plot_embedding_results(filename, title="Image", figsize = (8, 6)):
    img = mpimg.imread(filename)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis("off")  
    plt.title(title, fontweight="bold")
    plt.show()



def plot_progression_all_models(figures_and_axes):
    fig = plt.figure(figsize=(18, 10))
    
    ax1 = plt.subplot2grid((2, 6), (0, 1), colspan=2)
    ax2 = plt.subplot2grid((2, 6), (0, 3), colspan=2)
    
    ax3 = plt.subplot2grid((2, 6), (1, 0), colspan=2)
    ax4 = plt.subplot2grid((2, 6), (1, 2), colspan=2)
    ax5 = plt.subplot2grid((2, 6), (1, 4), colspan=2)
    
    SLOPE_fig, SLOPE_ax = figures_and_axes[0]
    for line in SLOPE_ax.lines:
        ax1.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), marker=line.get_marker())
    ax1.set_xlabel('Age', fontweight='bold')
    ax1.set_ylabel('Pseudotime', fontweight='bold')
    ax1.set_title('SLOPE', fontweight='bold')
    
    Autoencoder_fig, Autoencoder_ax = figures_and_axes[1]
    for line in Autoencoder_ax.lines:
        ax2.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), marker=line.get_marker())
    ax2.set_xlabel('Age', fontweight='bold')
    ax2.set_ylabel('Pseudotime', fontweight='bold')
    ax2.set_title('Autoencoder', fontweight='bold')
    
    mlp_fig, mlp_ax = figures_and_axes[2]
    for line in mlp_ax.lines:
        ax3.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), marker=line.get_marker())
    ax3.set_title('Logistic Regression', fontweight='bold')
    ax3.set_xlabel('Age', fontweight='bold')
    ax3.set_ylabel('Logit', fontweight='bold')
    
    en_fig, en_ax = figures_and_axes[3]
    for line in en_ax.lines:
        ax4.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), marker=line.get_marker())
    ax4.set_title('Elastic Net', fontweight='bold')
    ax4.set_xlabel('Age', fontweight='bold')
    ax4.set_ylabel('Logit', fontweight='bold')
    
    lr_fig, lr_ax = figures_and_axes[4]
    for line in lr_ax.lines:
        ax5.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), marker=line.get_marker())
    ax5.set_title('MLP', fontweight='bold')
    ax5.set_xlabel('Age', fontweight='bold')
    ax5.set_ylabel('Logit', fontweight='bold')
    
    fig.delaxes(plt.subplot2grid((2, 6), (0, 0)))
    fig.delaxes(plt.subplot2grid((2, 6), (0, 5)))
    
    fig.suptitle('Subject Progression over Age', fontsize=16, fontweight='bold', y=1.05, ha='center')
    
    plt.tight_layout()
    plt.show()


def plot_violation_metrics(results, thresholds, figsize=(15, 8)):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    initial_threshold_ratios = [metrics['vio_ratios'][0] for metrics in results.values()]
    models = list(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    
    # Plot 1: Violation Ratio vs Threshold for each model
    for name, metrics in results.items():
        axs[0].plot(thresholds, metrics['vio_ratios'], label=name, marker='o')
    axs[0].set_title('Violation Ratio vs Threshold', fontweight='bold')
    axs[0].set_xlabel('Threshold', fontweight='bold')
    axs[0].set_ylabel('Violation Ratio', fontweight='bold')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot 2: Violation Gap vs Threshold for each model
    for name, metrics in results.items():
        axs[1].plot(thresholds, metrics['vio_gaps'], label=name, marker='o')
    axs[1].set_title('Violation Gap vs Threshold', fontweight='bold')
    axs[1].set_xlabel('Threshold', fontweight='bold')
    axs[1].set_ylabel('Violation Gap', fontweight='bold')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
def plot_umap_embeddings(train_data, test_data, y_train, y_test, title):

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    sns.scatterplot(
        x=train_data["UMAP1"],
        y=train_data["UMAP2"],
        hue=y_train,
        palette="viridis",
        ax=axes[0]
    )
    axes[0].set_title("Train Data")
    axes[0].set_xlabel("UMAP Dimension 1")
    axes[0].set_ylabel("UMAP Dimension 2")
    axes[0].legend(title="Labels", loc="best", fontsize="small")

    sns.scatterplot(
        x=test_data["UMAP1"],
        y=test_data["UMAP2"],
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
