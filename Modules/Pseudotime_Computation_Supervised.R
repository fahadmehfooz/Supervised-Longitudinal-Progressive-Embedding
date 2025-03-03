library(ggplot2)
library(dplyr)
library(ggpubr)
library(RColorBrewer)
library(slingshot)
library(fields)

# Working directory needs to be ->/Supervised-Longitudinal-Progressive-Embedding/Modules
# Ensure you follow our structure
# Set working directory

output_dir <- "../../Supervised-Longitudinal-Progressive-Embedding/Temp Files"

# Standard plot theme
standard_theme <- theme_minimal() + 
  theme(
    plot.title = element_text(hjust = 0.5, size = 8, face = "bold"),
    axis.title = element_text(size = 8, face = "bold"),
    axis.text.x = element_text(size = 8, face = "bold", color = "black", angle = 45, hjust = 1),
    axis.text.y = element_text(size = 8, face = "bold", color = "black"),
    legend.title = element_text(size = 8, face = "bold"),
    legend.text = element_text(size = 8, face = "bold"),
    legend.position = "right",
    legend.key.size = unit(0.3, "cm"),
    axis.line.x = element_line(color = "black", linewidth = 0.1),
    axis.line.y = element_line(color = "black", linewidth = 0.1),
    axis.ticks = element_line(color = "black", linewidth = 0.1),
    axis.ticks.length = unit(0.05, "cm"),
    panel.grid = element_blank(),
    plot.margin = margin(10, 10, 10, 10, "pt")
  )

# Function to prepare data
prepare_data <- function(df) {
  df$DXGrp <- factor(df$DXGrp, levels = c(1, 2, 3, 4), labels = c("CN", "EMCI", "LMCI", "AD"))
  df$DX <- df$DXGrp
  return(df)
}

# Read CSV files using relative paths
data_dir <- "../../Supervised-Longitudinal-Progressive-Embedding/Embeddings"

train_lr <- prepare_data(read.csv(file.path(data_dir, "train_df_lr.csv")))
test_lr <- prepare_data(read.csv(file.path(data_dir, "test_df_lr.csv")))
train_mlp <- prepare_data(read.csv(file.path(data_dir, "train_df_mlp.csv")))
test_mlp <- prepare_data(read.csv(file.path(data_dir, "test_df_mlp.csv")))
train_en <- prepare_data(read.csv(file.path(data_dir, "train_df_en.csv")))
test_en <- prepare_data(read.csv(file.path(data_dir, "test_df_en.csv")))

# Function to generate boxplots
generate_model_boxplot <- function(data, model_type, dataset_type) {
  my_colors <- c("CN" = "#F8766D", "EMCI" = "#7CAE00", "LMCI" = "#00BFC4", "AD" = "#C77CFF")
  my_comparisons <- list(c("CN", "EMCI"), c("EMCI", "LMCI"), c("LMCI", "AD"))
  
  p <- ggboxplot(data, x = "DX", y = "Pseudotime_Normalized",
                 fill = "DX", outlier.size = 0.001) +
    stat_compare_means(comparisons = my_comparisons,
                       aes(label = ..p.signif..),
                       method = "t.test", size = 3.5,
                       step.increase = 0.15, tip.length = 0.01) +
    labs(title = "Pseudotime Vs Diagnosis Group",
         y = "Pseudotime", x = "Diagnosis Groups") +
    labs(fill = "Diagnosis\nGroups") +
    scale_fill_manual(values = my_colors) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    standard_theme +
    theme(legend.position = "right",
          plot.title = element_text(hjust = 0.5)) +
    guides(fill = guide_legend(override.aes = list(color = NA)))
  
  filename <- paste0("Plot_", dataset_type, "_boxplot_", tolower(model_type), ".png")
  ggsave(file.path(output_dir, filename), p, width = 2.5, height = 2, dpi = 300)
}

# Generate boxplots
generate_model_boxplot(train_lr, "LR", "train")
generate_model_boxplot(test_lr, "LR", "test")
generate_model_boxplot(train_mlp, "MLP", "train")
generate_model_boxplot(test_mlp, "MLP", "test")
generate_model_boxplot(train_en, "EN", "train")
generate_model_boxplot(test_en, "EN", "test")

graphics.off()
