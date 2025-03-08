library(ggplot2)
library(dplyr)
library(ggpubr)
library(grid)

# Working directory needs to be ->/Supervised-Longitudinal-Progressive-Embedding/Modules
# Ensure you follow our structure
# Set working directory
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

generate_model_boxplot <- function(data, model_type, dataset_type, show_x_title = FALSE) {
  my_colors <- c("CN" = "#F8766D", "EMCI" = "#7CAE00", "LMCI" = "#00BFC4", "AD" = "#C77CFF")
  my_comparisons <- list(c("CN", "EMCI"), c("EMCI", "LMCI"), c("LMCI", "AD"))
  
  x_title <- if(show_x_title) "Diagnosis Groups" else NULL
  
  ggboxplot(data, x = "DX", y = "Pseudotime_Normalized",
            fill = "DX", outlier.size = 0.001) +
    stat_compare_means(comparisons = my_comparisons,
                       aes(label = ..p.signif..),
                       method = "t.test", size = 3.5,
                       step.increase = 0.15, tip.length = 0.01) +
    labs(title = paste0(model_type), 
         y = NULL, x = x_title) + 
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    standard_theme +
    theme(
      panel.grid = element_blank(),
      axis.text.x = element_text(color = "black", size = 8, face = "bold", angle = 45, hjust = 1),
      axis.text.y = element_text(color = "black", size = 8, face = "bold"),
      plot.margin = margin(10, 10, 10, 10, "pt"),
      legend.position = "none"
    )
}

plot_objects <- list(
  train_lr = generate_model_boxplot(train_lr, "Logistic Regression", "Train"),
  train_en = generate_model_boxplot(train_en, "Elastic Net", "Train"),
  train_mlp = generate_model_boxplot(train_mlp, "MLP", "Train"),
  test_lr = generate_model_boxplot(test_lr, "Logistic Regression", "Test"),
  test_en = generate_model_boxplot(test_en, "Elastic Net", "Test"),
  test_mlp = generate_model_boxplot(test_mlp, "MLP", "Test")
)

combined_train <- ggarrange(
  plot_objects$train_lr,
  plot_objects$train_en + theme(axis.text.y = element_blank()),
  plot_objects$train_mlp + theme(axis.text.y = element_blank()),
  nrow = 1,
  labels = c("(a)", "(b)", "(c)"),
  font.label = list(size = 8, face = "bold"),
  legend = "none"
) %>% 
  annotate_figure(
    left = textGrob("Pseudotime", rot = 90, gp = gpar(fontsize = 8, fontface = "bold")),
    bottom = textGrob("Diagnosis Groups", gp = gpar(fontsize = 8, fontface = "bold"))
  )

combined_test <- ggarrange(
  plot_objects$test_lr,
  plot_objects$test_en + theme(axis.text.y = element_blank()),
  plot_objects$test_mlp + theme(axis.text.y = element_blank()),
  nrow = 1,
  labels = c("(a)", "(b)", "(c)"),
  font.label = list(size = 8, face = "bold"),
  legend = "none"
) %>% 
  annotate_figure(
    left = textGrob("Pseudotime", rot = 90, gp = gpar(fontsize = 10, fontface = "bold")),
    bottom = textGrob("Diagnosis Groups", gp = gpar(fontsize = 8, fontface = "bold"))
  )

ggsave(file.path(output_dir, "Supervised_train_plots.png"), combined_train,
       width = 5, height = 2, dpi = 300, bg = "white")
ggsave(file.path(output_dir, "Supervised_test_plots.png"), combined_test,
       width = 5, height = 2, dpi = 300, bg = "white")