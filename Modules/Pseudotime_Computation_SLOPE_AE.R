library(ggplot2)
library(dplyr)
library(ggpubr)
library(RColorBrewer)
library(slingshot)
library(fields)

# Working directory needs to be ->/Supervised-Longitudinal-Progressive-Embedding/Modules
# Ensure you follow our structure
# Set working directory

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))  
print(getwd())  

run_pseudotime_analysis <- function(train_umap_path, test_umap_path, output_prefix) {

  plot_width <- 2.5
  plot_height <- 2
  output_dir <- "../../Supervised-Longitudinal-Progressive-Embedding/Temp Files/"
  
  standard_theme <- theme_minimal() +
    theme(
      plot.title = element_text(hjust = 1, size = 8, face = "bold"),
      axis.title = element_text(size = 8, face = "bold"), 
      axis.text = element_text(size = 8),
      legend.title = element_text(size = 8, face = "bold"),
      legend.text = element_text(size = 8, face = "bold"),
      legend.position = "right",
      legend.key.width = unit(0.1, "cm"),
      legend.key.height = unit(0.2, "cm"),
      axis.line.x = element_line(color = "black", linewidth = 0.1),
      axis.line.y = element_line(color = "black", linewidth = 0.1),
      axis.ticks = element_line(color = "black", linewidth = 0.1),
      axis.ticks.length = unit(0.05, "cm")
    )
  
  Training_set <- read.csv(train_umap_path)
  Testing_set <- read.csv(test_umap_path)
  
  colors <- rev(colorRampPalette(brewer.pal(11, 'Spectral')[-6])(100))
                breaks <- seq(0, 1, length.out = 101)
                
                # ----- Data Preparation  -----
                Training_set$DXGrp <- factor(Training_set$DXGrp,
                                             levels = c(1, 2, 3, 4),
                                             labels = c("CN", "EMCI", "LMCI", "AD"))
                Testing_set$DXGrp <- factor(Testing_set$DXGrp,
                                            levels = c(1, 2, 3, 4),
                                            labels = c("CN", "EMCI", "LMCI", "AD"))
                
                # ----- Slingshot Analysis  -----
                rd_train <- as.matrix(Training_set[, c("UMAP1", "UMAP2")])
                sds_train <- slingshot(rd_train, clusterLabels = Training_set$DXGrp, start.clus = "CN")
                
                train_pseudotime <- slingPseudotime(sds_train)
                Training_set$SlingFusedPhatePseu <- rowMeans(train_pseudotime, na.rm = TRUE)
                min_train <- min(Training_set$SlingFusedPhatePseu, na.rm = TRUE)
                max_train <- max(Training_set$SlingFusedPhatePseu, na.rm = TRUE)
                Training_set$Pseudotime_Normalized <- (Training_set$SlingFusedPhatePseu - min_train) / (max_train - min_train)
                
                rd_test <- as.matrix(Testing_set[, c("UMAP1", "UMAP2")])
                sds_test <- predict(sds_train, rd_test)
                test_pseudotime <- slingPseudotime(sds_test)
                Testing_set$SlingFusedPhatePseu <- rowMeans(test_pseudotime, na.rm = TRUE)
                Testing_set$Pseudotime_Normalized <- (Testing_set$SlingFusedPhatePseu - min_train) / (max_train - min_train)
                
                Training_set$Color <- colors[cut(Training_set$Pseudotime_Normalized, breaks = breaks, include.lowest = TRUE)]
                Testing_set$Color <- colors[cut(Testing_set$Pseudotime_Normalized, breaks = breaks, include.lowest = TRUE)]
                
                Training_set$DX <- Training_set$DXGrp  
                Testing_set$DX <- Testing_set$DXGrp
                
                # ----- Segment Preparation  -----
                set.seed(42)
                prepare_segments <- function(data) {
                  segments <- data %>%
                    arrange(RID, EXAMDATE) %>%
                    group_by(RID) %>%
                    mutate(
                      next_UMAP1 = lead(UMAP1),
                      next_UMAP2 = lead(UMAP2),
                      next_RID   = lead(RID)
                    ) %>%
                    filter(RID == next_RID) %>%
                    ungroup() %>%
                    select(-next_RID) %>%
                    na.omit()
                  return(segments)
                }
                
                # ----- Subject Selection  -----
                train_subject_counts <- Training_set %>%
                  group_by(RID) %>%
                  summarize(count = n()) %>%
                  filter(count > 2)
                
                test_subject_counts <- Testing_set %>%
                  group_by(RID) %>%
                  summarize(count = n()) %>%
                  filter(count > 2)
                
                train_subjects_by_dx <- Training_set %>%
                  filter(RID %in% train_subject_counts$RID) %>%
                  group_by(RID) %>%
                  summarize(DXGrp = first(DXGrp)) %>%
                  group_by(DXGrp) %>%
                  summarize(subjects = list(RID))
                
                selected_train_subjects <- c()
                for(dx in c("CN", "EMCI", "LMCI", "AD")) {
                  dx_subjects <- train_subjects_by_dx$subjects[train_subjects_by_dx$DXGrp == dx]
                  if(length(dx_subjects) > 0 && !is.null(dx_subjects[[1]])) {
                    selected <- sample(dx_subjects[[1]], min(3, length(dx_subjects[[1]])))
                    selected_train_subjects <- c(selected_train_subjects, selected)
                  }
                }
                
                no_of_subjects = 10
                
                if(length(selected_train_subjects) < no_of_subjects) {
                  remaining_subjects <- setdiff(train_subject_counts$RID, selected_train_subjects)
                  additional_subjects <- sample(remaining_subjects, 
                                                min(no_of_subjects - length(selected_train_subjects), length(remaining_subjects)))
                  selected_train_subjects <- c(selected_train_subjects, additional_subjects)
                }
                
                test_subjects_by_dx <- Testing_set %>%
                  filter(RID %in% test_subject_counts$RID) %>%
                  group_by(RID) %>%
                  summarize(DXGrp = first(DXGrp)) %>%
                  group_by(DXGrp) %>%
                  summarize(subjects = list(RID))
                
                selected_test_subjects <- c()
                for(dx in c("CN", "EMCI", "LMCI", "AD")) {
                  dx_subjects <- test_subjects_by_dx$subjects[test_subjects_by_dx$DXGrp == dx]
                  if(length(dx_subjects) > 0 && !is.null(dx_subjects[[1]])) {
                    selected <- sample(dx_subjects[[1]], min(3, length(dx_subjects[[1]])))
                    selected_test_subjects <- c(selected_test_subjects, selected)
                  }
                }
                
                if(length(selected_test_subjects) < no_of_subjects) {
                  remaining_subjects <- setdiff(test_subject_counts$RID, selected_test_subjects)
                  additional_subjects <- sample(remaining_subjects, 
                                                min(no_of_subjects - length(selected_test_subjects), length(remaining_subjects)))
                  selected_test_subjects <- c(selected_test_subjects, additional_subjects)
                }
                
                # ----- Data Filtering  -----
                selected_train_data <- Training_set %>% 
                  filter(RID %in% selected_train_subjects)
                selected_test_data <- Testing_set %>% 
                  filter(RID %in% selected_test_subjects)
                
                highlight_data_train <- prepare_segments(selected_train_data)
                highlight_data_test <- prepare_segments(selected_test_data)
                
                background_train_data <- Training_set %>% 
                  filter(!(RID %in% selected_train_subjects))
                background_test_data <- Testing_set %>% 
                  filter(!(RID %in% selected_test_subjects))
                
                # ----- Plot Limits  -----
                umap1_limits <- range(c(Training_set$UMAP1, Testing_set$UMAP1), na.rm = TRUE)
                umap2_limits <- range(c(Training_set$UMAP2, Testing_set$UMAP2), na.rm = TRUE)
                umap1_buffer <- diff(umap1_limits) * 0.05
                umap2_buffer <- diff(umap2_limits) * 0.05
                umap1_limits <- c(umap1_limits[1] - umap1_buffer, umap1_limits[2] + umap1_buffer)
                umap2_limits <- c(umap2_limits[1] - umap2_buffer, umap2_limits[2] + umap2_buffer)
                
                # ----- Plot Function -----
                create_plot <- function(highlight_data, background_data, color_by, title,
                                        is_categorical = FALSE, file_suffix) {
                  p <- ggplot() +
                    coord_cartesian(xlim = umap1_limits, ylim = umap2_limits)
                  
                  if (is_categorical) {
                    p <- p +
                      geom_point(data = background_data, aes(UMAP1, UMAP2), color = "gray80", alpha = 0.3, size = 0.8) +
                      geom_point(data = highlight_data, aes(UMAP1, UMAP2, color = !!sym(color_by)), alpha = 0.9, size = 1.2) +
                      scale_color_manual(
                        values = c("CN" = "#F8766D", "EMCI" = "#7CAE00", "LMCI" = "#00BFC4", "AD" = "#C77CFF"),
                        name = "Diagnosis\nGroup"
                      )
                  } else {
                    p <- p +
                      geom_point(data = background_data, aes(UMAP1, UMAP2), color = "gray80", alpha = 0.3, size = 0.8) +
                      geom_point(data = highlight_data, aes(UMAP1, UMAP2, color = !!sym(color_by)), alpha = 0.9, size = 1.2) +
                      scale_color_gradientn(colors = colors, name = "Pseudotime\nStaging Score", limits = c(0, 1))
                  }
                  
                  p <- p +
                    geom_path(data = as.data.frame(slingCurves(sds_train)[[1]]$s),
                              aes(UMAP1, UMAP2), color = "black", linewidth = 0.3, alpha = 0.7) +
                    geom_segment(data = highlight_data,
                                 aes(x = UMAP1, y = UMAP2, xend = next_UMAP1, yend = next_UMAP2),
                                 color = "black", size = 0.3, alpha = 0.9,
                                 arrow = arrow(type = "closed", length = unit(0.03, "inches"))) +
                    labs(x = "UMAP1", y = "UMAP2", title = title) +
                    standard_theme +
                    theme(
                      panel.grid = element_blank(),
                      text = element_text(color = "black"),
                      axis.text = element_text(color = "black"),
                      legend.position = "right",
                      legend.key.size = unit(0.3, "cm"),
                      legend.spacing = unit(0.1, "cm"),
                      legend.margin = margin(0, 0, 0, 0)
                    )
                  
                  file_path <- paste0(output_dir, output_prefix, "_", file_suffix)
                  ggsave(file_path, p, width = plot_width, height = plot_height, dpi = 300)
                  return(file_path)
                }
                
                # ----- Generate All Plots -----
                generated_files <- list()
                
                # Training Plots
                generated_files$train_pseudotime <- create_plot(
                  highlight_data_train, background_train_data,
                  "Pseudotime_Normalized", "Mapped By Pseudotime", FALSE, "Plot1_train_pseudotime.png")
                
                generated_files$train_dx <- create_plot(
                  highlight_data_train, background_train_data,
                  "DXGrp", "Mapped By Diagnosis Group", TRUE, "Plot2_train_diagnosis.png")
                
                # Testing Plots
                generated_files$test_pseudotime <- create_plot(
                  highlight_data_test, background_test_data,
                  "Pseudotime_Normalized", "Mapped By Pseudotime", FALSE, "Plot1_test_pseudotime.png")
                
                generated_files$test_dx <- create_plot(
                  highlight_data_test, background_test_data,
                  "DXGrp", "Mapped By Diagnosis Group", TRUE, "Plot2_test_diagnosis.png")
                
                # ----- Boxplots  -----
                generate_boxplot <- function(data, file_suffix) {
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
                    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
                    standard_theme +
                    theme(
                      panel.grid = element_blank(),
                      axis.text.x = element_text(angle = 45, hjust = 1, color = "black", size = 8, face = "bold"), # Explicit black color
                      axis.text.y = element_text(color = "black", size = 8, face = "bold"), # Explicit black color
                      plot.title = element_text(hjust = 0.5),
                      plot.margin = margin(10, 10, 10, 10, "pt")
                    ) +
                    guides(color = "none", fill = guide_legend(override.aes = list(color = NA)))
                  
                  file_path <- paste0(output_dir, output_prefix, "_", file_suffix)
                  ggsave(file_path, p, width = plot_width, height = plot_height, dpi = 300)
                  file_path
                }
                
                generated_files$train_boxplot <- generate_boxplot(Training_set, "Plot3_train_boxplot.png")
                generated_files$test_boxplot <- generate_boxplot(Testing_set, "Plot3_test_boxplot.png")
                
                # ----- Save Data  -----
                write.csv(Training_set[, c("UMAP1", "UMAP2", "DXGrp", "DX", "RID", "AGE", "VISCODE2", "SlingFusedPhatePseu", "Pseudotime_Normalized")],
                          file = paste0(output_dir, output_prefix, "_Pseudotime_train.csv"),
                          row.names = FALSE)
                
                write.csv(Testing_set[, c("UMAP1", "UMAP2", "DXGrp", "DX", "RID", "AGE", "VISCODE2", "SlingFusedPhatePseu", "Pseudotime_Normalized")],
                          file = paste0(output_dir, output_prefix, "_Pseudotime_test.csv"),
                          row.names = FALSE)
                
                # Return all results
                list(
                  training_data = Training_set,
                  testing_data = Testing_set,
                  generated_files = generated_files,
                  sds_train = sds_train,
                  sds_test = sds_test
                )
}


data_dir <- "../../Supervised-Longitudinal-Progressive-Embedding/Embeddings"

# For SLOPE analysis
slope_results <- run_pseudotime_analysis(
  file.path(data_dir, "SLOPE_train_umap.csv"), 
  file.path(data_dir, "SLOPE_test_umap.csv"),
  "SLOPE"
)


# For Autoencoder analysis
ae_results <- run_pseudotime_analysis(
  file.path(data_dir, "Autoencoder_train_umap.csv"), 
  file.path(data_dir, "Autoencoder_test_umap.csv"),
  "Autoencoder"
)