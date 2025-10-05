library(ggplot2)
library(dplyr)
library(tidyr)
library(cowplot)
library(slingshot)
library(RColorBrewer)
library(grid)
library(gridExtra)
library(png)



setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
print(getwd())
output_dir <- "../../Supervised-Longitudinal-Progressive-Embedding/Temp Files/"


create_fig1 <- function() {
  # SLOPE plots
  slope_train_dx <- paste0(output_dir, "SLOPE_Plot2_train_diagnosis.png")
  slope_train_pseudo <- paste0(output_dir, "SLOPE_Plot1_train_pseudotime.png")
  slope_train_box <- paste0(output_dir, "SLOPE_Plot3_train_boxplot.png")
  
  slope_test_dx <- paste0(output_dir, "SLOPE_Plot2_test_diagnosis.png")
  slope_test_pseudo <- paste0(output_dir, "SLOPE_Plot1_test_pseudotime.png")
  slope_test_box <- paste0(output_dir, "SLOPE_Plot3_test_boxplot.png")
  
  # Autoencoder test plots
  ae_test_dx <- paste0(output_dir, "Autoencoder_Plot2_test_diagnosis.png")
  ae_test_pseudo <- paste0(output_dir, "Autoencoder_Plot1_test_pseudotime.png")
  ae_test_box <- paste0(output_dir, "Autoencoder_Plot3_test_boxplot.png")
  
  # Supervised plots without labels 
  Supervised_no_labels <- ggarrange(
    supervised_objects$test_lr,
    supervised_objects$test_en + theme(axis.text.y = element_blank()),
    supervised_objects$test_mlp + theme(axis.text.y = element_blank()),
    nrow = 1,
    labels = NULL, 
    legend = "none"
  ) %>% 
    annotate_figure(
      left = textGrob("Normalized Logits", rot = 90, gp = gpar(fontsize = 8, fontface = "bold"))
    )
  
  ggsave(paste0(output_dir, "Supervised_no_labels.png"), Supervised_no_labels,
         width = 6, height = 2, dpi = 300, bg = "white")
  
  supervised_plots_path <- paste0(output_dir, "Supervised_no_labels.png")
  
  # Set up dimensions
  plot_width <- 2
  plot_height <- 2
  output_path <- paste0(output_dir, "Fig_1_paper.png")
  
  png(output_path, width = plot_width * 3, height = plot_height * 4, units = "in", res = 300)
  
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(4, 3)))
  
  # Row headers
  pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 1:3))
  grid.text("SLOPE - Training Subjects", x = unit(0.5, "npc"), y = unit(0.98, "npc"),
            just = c("center", "top"), gp = gpar(fontface = "bold", fontsize = 10))
  popViewport()
  
  # Row 1: SLOPE Train Plots (a, b, c)
  # Plot 1,1 - SLOPE Train Diagnosis
  pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 1))
  img1 <- readPNG(slope_train_dx)
  grid.raster(img1, width = unit(1, "npc"), height = unit(0.9, "npc"), y = unit(0.45, "npc"))
  grid.text("(a)", x = unit(0.05, "npc"), y = unit(0.9, "npc"),
            just = c("left", "top"), gp = gpar(fontface = "bold", fontsize = 7))
  popViewport()
  
  # Plot 1,2 - SLOPE Train Pseudotime
  pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 2))
  img2 <- readPNG(slope_train_pseudo)
  grid.raster(img2, width = unit(1, "npc"), height = unit(0.9, "npc"), y = unit(0.45, "npc"))
  grid.text("(b)", x = unit(0.05, "npc"), y = unit(0.9, "npc"),
            just = c("left", "top"), gp = gpar(fontface = "bold", fontsize = 7))
  popViewport()
  
  # Plot 1,3 - SLOPE Train Boxplot
  pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 3))
  img3 <- readPNG(slope_train_box)
  grid.raster(img3, width = unit(1, "npc"), height = unit(0.9, "npc"), y = unit(0.45, "npc"))
  grid.text("(c)", x = unit(0.05, "npc"), y = unit(0.9, "npc"),
            just = c("left", "top"), gp = gpar(fontface = "bold", fontsize = 7))
  popViewport()
  
  # Row 2 header
  pushViewport(viewport(layout.pos.row = 2, layout.pos.col = 1:3))
  grid.text("SLOPE - Test Subjects", x = unit(0.5, "npc"), y = unit(0.98, "npc"),
            just = c("center", "top"), gp = gpar(fontface = "bold", fontsize = 10))
  popViewport()
  
  # Row 2: SLOPE Test Plots (d, e, f)
  # Plot 2,1 - SLOPE Test Diagnosis
  pushViewport(viewport(layout.pos.row = 2, layout.pos.col = 1))
  img4 <- readPNG(slope_test_dx)
  grid.raster(img4, width = unit(1, "npc"), height = unit(0.9, "npc"), y = unit(0.45, "npc"))
  grid.text("(d)", x = unit(0.05, "npc"), y = unit(0.9, "npc"),
            just = c("left", "top"), gp = gpar(fontface = "bold", fontsize = 7))
  popViewport()
  
  # Plot 2,2 - SLOPE Test Pseudotime
  pushViewport(viewport(layout.pos.row = 2, layout.pos.col = 2))
  img5 <- readPNG(slope_test_pseudo)
  grid.raster(img5, width = unit(1, "npc"), height = unit(0.9, "npc"), y = unit(0.45, "npc"))
  grid.text("(e)", x = unit(0.05, "npc"), y = unit(0.9, "npc"),
            just = c("left", "top"), gp = gpar(fontface = "bold", fontsize = 7))
  popViewport()
  
  # Plot 2,3 - SLOPE Test Boxplot
  pushViewport(viewport(layout.pos.row = 2, layout.pos.col = 3))
  img6 <- readPNG(slope_test_box)
  grid.raster(img6, width = unit(1, "npc"), height = unit(0.9, "npc"), y = unit(0.45, "npc"))
  grid.text("(f)", x = unit(0.05, "npc"), y = unit(0.9, "npc"),
            just = c("left", "top"), gp = gpar(fontface = "bold", fontsize = 7))
  popViewport()
  
  # Row 3 header
  pushViewport(viewport(layout.pos.row = 3, layout.pos.col = 1:3))
  grid.text("Autoencoder - Test Subjects", x = unit(0.5, "npc"), y = unit(0.98, "npc"),
            just = c("center", "top"), gp = gpar(fontface = "bold", fontsize = 10))
  popViewport()
  
  # Row 3: Autoencoder Test Plots (g, h, i)
  # Plot 3,1 - Autoencoder Test Diagnosis
  pushViewport(viewport(layout.pos.row = 3, layout.pos.col = 1))
  img7 <- readPNG(ae_test_dx)
  grid.raster(img7, width = unit(1, "npc"), height = unit(0.9, "npc"), y = unit(0.45, "npc"))
  grid.text("(g)", x = unit(0.05, "npc"), y = unit(0.9, "npc"),
            just = c("left", "top"), gp = gpar(fontface = "bold", fontsize = 7))
  popViewport()
  
  # Plot 3,2 - Autoencoder Test Pseudotime
  pushViewport(viewport(layout.pos.row = 3, layout.pos.col = 2))
  img8 <- readPNG(ae_test_pseudo)
  grid.raster(img8, width = unit(1, "npc"), height = unit(0.9, "npc"), y = unit(0.45, "npc"))
  grid.text("(h)", x = unit(0.05, "npc"), y = unit(0.9, "npc"),
            just = c("left", "top"), gp = gpar(fontface = "bold", fontsize = 7))
  popViewport()
  
  # Plot 3,3 - Autoencoder Test Boxplot
  pushViewport(viewport(layout.pos.row = 3, layout.pos.col = 3))
  img9 <- readPNG(ae_test_box)
  grid.raster(img9, width = unit(1, "npc"), height = unit(0.9, "npc"), y = unit(0.45, "npc"))
  grid.text("(i)", x = unit(0.05, "npc"), y = unit(0.9, "npc"),
            just = c("left", "top"), gp = gpar(fontface = "bold", fontsize = 7))
  popViewport()
  
  # Row 4 header
  pushViewport(viewport(layout.pos.row = 4, layout.pos.col = 1:3))
  grid.text("Supervised Learning - Test Subjects", x = unit(0.5, "npc"), y = unit(0.98, "npc"),
            just = c("center", "top"), gp = gpar(fontface = "bold", fontsize = 10))
  popViewport()
  
  # Row 4: Supervised Plots (j, k, l) 
  pushViewport(viewport(layout.pos.row = 4, layout.pos.col = 1:3))
  img10 <- readPNG(supervised_plots_path)
  grid.raster(img10, width = unit(1, "npc"), height = unit(0.9, "npc"), y = unit(0.45, "npc"))
  grid.text("(j)", x = unit(0.02, "npc"), y = unit(0.9, "npc"),
            just = c("left", "top"), gp = gpar(fontface = "bold", fontsize = 7))
  grid.text("(k)", x = unit(0.35, "npc"), y = unit(0.9, "npc"),
            just = c("left", "top"), gp = gpar(fontface = "bold", fontsize = 7))
  grid.text("(l)", x = unit(0.68, "npc"), y = unit(0.9, "npc"),
            just = c("left", "top"), gp = gpar(fontface = "bold", fontsize = 7))
  popViewport()
  
  dev.off()
  
  return(output_path)
}


create_fig2 <- function(violation_data, thresholds) {
  
  # Validate that both parameters are provided
  if (missing(violation_data)) {
    stop("violation_data parameter is required")
  }
  if (missing(thresholds)) {
    stop("thresholds parameter is required")
  }
  
  # Validate input structure
  if (!is.list(violation_data)) {
    stop("violation_data must be a list")
  }
  
  if (!is.numeric(thresholds)) {
    stop("thresholds must be a numeric vector")
  }
  
  # Check that each model has both vio_ratios and vio_gaps
  for (model_name in names(violation_data)) {
    if (!is.list(violation_data[[model_name]])) {
      stop(paste("Data for model", model_name, "must be a list"))
    }
    if (!all(c("vio_ratios", "vio_gaps") %in% names(violation_data[[model_name]]))) {
      stop(paste("Model", model_name, "must have both 'vio_ratios' and 'vio_gaps' components"))
    }
    
    # Validate that arrays match threshold length
    if (length(violation_data[[model_name]]$vio_ratios) != length(thresholds)) {
      stop(paste("vio_ratios for model", model_name, "must have same length as thresholds"))
    }
    if (length(violation_data[[model_name]]$vio_gaps) != length(thresholds)) {
      stop(paste("vio_gaps for model", model_name, "must have same length as thresholds"))
    }
  }
  
  # Plot dimensions and theme
  plot_width <- 4
  plot_height <- 4
  
  standard_theme <- theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 10, face = "bold"),
      axis.title = element_text(size = 9, face = "bold"),
      axis.text = element_text(size = 8),
      legend.title = element_text(size = 8, face = "bold"),
      legend.text = element_text(size = 8),
      legend.position = "bottom", 
      legend.key.width = unit(0.1, "cm"),
      legend.key.height = unit(0.2, "cm"),
      axis.line.x = element_line(color = "black", linewidth = 0.1),
      axis.line.y = element_line(color = "black", linewidth = 0.1),
      axis.ticks = element_line(color = "black", linewidth = 0.1),
      axis.ticks.length = unit(0.05, "cm"),
      panel.grid.major = element_line(color = "grey90"),
      panel.grid.minor = element_line(color = "grey95")
    )
  
  create_df <- function(data_list, metric_name) {
    df_list <- list()
    
    for(model in names(data_list)) {
      df_list[[model]] <- data.frame(
        Model = model,
        Threshold = thresholds,
        Value = data_list[[model]][[ifelse(metric_name == "Violation Ratio", "vio_ratios", "vio_gaps")]]
      )
    }
    
    do.call(rbind, df_list)
  }
  
  vio_ratio_df <- create_df(violation_data, "Violation Ratio")
  vio_gap_df <- create_df(violation_data, "Violation Gap")
  
  # Dynamic color assignment based on models in violation_data
  model_names <- names(violation_data)
  default_colors <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f")
  model_colors <- setNames(default_colors[1:length(model_names)], model_names)
  
  known_colors <- c(
    "SLOPE" = "#1f77b4",
    "Autoencoder" = "#ff7f0e", 
    "Logistic Regression" = "#2ca02c",
    "Elastic Net" = "#d62728",
    "MLP" = "#9467bd"
  )
  
  for (model in names(known_colors)) {
    if (model %in% model_names) {
      model_colors[model] <- known_colors[model]
    }
  }
  
  # ============= PLOT A: VIOLATION RATIO =============
  plot_a <- ggplot(vio_ratio_df, aes(x = Threshold, y = Value, color = Model)) +
    geom_line(size = 1) +
    geom_point(size = 2) +
    scale_color_manual(values = model_colors) +
    labs(
      x = "Threshold",
      y = "Violation Ratio"
    ) +
    standard_theme +
    scale_x_continuous(breaks = thresholds) +
    scale_y_continuous(limits = c(0, max(vio_ratio_df$Value) * 1.1)) +
    theme(
      legend.position = c(0.95, 0.95),  
      legend.justification = c(1, 1),  
      legend.background = element_rect(fill = "white", color = "black", size = 0.5),
      legend.margin = margin(4, 4, 4, 4),
      legend.key.size = unit(0.8, "lines"),
      legend.text = element_text(size = 7),
      legend.title = element_blank()
    )
  
  # ============= PLOT B: VIOLATION GAP =============
  plot_b <- ggplot(vio_gap_df, aes(x = Threshold, y = Value, color = Model)) +
    geom_line(size = 1) +
    geom_point(size = 2) +
    scale_color_manual(values = model_colors) +
    labs(
      x = "Threshold",
      y = "Violation Gap"
    ) +
    standard_theme +
    scale_x_continuous(breaks = thresholds) +
    scale_y_continuous(limits = c(0, max(vio_gap_df$Value) * 1.1)) +
    theme(
      legend.position = c(0.05, 0.95),  # Position legend inside plot (top-left)
      legend.justification = c(0, 1),   # Anchor point of legend
      legend.background = element_rect(fill = "white", color = "black", size = 0.5),
      legend.margin = margin(4, 4, 4, 4),
      legend.key.size = unit(0.8, "lines"),
      legend.text = element_text(size = 7),
      legend.title = element_blank()
    )
  
  # ============= TRAJECTORY DATA PREPARATION =============
  
  data_dir <- "../../Supervised-Longitudinal-Progressive-Embedding/Embeddings"
  Training_set <- read.csv(file.path(data_dir, "SLOPE_train_umap.csv"))
  Testing_set <- read.csv(file.path(data_dir, "SLOPE_test_umap.csv"))
  
  # Data Preparation
  Training_set$DXGrp <- factor(Training_set$DXGrp,
                               levels = c(1, 2, 3, 4),
                               labels = c("CN", "EMCI", "LMCI", "AD"))
  Testing_set$DXGrp <- factor(Testing_set$DXGrp,
                              levels = c(1, 2, 3, 4),
                              labels = c("CN", "EMCI", "LMCI", "AD"))
  
  rd_train <- as.matrix(Training_set[, c("UMAP1", "UMAP2")])
  sds_train <- slingshot(rd_train, clusterLabels = Training_set$DXGrp, start.clus = "CN")
  
  # Segment Preparation
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
  
  target_train_subjects <- c(4489, 4357, 778)
  target_test_subjects <- c(1016, 4974, 4659)
  
  train_subject_counts <- Training_set %>%
    group_by(RID) %>%
    summarize(count = n()) %>%
    filter(count > 1, RID %in% target_train_subjects)
  
  test_subject_counts <- Testing_set %>%
    group_by(RID) %>%
    summarize(count = n()) %>%
    filter(count > 1, RID %in% target_test_subjects)
  
  # Use target subjects that exist in the data
  selected_train_subjects <- train_subject_counts$RID
  selected_test_subjects <- test_subject_counts$RID
  
  print(paste("Training subjects selected:", paste(selected_train_subjects, collapse = ", ")))
  print(paste("Test subjects selected:", paste(selected_test_subjects, collapse = ", ")))
  
  # Data Filtering
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
  
  # Plot Limits
  umap1_limits <- range(c(Training_set$UMAP1, Testing_set$UMAP1), na.rm = TRUE)
  umap2_limits <- range(c(Training_set$UMAP2, Testing_set$UMAP2), na.rm = TRUE)
  umap1_buffer <- diff(umap1_limits) * 0.05
  umap2_buffer <- diff(umap2_limits) * 0.05
  umap1_limits <- c(umap1_limits[1] - umap1_buffer, umap1_limits[2] + umap1_buffer)
  umap2_limits <- c(umap2_limits[1] - umap2_buffer, umap2_limits[2] + umap2_buffer)
  
  create_trajectory_plot <- function(highlight_data, background_data, selected_data, title) {
    
    all_categories <- data.frame(
      UMAP1 = c(NA, NA, NA, NA),
      UMAP2 = c(NA, NA, NA, NA), 
      DXGrp = factor(c("CN", "EMCI", "LMCI", "AD"), levels = c("CN", "EMCI", "LMCI", "AD"))
    )
    
    # Combine with selected data
    plot_data <- rbind(selected_data[,c("UMAP1", "UMAP2", "DXGrp")], all_categories)
    
    p <- ggplot() +
      coord_cartesian(xlim = umap1_limits, ylim = umap2_limits) +
      
      geom_point(data = background_data, aes(UMAP1, UMAP2), 
                 color = "grey70", alpha = 0.3, size = 0.3) +
      
      # Plot invisible points for all categories to force legend
      geom_point(data = all_categories, aes(UMAP1, UMAP2, color = DXGrp), 
                 alpha = 0, size = 0) +
      
      # Make selected points larger and colored by diagnosis
      geom_point(data = selected_data, aes(UMAP1, UMAP2, color = DXGrp), 
                 alpha = 0.9, size = 2) +
      
      scale_color_manual(
        values = c("CN" = "#F8766D", "EMCI" = "#7CAE00", "LMCI" = "#00BFC4", "AD" = "#C77CFF"),
        name = NULL,
        breaks = c("CN", "EMCI", "LMCI", "AD"),
        labels = c("CN", "EMCI", "LMCI", "AD")
      ) +
      
      # Override legend appearance to show all colors
      guides(color = guide_legend(
        override.aes = list(alpha = 1, size = 2),
        title = NULL
      )) +
      
      # Keep the trajectory line
      geom_path(data = as.data.frame(slingCurves(sds_train)[[1]]$s),
                aes(UMAP1, UMAP2), color = "black", linewidth = 0.3, alpha = 0.7) +
      
      # Keep the arrows for selected subjects
      geom_segment(data = highlight_data,
                   aes(x = UMAP1, y = UMAP2, xend = next_UMAP1, yend = next_UMAP2),
                   color = "black", linewidth = 0.5, alpha = 0.9,
                   arrow = arrow(type = "closed", length = unit(0.05, "inches"))) +
      
      labs(x = "UMAP1", y = "UMAP2", title = title) +
      standard_theme +
      theme(
        panel.grid = element_blank(),
        text = element_text(color = "black"),
        axis.text = element_text(color = "black"),
        legend.position = c(0.02, 0.98),
        legend.justification = c(0, 1),
        legend.background = element_rect(fill = "white", color = "black", linewidth = 0.5),
        legend.margin = margin(4, 4, 4, 4),
        legend.key.size = unit(0.8, "lines"),
        legend.text = element_text(size = 7),
        legend.spacing = unit(0.1, "cm"),
        legend.direction = "vertical"
      )
    
    return(p)
  }
  
  # ============= PLOT C: TRAINING TRAJECTORY =============
  plot_c <- create_trajectory_plot(highlight_data_train, background_train_data, selected_train_data, "Training Subjects")
  
  # ============= PLOT D: TEST TRAJECTORY =============  
  plot_d <- create_trajectory_plot(highlight_data_test, background_test_data, selected_test_data, "Test Subjects")
  
  legend <- get_legend(
    plot_a + theme(legend.position = "bottom") +
      guides(color = guide_legend(nrow = 1))
  )
  
  plots_grid <- plot_grid(
    plot_a, plot_b,
    plot_c, plot_d,
    labels = c("(a)", "(b)", "(c)", "(d)"),
    label_size = 12,
    label_fontface = "bold",
    ncol = 2,
    align = "hv"
  )
  
  final_plot <- plot_grid(
    plot_a, plot_b,
    plot_c, plot_d,
    labels = c("(a)", "(b)", "(c)", "(d)"),
    label_size = 12,
    label_fontface = "bold",
    ncol = 2,
    align = "hv"
  )
  
  ggsave("../../Supervised-Longitudinal-Progressive-Embedding/Temp Files/Fig_2_paper.png", final_plot, 
         width = 8, height = 8, dpi = 300, bg = "white")
  
  return(final_plot)
}


fig1_paper <- create_fig1()
print(paste("paper fig 1 saved to:", fig1_paper))



violation_data <- list(
  'SLOPE' = list(
    vio_ratios = c(0.35467980295566504, 0.09359605911330049, 0.03940886699507389, 0.0, 0.0, 0.0),
    vio_gaps = c(0.0414440455254501, 0.09060006511646122, 0.11977073569438804, 0, 0, 0)
  ),
  'Autoencoder' = list(
    vio_ratios = c(0.35960591133004927, 0.15270935960591134, 0.059113300492610835, 0.014778325123152709, 0.0, 0.0),
    vio_gaps = c(0.05057595975699082, 0.09640661112177928, 0.1373930025589684, 0.17659916136783368, 0, 0)
  ),
  'Logistic Regression' = list(
    vio_ratios = c(0.5172413793103449, 0.32019704433497537, 0.16748768472906403, 0.059113300492610835, 0.019704433497536946, 0.009852216748768473),
    vio_gaps = c(0.08252326882299052, 0.11622498193512151, 0.15226536101675245, 0.20405313875866052, 0.25957025021275, 0.3012357949806872)
  ),
  'Elastic Net' = list(
    vio_ratios = c(0.5320197044334976, 0.3251231527093596, 0.18719211822660098, 0.059113300492610835, 0.024630541871921183, 0.009852216748768473),
    vio_gaps = c(0.08318849309387982, 0.11787237539488009, 0.1504983165113827, 0.20577191701646802, 0.24846077871621106, 0.30422637042236256)
  ),
  'MLP' = list(
    vio_ratios = c(0.5960591133004927, 0.15270935960591134, 0.07389162561576355, 0.029556650246305417, 0.014778325123152709, 0.0049261083743842365),
    vio_gaps = c(0.046463406775608534, 0.11183454913477744, 0.1532822608947754, 0.1963732639948527, 0.22787582874298096, 0.26398101449012756)
  )
)

# Thresholds
thresholds <- c(0.0, 0.05, 0.10, 0.15, 0.20, 0.25)


fig2_paper <- create_fig2(violation_data, thresholds)
print(fig2_paper)

file.remove(paste0(output_dir, "SLOPE_Plot1_train_pseudotime.png"))
file.remove(paste0(output_dir, "SLOPE_Plot2_train_diagnosis.png"))
file.remove(paste0(output_dir, "SLOPE_Plot3_train_boxplot.png"))
file.remove(paste0(output_dir, "SLOPE_Plot1_test_pseudotime.png"))
file.remove(paste0(output_dir, "SLOPE_Plot2_test_diagnosis.png"))
file.remove(paste0(output_dir, "SLOPE_Plot3_test_boxplot.png"))
file.remove(paste0(output_dir, "Autoencoder_Plot1_train_pseudotime.png"))
file.remove(paste0(output_dir, "Autoencoder_Plot2_train_diagnosis.png"))
file.remove(paste0(output_dir, "Autoencoder_Plot3_train_boxplot.png"))
file.remove(paste0(output_dir, "Autoencoder_Plot1_test_pseudotime.png"))
file.remove(paste0(output_dir, "Autoencoder_Plot2_test_diagnosis.png"))
file.remove(paste0(output_dir, "Autoencoder_Plot3_test_boxplot.png"))
file.remove(paste0(output_dir, "Supervised_test_plots.png"))
file.remove(paste0(output_dir, "Supervised_train_plots.png"))
file.remove(paste0(output_dir, "Supervised_no_labels.png"))


