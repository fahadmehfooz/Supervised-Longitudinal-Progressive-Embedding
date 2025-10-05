library(ggplot2)
library(dplyr)
library(ggpubr)
library(RColorBrewer)
library(slingshot)
library(fields)
library(cowplot)
library(grid)
library(gridExtra)
library(png)

# Working directory needs to be ->/Supervised-Longitudinal-Progressive-Embedding/Modules
# Ensure you follow the structure
# Set working directory

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
print(getwd())
output_dir <- "../../Supervised-Longitudinal-Progressive-Embedding/Temp Files/"

# Function to prepare segments for visualization
prepare_segments <- function(data) {
  # Ensure data is arranged properly by RID and EXAMDATE
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
  
  all_points <- data %>%
    filter(RID %in% unique(segments$RID))
  
  return(list(
    segments = segments,
    points = all_points
  ))
}

select_subjects_by_dx <- function(data, n_per_group = 5) {
  # Last diagnosis for each subject
  last_dx_per_subject <- data %>%
    group_by(RID) %>%
    arrange(EXAMDATE) %>%
    slice_tail(n = 1) %>%
    select(RID, DX) %>%
    ungroup()
  
  # Subjects with > 2 timepoints
  subjects_multi <- data %>%
    group_by(RID) %>%
    summarise(count = n()) %>%
    filter(count > 2) %>%
    pull(RID)
  
  # Subjects with > 2 timepoints and their last diagnosis
  eligible_subjects <- last_dx_per_subject %>%
    filter(RID %in% subjects_multi)
  
  # Initialize empty vector for selected subjects
  selected <- c()
  
  # Unique diagnosis groups in the data
  dx_groups <- unique(eligible_subjects$DX)
  
  for (dx in dx_groups) {
    dx_subjects <- eligible_subjects %>%
      filter(DX == dx) %>%
      pull(RID)
    
    # If we have enough subjects for this group
    if (length(dx_subjects) >= n_per_group) {
      selected <- c(selected, sample(dx_subjects, n_per_group))
    } else {
      # If not enough subjects, take all available
      selected <- c(selected, dx_subjects)
      warning(paste("Only", length(dx_subjects), "subjects available for", dx, "group"))
    }
  }
  
  return(selected)
}

run_pseudotime_analysis <- function(train_umap_path, test_umap_path, output_prefix) {
  plot_width <- 2
  plot_height <- 2
  
  legend_position = "bottom"
  standard_theme <- theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 8, face = "bold"),
      axis.title = element_text(size = 8, face = "bold"),
      axis.text = element_text(size = 8),
      legend.title = element_text(size = 8, face = "bold"),
      legend.text = element_text(size = 8),
      legend.position = "bottom", 
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
  
  
  # ----- Subject Selection for plotting -----
  set.seed(50)
  
  # Applying to training set
  selected_train_subjects <- select_subjects_by_dx(Training_set)
  
  # Applying to testing set
  selected_test_subjects <- select_subjects_by_dx(Testing_set)
  
  print(selected_test_subjects)
  
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
  
  # -----Plot Function -----
  create_plot <- function(highlight_data, background_data, color_by, title,
                          is_categorical = FALSE, file_suffix) {
    
    # Extracting components from highlight_data
    highlight_segments <- highlight_data$segments
    highlight_points <- highlight_data$points
    
    p <- ggplot() +
      coord_cartesian(xlim = umap1_limits, ylim = umap2_limits)
    
    if (!is_categorical) {
      p <- p +
        geom_point(data = background_data, aes(UMAP1, UMAP2, color = !!sym(color_by)), alpha = 0.9, size = 0.5) +
        geom_point(data = highlight_points, aes(UMAP1, UMAP2, color = !!sym(color_by)), alpha = 0.9, size = 0.5) +
        scale_color_gradientn(
          colors = colors, 
          name = "Pseudotime",
          limits = c(0, 1),
          breaks = c(0, 1),
          labels = c("0", "1")
        ) +
        guides(color = guide_colorbar(
          ticks = FALSE,
          frame.colour = NA, 
          title.position = "top",
          title.hjust = 0.5,
          direction = "horizontal",
          barwidth = unit(2, "cm"),
          barheight = unit(0.2, "cm")
        ))
    }
    
    else {
      # Separate data by diagnosis groups
      background_emci <- background_data %>% filter(!!sym(color_by) == "EMCI")
      background_lmci <- background_data %>% filter(!!sym(color_by) == "LMCI")
      background_ad <- background_data %>% filter(!!sym(color_by) == "AD")
      background_cn <- background_data %>% filter(!!sym(color_by) == "CN")
      
      highlight_emci <- highlight_points %>% filter(!!sym(color_by) == "EMCI")
      highlight_lmci <- highlight_points %>% filter(!!sym(color_by) == "LMCI")
      highlight_ad <- highlight_points %>% filter(!!sym(color_by) == "AD")
      highlight_cn <- highlight_points %>% filter(!!sym(color_by) == "CN")
      
      p <- p +
        # Plot EMCI first
        geom_point(data = background_emci, aes(UMAP1, UMAP2, color = !!sym(color_by)), alpha = 0.9, size = 0.5) +
        geom_point(data = highlight_emci, aes(UMAP1, UMAP2, color = !!sym(color_by)), alpha = 0.9, size = 0.5) +
        # Plot LMCI second
        geom_point(data = background_lmci, aes(UMAP1, UMAP2, color = !!sym(color_by)), alpha = 0.9, size = 0.5) +
        geom_point(data = highlight_lmci, aes(UMAP1, UMAP2, color = !!sym(color_by)), alpha = 0.9, size = 0.5) +
        # Plot AD third (on top of EMCI/LMCI)
        geom_point(data = background_ad, aes(UMAP1, UMAP2, color = !!sym(color_by)), alpha = 0.9, size = 0.5) +
        geom_point(data = highlight_ad, aes(UMAP1, UMAP2, color = !!sym(color_by)), alpha = 0.9, size = 0.5) +
        # Plot CN last (on top of all others)
        geom_point(data = background_cn, aes(UMAP1, UMAP2, color = !!sym(color_by)), alpha = 0.9, size = 0.5) +
        geom_point(data = highlight_cn, aes(UMAP1, UMAP2, color = !!sym(color_by)), alpha = 0.9, size = 0.5) +
        scale_color_manual(
          values = c("CN" = "#F8766D", "EMCI" = "#7CAE00", "LMCI" = "#00BFC4", "AD" = "#C77CFF"),
          name = NULL
        )
    }
    
    p <- p +
      geom_path(data = as.data.frame(slingCurves(sds_train)[[1]]$s),
                aes(UMAP1, UMAP2), color = "black", linewidth = 0.3, alpha = 0.7) +
      labs(x = "UMAP1", y = "UMAP2") +
      standard_theme +
      theme(
        panel.grid = element_blank(),
        text = element_text(color = "black"),
        axis.text = element_text(color = "black"),
        legend.key.size = unit(0.3, "cm"),
        legend.spacing = unit(0.1, "cm"),
        legend.margin = margin(0, 0, 0, 0),
        legend.box.margin = margin(-10, 0, 0, 0),
        legend.justification = "center"
      )
    
    if (is_categorical) {
      p <- p + theme(
        legend.direction = "vertical",
        legend.position = c(0.5, 0.2),
        legend.title = element_text(size = 8, face = "bold"),
        legend.text = element_text(size = 8, face = "bold")
      )
    } else {
      p <- p + theme(
        legend.direction = "horizontal",
        legend.position = c(0.5, 0.1),
        legend.title.position = "top",
        legend.title = element_text(size = 8, face = "bold"),
        legend.text = element_text(size = 8, face = "bold"),
        legend.box = "horizontal",
        legend.box.just = "center",
        legend.key.width = unit(0.5, "cm"),
        legend.key.height = unit(0.2, "cm")
      )
    }
    
    file_path <- paste0(output_dir, output_prefix, "_", file_suffix)
    ggsave(file_path, p, width = plot_width, height = plot_height, dpi = 300)
    return(file_path)
  }
  
  # ----- Generating All Plots -----
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
  generate_boxplot <-   function(data, file_suffix) {
    my_comparisons <- list(c("CN", "EMCI"), c("EMCI", "LMCI"), c("LMCI", "AD"))
    
    p <- ggboxplot(data, x = "DX", y = "Pseudotime_Normalized",
                   fill = "DX", outlier.size = 0.001) +
      labs(y = "Pseudotime", x = NULL) + 
      scale_y_continuous(expand = expansion(mult = c(0, 0.2))) + 
      standard_theme +
      theme(
        panel.grid = element_blank(),
        axis.text.x = element_text(color = "black", size = 8, face = "bold", angle = 45, hjust = 1),
        axis.text.y = element_text(color = "black", size = 8, face = "bold"),
        plot.margin = margin(10, 10, 10, 10, "pt"),
        legend.position = "none"
      )
    
    format_pvalue_scientific <- function(p_val) {
      if (p_val >= 0.01) {
        return(paste0("p = ", sprintf("%.3f", p_val)))
      } else {
        sci_notation <- sprintf("%.1e", p_val)
        parts <- strsplit(sci_notation, "e")[[1]]
        mantissa <- as.numeric(parts[1])
        exponent <- as.integer(parts[2])
        
        return(paste0("p = ", sprintf("%.1f", mantissa), "e", sprintf("%+03d", exponent)))
      }
    }
    
    y_max <- max(data$Pseudotime_Normalized, na.rm = TRUE)
    y_range <- diff(range(data$Pseudotime_Normalized, na.rm = TRUE))
    
    for (i in seq_along(my_comparisons)) {
      comparison <- my_comparisons[[i]]
      group1_data <- data[data$DX == comparison[1], "Pseudotime_Normalized"]
      group2_data <- data[data$DX == comparison[2], "Pseudotime_Normalized"]
      t_test_result <- t.test(group1_data, group2_data)
      p_val <- t_test_result$p.value
      
      p_label <- format_pvalue_scientific(p_val)
      
      x1 <- which(levels(factor(data$DX)) == comparison[1])
      x2 <- which(levels(factor(data$DX)) == comparison[2])
      y_position <- y_max + y_range * (0.05 + 0.08 * (i - 1))
      
      p <- p + 
        annotate("segment", x = x1, xend = x2, y = y_position, yend = y_position, 
                 color = "black", size = 0.2) +
        annotate("segment", x = x1, xend = x1, y = y_position, yend = y_position - y_range * 0.02, 
                 color = "black", size = 0.2) +
        annotate("segment", x = x2, xend = x2, y = y_position, yend = y_position - y_range * 0.02, 
                 color = "black", size = 0.2) +
        annotate("text", x = (x1 + x2)/2, y = y_position + y_range * 0.035, 
                 label = p_label, size = 1.4, fontface = "bold")
    }
    
    file_path <- paste0(output_dir, output_prefix, "_", file_suffix)
    ggsave(file_path, p, width = plot_width, height = plot_height, dpi = 300)
    return(file_path)
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
  "SLOPE")

# For Autoencoder analysis
ae_results <- run_pseudotime_analysis(
  file.path(data_dir, "Autoencoder_train_umap.csv"), 
  file.path(data_dir, "Autoencoder_test_umap.csv"),
  "Autoencoder")

# Function to combine plots
combine_plots <- function(model_name, data_type) {
  plot1_path <- paste0(output_dir, model_name, "_Plot2_", data_type, "_diagnosis.png")
  plot2_path <- paste0(output_dir, model_name, "_Plot1_", data_type, "_pseudotime.png")
  plot3_path <- paste0(output_dir, model_name, "_Plot3_", data_type, "_boxplot.png")
  
  plot_width <- 2
  plot_height <- 2
  
  output_path <- paste0(output_dir, model_name, "_", data_type, "_plots.png")
  png(output_path, width = plot_width * 3, height = plot_height, units = "in", res = 300)
  
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(1, 3)))
  
  # Plot 1
  pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 1))
  img1 <- readPNG(plot1_path)
  grid.raster(img1, width = unit(1, "npc"), height = unit(1, "npc"))
  grid.text("(a)", x = unit(0.05, "npc"), y = unit(0.95, "npc"),
            just = c("left", "top"), gp = gpar(fontface = "bold", fontsize = 7))
  popViewport()
  
  # Plot 2
  pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 2))
  img2 <- readPNG(plot2_path)
  grid.raster(img2, width = unit(1, "npc"), height = unit(1, "npc"))
  grid.text("(b)", x = unit(0.05, "npc"), y = unit(0.95, "npc"),
            just = c("left", "top"), gp = gpar(fontface = "bold", fontsize = 7))
  popViewport()
  
  # Plot 3
  pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 3))
  img3 <- readPNG(plot3_path)
  grid.raster(img3, width = unit(1, "npc"), height = unit(1, "npc"))
  grid.text("(c)", x = unit(0.05, "npc"), y = unit(0.95, "npc"),
            just = c("left", "top"), gp = gpar(fontface = "bold", fontsize = 7))
  popViewport()
  
  dev.off()
  
  return(output_path)
}

# --------------------------------------SUPERVISED--------------------------------------------

train_lr <- read.csv(file.path(data_dir, "train_df_lr.csv"))
train_en <- read.csv(file.path(data_dir, "train_df_en.csv"))  
train_mlp <- read.csv(file.path(data_dir, "train_df_mlp.csv"))
test_lr <- read.csv(file.path(data_dir, "test_df_lr.csv"))
test_en <- read.csv(file.path(data_dir, "test_df_en.csv"))
test_mlp <- read.csv(file.path(data_dir, "test_df_mlp.csv"))


train_lr$DX <- factor(train_lr$DXGrp, 
                      levels = c(1, 2, 3, 4),
                      labels = c("CN", "EMCI", "LMCI", "AD"))
train_en$DX <- factor(train_en$DXGrp, 
                      levels = c(1, 2, 3, 4),
                      labels = c("CN", "EMCI", "LMCI", "AD"))
train_mlp$DX <- factor(train_mlp$DXGrp, 
                       levels = c(1, 2, 3, 4),
                       labels = c("CN", "EMCI", "LMCI", "AD"))
test_lr$DX <- factor(test_lr$DXGrp, 
                     levels = c(1, 2, 3, 4),
                     labels = c("CN", "EMCI", "LMCI", "AD"))
test_en$DX <- factor(test_en$DXGrp, 
                     levels = c(1, 2, 3, 4),
                     labels = c("CN", "EMCI", "LMCI", "AD"))
test_mlp$DX <- factor(test_mlp$DXGrp, 
                      levels = c(1, 2, 3, 4),
                      labels = c("CN", "EMCI", "LMCI", "AD"))

generate_boxplot_supervised <- function(data, model_type, dataset_type, show_x_title = FALSE) {
  legend_position = "bottom"
  standard_theme <- theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 8, face = "bold"),
      axis.title = element_text(size = 8, face = "bold"),
      axis.text = element_text(size = 8),
      legend.title = element_text(size = 8, face = "bold"),
      legend.text = element_text(size = 8),
      legend.position = "bottom", 
      legend.key.width = unit(0.1, "cm"),
      legend.key.height = unit(0.2, "cm"),
      axis.line.x = element_line(color = "black", linewidth = 0.1),
      axis.line.y = element_line(color = "black", linewidth = 0.1),
      axis.ticks = element_line(color = "black", linewidth = 0.1),
      axis.ticks.length = unit(0.05, "cm")
    )
  
  my_colors <- c("CN" = "#F8766D", "EMCI" = "#7CAE00", "LMCI" = "#00BFC4", "AD" = "#C77CFF")
  my_comparisons <- list(c("CN", "EMCI"), c("EMCI", "LMCI"), c("LMCI", "AD"))
  
  x_title <- if(show_x_title) "Diagnosis Groups" else NULL
  
  p <- ggboxplot(data, x = "DX", y = "Pseudotime_Normalized",
                 fill = "DX", outlier.size = 0.001) +
    labs(title = paste0(model_type), y = NULL, x = x_title) +  
    scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +  
    standard_theme +
    theme(
      panel.grid = element_blank(),
      axis.text.x = element_text(color = "black", size = 8, face = "bold", angle = 45, hjust = 1),
      axis.text.y = element_text(color = "black", size = 8, face = "bold"),
      plot.margin = margin(10, 10, 10, 10, "pt"),
      legend.position = "none"
    )
  
  format_pvalue_scientific <- function(p_val) {
    if (p_val >= 0.01) {
      return(paste0("p = ", sprintf("%.3f", p_val)))
    } else {
      sci_notation <- sprintf("%.1e", p_val)
      parts <- strsplit(sci_notation, "e")[[1]]
      mantissa <- as.numeric(parts[1])
      exponent <- as.integer(parts[2])
      
      return(paste0("p = ", sprintf("%.1f", mantissa), "e", sprintf("%+03d", exponent)))
    }
  }
  
  y_max <- max(data$Pseudotime_Normalized, na.rm = TRUE)
  y_range <- diff(range(data$Pseudotime_Normalized, na.rm = TRUE))
  
  for (i in seq_along(my_comparisons)) {
    comparison <- my_comparisons[[i]]
    group1_data <- data[data$DX == comparison[1], "Pseudotime_Normalized"]
    group2_data <- data[data$DX == comparison[2], "Pseudotime_Normalized"]
    t_test_result <- t.test(group1_data, group2_data)
    p_val <- t_test_result$p.value
    
    p_label <- format_pvalue_scientific(p_val)
    
    x1 <- which(levels(factor(data$DX)) == comparison[1])
    x2 <- which(levels(factor(data$DX)) == comparison[2])
    y_position <- y_max + y_range * (0.05 + 0.08 * (i - 1))
    
    p <- p + 
      annotate("segment", x = x1, xend = x2, y = y_position, yend = y_position, 
               color = "black", size = 0.2) +
      annotate("segment", x = x1, xend = x1, y = y_position, yend = y_position - y_range * 0.02, 
               color = "black", size = 0.2) +
      annotate("segment", x = x2, xend = x2, y = y_position, yend = y_position - y_range * 0.02, 
               color = "black", size = 0.2) +
      annotate("text", x = (x1 + x2)/2, y = y_position + y_range * 0.035, 
               label = p_label, size = 1.4, fontface = "bold")
  }
  
  return(p)
}


supervised_objects <- list(
  train_lr = generate_boxplot_supervised(train_lr, "Logistic Regression", "Train"),
  train_en = generate_boxplot_supervised(train_en, "Elastic Net", "Train"),
  train_mlp = generate_boxplot_supervised(train_mlp, "MLP", "Train"),
  test_lr = generate_boxplot_supervised(test_lr, "Logistic Regression", "Test"),
  test_en = generate_boxplot_supervised(test_en, "Elastic Net", "Test"),
  test_mlp = generate_boxplot_supervised(test_mlp, "MLP", "Test")
)

# Create combined plots for supervised models
Supervised_train_plots <- ggarrange(
  supervised_objects$train_lr,
  supervised_objects$train_en + theme(axis.text.y = element_blank()),
  supervised_objects$train_mlp + theme(axis.text.y = element_blank()),
  nrow = 1,
  labels = c("(a)", "(b)", "(c)"),
  font.label = list(size = 4, face = "bold"),
  legend = "none"
) %>% 
  annotate_figure(
    left = textGrob("Normalized Logits", rot = 90, gp = gpar(fontsize = 8, fontface = "bold")),
    bottom = textGrob("Diagnosis Groups", gp = gpar(fontsize = 8, fontface = "bold"))
  )

Supervised_test_plots <- ggarrange(
  supervised_objects$test_lr,
  supervised_objects$test_en + theme(axis.text.y = element_blank()),
  supervised_objects$test_mlp + theme(axis.text.y = element_blank()),
  nrow = 1,
  labels = c("(a)", "(b)", "(c)"),
  font.label = list(size = 4, face = "bold"),
  legend = "none"
) %>% 
  annotate_figure(
    left = textGrob("Normalized Logits", rot = 90, gp = gpar(fontsize = 8, fontface = "bold")),
    bottom = textGrob("Diagnosis Groups", gp = gpar(fontsize = 8, fontface = "bold"))
  )

ggsave(file.path(output_dir, "Supervised_train_plots.png"), Supervised_train_plots,
       width = 5, height = 2, dpi = 300, bg = "white")
ggsave(file.path(output_dir, "Supervised_test_plots.png"), Supervised_test_plots,
       width = 5, height = 2, dpi = 300, bg = "white")