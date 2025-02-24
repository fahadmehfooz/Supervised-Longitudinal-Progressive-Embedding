library(ggplot2)
library(dplyr)
library(ggpubr)
library(RColorBrewer)
library(slingshot)
library(fields)


standard_font_size <- 20
bold_font_size <- 25
font_family <- "Arial" 

standard_theme <- theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12),
    panel.grid.major = element_line(color = "gray90"),
    panel.grid.minor = element_blank()
  )



MappingAV45 <- read.csv("/Users/fahadmehfooz/Downloads/train_encodings_umap_dl.csv")
TestMappingAV45 <- read.csv("/Users/fahadmehfooz/Downloads/test_encodings_umap_dl.csv")

colors <- rev(colorRampPalette(brewer.pal(11, 'Spectral')[-6])(100))

MappingAV45$DXGrp <- factor(MappingAV45$DXGrp, 
                            levels = c(1, 2, 3, 4), 
                            labels = c("CN", "EMCI", "LMCI", "AD"))

TestMappingAV45$DXGrp <- factor(TestMappingAV45$DXGrp, 
                                levels = c(1, 2, 3, 4), 
                                labels = c("CN", "EMCI", "LMCI", "AD"))

# Train Slingshot with STARTING CLUSTER as CN
rd_train <- as.matrix(MappingAV45[, c("UMAP1", "UMAP2")])
sds_train <- slingshot(rd_train, clusterLabels = MappingAV45$DXGrp, start.clus = "CN")

# Geting pseudotime for training data
train_pseudotime <- slingPseudotime(sds_train)
MappingAV45$SlingFusedPhatePseu <- rowMeans(train_pseudotime, na.rm = TRUE)

# Normalizing training data
min_train <- min(MappingAV45$SlingFusedPhatePseu, na.rm = TRUE)
max_train <- max(MappingAV45$SlingFusedPhatePseu, na.rm = TRUE)
MappingAV45$Pseudotime_Normalized <- (MappingAV45$SlingFusedPhatePseu - min_train) / (max_train - min_train)

rd_test <- as.matrix(TestMappingAV45[, c("UMAP1", "UMAP2")])
sds_test <- predict(sds_train, rd_test)  # Project onto training curves

# Getting test pseudotime
test_pseudotime <- slingPseudotime(sds_test)
TestMappingAV45$SlingFusedPhatePseu <- rowMeans(test_pseudotime, na.rm = TRUE)
TestMappingAV45$Pseudotime_Normalized <- (TestMappingAV45$SlingFusedPhatePseu - min_train) / (max_train - min_train)

breaks <- seq(0, 1, length.out = 101)
MappingAV45$Color <- colors[cut(MappingAV45$Pseudotime_Normalized, breaks = breaks, include.lowest = TRUE)]
TestMappingAV45$Color <- colors[cut(TestMappingAV45$Pseudotime_Normalized, breaks = breaks, include.lowest = TRUE)]

MappingAV45$DX <- MappingAV45$DXGrp  
TestMappingAV45$DX <- TestMappingAV45$DXGrp


set.seed(42)

segments_data_train <- MappingAV45 %>%
  arrange(RID, AGE) %>%
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

segments_data_test <- TestMappingAV45 %>%
  arrange(RID, AGE) %>%
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


highlight_indices_train <- sample(nrow(segments_data_train), 100)
highlight_data_train <- segments_data_train[highlight_indices_train, ]
remaining_data_train <- segments_data_train[-highlight_indices_train, ]

highlight_indices_test <- sample(nrow(segments_data_test), 60)
highlight_data_test <- segments_data_test[highlight_indices_test, ]
remaining_data_test <- segments_data_test[-highlight_indices_test, ]


#---------------------------Training ------------------------------------

# plotting with getCurves from Slingshot
pto <- getLineages(rd, cl)
pto <- getCurves(pto)
sds <- as.SlingshotDataSet(pto)


dev.new(width=10, height=7.5) 


# Plot 1: Train data vs pseudotime along with principal curve
png("/Users/fahadmehfooz/Desktop/AE/SLOPE/Temp Files/Plot1_train_vs_pseudotime_dl.png")

# Set up the layout
layout(matrix(c(1, 2), ncol = 2), widths = c(4, 1))

par(mar = c(5, 5, 4, 4))

train_plot1 <- ggplot() +  
  geom_segment(
    data = highlight_data_train,
    aes(x = UMAP1, y = UMAP2, xend = next_UMAP1, yend = next_UMAP2),
    color = "black",  # Uniform color for vectors
    size = 0.5, alpha = 1, 
    arrow = arrow(type = "closed", length = unit(0.05, "inches"))  
  ) +  
  geom_segment(
    data = remaining_data_train,
    aes(x = UMAP1, y = UMAP2, xend = next_UMAP1, yend = next_UMAP2),
    color = "gray60",  # Uniform color for vectors
    size = 0.4, alpha = 0.1,
    arrow = arrow(type = "closed", length = unit(0.025, "inches")) 
  ) +  
  geom_path(
    data = as.data.frame(slingCurves(sds_train)[[1]]$s),
    aes(x = UMAP1, y = UMAP2),
    color = "black", linewidth = 0.2
  ) +  
  geom_point(
    data = segments_data_train,
    aes(x = UMAP1, y = UMAP2, color = Pseudotime_Normalized),
    alpha = 0.9, size = 0.7  # Reduced point size
  ) +  
  scale_color_gradientn(colors = colors, name = "Pseudotime", limits = c(0, 1)) +  
  labs(x = "UMAP1", y = "UMAP2", title = "UMAP Embedding Mapped By Pseudotime") +  
  theme_minimal() +  
  theme(
    panel.grid = element_blank(),
    legend.position = "right",
    text = element_text(color = "black"),
    axis.text = element_text(color = "black")
  )

print(train_plot1)


dev.off()


# Plot 2: Boxplot of Pseudotime across different diagnosis groups

my_color1 <- c("#F8766D", "#7CAE00", "#00BFC4", "#C77CFF")
names(my_color1) <- unique(as.character(DXGrp_train))
MappingAV45$DX <- DXGrp_train

my_comparisons = list( c("CN", "EMCI"), c("EMCI", "LMCI"), c("LMCI", "AD") )

dev.new(width=10, height=7.5) 
par(mar = c(5, 5, 4, 5))

png("/Users/fahadmehfooz/Desktop/AE/SLOPE/Temp Files/Plot2_train_boxplot_dl.png")

ggboxplot(MappingAV45, x = "DX", y = "Pseudotime_Normalized",
          fill = "DX",ylab = "Pseudotime",xlab = "Diagnosis Groups")+ 
  stat_compare_means(comparisons = my_comparisons,aes(label = ..p.signif..),
                     method = "t.test", size = 5)+
  ggtitle("Pseudotime Vs Diagnosis Group") +
  theme(standard_theme,  plot.title = element_text(hjust = 0.5))  

dev.off()

# Plot 3: Scatter plot of UMAP1 vs UMAP2 colored by DX Group

dev.new(width=10, height=7.5) 
par(mar = c(5, 5, 4, 5))

png("/Users/fahadmehfooz/Desktop/AE/SLOPE/Temp Files/Plot3_train_scatter_vs_dxg_dl.png")

ggplot() +
  geom_segment(
    data = highlight_data_train,
    aes(x = UMAP1, y = UMAP2, xend = next_UMAP1, yend = next_UMAP2),
    color = "black",
    size = 0.5, alpha = 1,
    arrow = arrow(type = "closed", length = unit(0.05, "inches"))
  ) +
  geom_segment(
    data = remaining_data_train,
    aes(x = UMAP1, y = UMAP2, xend = next_UMAP1, yend = next_UMAP2),
    color = "gray60",
    size = 0.4, alpha = 0.1,
    arrow = arrow(type = "closed", length = unit(0.025, "inches"))
  ) +
  geom_path(
    data = as.data.frame(slingCurves(sds_train)[[1]]$s),
    aes(x = UMAP1, y = UMAP2),
    color = "black", linewidth = 0.2
  ) +
  geom_point(
    data = segments_data_train,
    aes(x = UMAP1, y = UMAP2, color = DXGrp),
    alpha = 0.9, size = 0.7
  ) +
  scale_color_manual(
    values = c(
      "CN" = "#F8766D",
      "EMCI" = "#7CAE00",
      "LMCI" = "#00BFC4",
      "AD" = "#C77CFF"
    ),
    name = "DX Group"
  ) +
  labs(
    x = "UMAP1",
    y = "UMAP2",
    title = "UMAP Embedding Mapped By Diagnosis Group"
  ) +
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    legend.position = "right",
    text = element_text(color = "black"),
    axis.text = element_text(color = "black")
  )



dev.off()

#---------------------------Testing ------------------------------------


dev.new(width=10, height=7.5) 


# Plot 1: Test data vs Pseudotime 
png("/Users/fahadmehfooz/Desktop/AE/SLOPE/Temp Files/Plot1_test_vs_pseudotime_dl.png")

# Set up the layout
layout(matrix(c(1, 2), ncol = 2), widths = c(4, 1))

par(mar = c(5, 5, 4, 4))

test_plot1 <- ggplot() +  
  geom_segment(
    data = highlight_data_test,
    aes(x = UMAP1, y = UMAP2, xend = next_UMAP1, yend = next_UMAP2),
    color = "black",  # Uniform color for vectors
    size = 0.5, alpha = 1,
    arrow = arrow(type = "closed", length = unit(0.05, "inches"))  
  ) +  
  geom_segment(
    data = remaining_data_test,
    aes(x = UMAP1, y = UMAP2, xend = next_UMAP1, yend = next_UMAP2),
    color = "gray60",  # Uniform color for vectors
    size = 0.4, alpha = 0.1,
    arrow = arrow(type = "closed", length = unit(0.025, "inches")) 
  ) +  
  geom_path(
    data = as.data.frame(slingCurves(sds_train)[[1]]$s),
    aes(x = UMAP1, y = UMAP2),
    color = "black", linewidth = 0.2
  ) +  
  geom_point(
    data = segments_data_test,
    aes(x = UMAP1, y = UMAP2, color = Pseudotime_Normalized),
    alpha = 0.9, size = 0.7 
  ) +  
  scale_color_gradientn(colors = colors, name = "Pseudotime", limits = c(0, 1)) +  
  labs(x = "UMAP1", y = "UMAP2", title = "UMAP Embedding Mapped By Pseudotime") +  
  theme_minimal() +  
  standard_theme +
  theme(
    panel.grid = element_blank(),
    legend.position = "right",
    text = element_text(color = "black"),
    axis.text = element_text(color = "black")
  )


print(test_plot1)

dev.off()


# Plot 2: Boxplot of Pseudotime across different diagnosis groups

# Map the color to each diagnostic group
my_color1 <- c("CN" = "#F8766D", "EMCI" = "#7CAE00", "LMCI" = "#00BFC4", "AD" = "#C77CFF")
my_comparisons <- list(c("CN", "EMCI"), c("EMCI", "LMCI"), c("LMCI", "AD"))


dev.new(width=10, height=7.5) 
par(mar = c(5, 5, 4, 5))

png("/Users/fahadmehfooz/Desktop/AE/SLOPE/Temp Files/Plot2_test_boxplot_dl.png")

#Plot 2: Pseudotime across different diagnosis group
ggboxplot(TestMappingAV45, x = "DX", y = "Pseudotime_Normalized",
          fill = "DX",ylab = "Pseudotime",xlab = "Diagnosis Group")+ 
  stat_compare_means(comparisons = my_comparisons,aes(label = ..p.signif..),
                     method = "t.test", size = 5)+
  ggtitle("Pseudotime Vs Diagnosis Group") +
  theme(standard_theme,  plot.title = element_text(hjust = 0.5)) 

dev.off()


# Plot 3: Scatter plot of UMAP1 vs UMAP2 colored by DX Group

dev.new(width=10, height=7.5) 
par(mar = c(5, 5, 4, 5))
png("/Users/fahadmehfooz/Desktop/AE/SLOPE/Temp Files/Plot3_test_scatter_vs_dxg_dl.png")

# Scatter plot using ggplot2
ggplot() +
  # Highlighted segments
  geom_segment(
    data = highlight_data_test,
    aes(x = UMAP1, y = UMAP2, xend = next_UMAP1, yend = next_UMAP2),
    color = "black",
    size = 0.5, alpha = 1,
    arrow = arrow(type = "closed", length = unit(0.05, "inches"))
  ) +
  # Background segments
  geom_segment(
    data = remaining_data_test,
    aes(x = UMAP1, y = UMAP2, xend = next_UMAP1, yend = next_UMAP2),
    color = "gray60",
    size = 0.4, alpha = 0.1,
    arrow = arrow(type = "closed", length = unit(0.025, "inches"))
  ) +
  # Principal curve
  geom_path(
    data = as.data.frame(slingCurves(sds_train)[[1]]$s),
    aes(x = UMAP1, y = UMAP2),
    color = "black", linewidth = 0.2
  ) +
  # Points colored by diagnostic group
  geom_point(
    data = TestMappingAV45,
    aes(x = UMAP1, y = UMAP2, color = DXGrp),
    alpha = 0.9, size = 0.7
  ) +
  scale_color_manual(
    values = c(
      "CN" = "#F8766D",
      "EMCI" = "#7CAE00",
      "LMCI" = "#00BFC4",
      "AD" = "#C77CFF"
    ),
    name = "DX Group"
  ) +
  labs(
    title = "UMAP Embedding Mapped By Diagnosis Group",
    x = "UMAP1",
    y = "UMAP2"
  ) +
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    legend.position = "right",
    text = element_text(color = "black"),
    axis.text = element_text(color = "black")
  )
dev.off()

write.csv(MappingAV45[, c("UMAP1", "UMAP2", "DXGrp", "DX", "RID", "AGE", "VISCODE2", "SlingFusedPhatePseu", "Pseudotime_Normalized")], 
          file = "/Users/fahadmehfooz/Desktop/AE/SLOPE/Temp Files/Psuedotime_train_dl.csv", 
          row.names = FALSE)


write.csv(TestMappingAV45[, c("UMAP1", "UMAP2", "DXGrp", "DX", "RID", "AGE", "VISCODE2", "SlingFusedPhatePseu", "Pseudotime_Normalized")], 
          file = "/Users/fahadmehfooz/Desktop/AE/SLOPE/Temp Files/Psuedotime_test_dl.csv", 
          row.names = FALSE)

graphics.off()




