library(ggplot2)
library(dplyr)
library(ggpubr)
library(RColorBrewer)
library(slingshot)
library(fields)


# Theme for ggplot2 plots
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


# Data Preparation
train_lr <- read.csv("/Users/fahadmehfooz/Downloads/train_df_lr.csv")
test_lr <- read.csv("/Users/fahadmehfooz/Downloads/test_df_lr.csv")

train_mlp <- read.csv("/Users/fahadmehfooz/Downloads/train_df_mlp.csv")
test_mlp <- read.csv("/Users/fahadmehfooz/Downloads/test_df_mlp.csv")

train_en <- read.csv("/Users/fahadmehfooz/Downloads/train_df_en.csv")
test_en <- read.csv("/Users/fahadmehfooz/Downloads/test_df_en.csv")

train_lr$SlingFusedPhatePseu <- rowMeans(train_lr["SlingFusedPhatePseu"], na.rm = TRUE)
test_lr$SlingFusedPhatePseu <- rowMeans(test_lr["SlingFusedPhatePseu"], na.rm = TRUE)

DXGrp_train <- train_lr$DXGrp
DXGrp_test <- test_lr$DXGrp

test_lr$DXGrp

DXGrp_train[DXGrp_train == 1] <- 'CN'
DXGrp_train[DXGrp_train == 2] <- 'EMCI'
DXGrp_train[DXGrp_train == 3] <- 'LMCI'
DXGrp_train[DXGrp_train == 4] <- 'AD'
DXGrp_train <- factor(DXGrp_train, levels = c("CN", "EMCI", "LMCI", "AD"))

DXGrp_test[DXGrp_test == 1] <- 'CN'
DXGrp_test[DXGrp_test == 2] <- 'EMCI'
DXGrp_test[DXGrp_test == 3] <- 'LMCI'
DXGrp_test[DXGrp_test == 4] <- 'AD'
DXGrp_test <- factor(DXGrp_test, levels = c("CN", "EMCI", "LMCI", "AD"))

train_lr$DX <- DXGrp_train
test_lr$DX <- DXGrp_test

DXGrp_train

train_mlp$DX <- DXGrp_train
test_mlp$DX <- DXGrp_test

train_en$DX <- DXGrp_train
test_en$DX <- DXGrp_test

test_en

# Define color palette and comparison pairs for boxplots
my_colors <- c("CN" = "#F8766D", "EMCI" = "#7CAE00", "LMCI" = "#00BFC4", "AD" = "#C77CFF")
my_comparisons <- list(c("CN", "EMCI"), c("EMCI", "LMCI"), c("LMCI", "AD"))

# --------------------------- LR ------------------------------------
dev.new(width=10, height=7.5) 

png("/Users/fahadmehfooz/Desktop/AE/SLOPE/Temp Files/Plot_train_boxplot_lr.png")

par(mar = c(5, 5, 4, 5))

# Plot pseudotime across different diagnosis groups in the training dataset
ggboxplot(train_lr, x = "DX", y = "Pseudotime_Normalized",
          fill = "DX", ylab = "Logit", xlab = "Diagnosis Group") +
  stat_compare_means(comparisons = my_comparisons, aes(label = ..p.signif..),
                     method = "t.test", size = 5) +
  ggtitle("Logits Vs Diagnosis Group")  +
  theme(standard_theme,  plot.title = element_text(hjust = 0.5)) 

dev.off()

dev.new(width=10, height=7.5) 

png("/Users/fahadmehfooz/Desktop/AE/SLOPE/Temp Files/Plot_test_boxplot_lr.png")

par(mar = c(5, 5, 4, 5))

# Plot pseudotime across different diagnosis groups in the test dataset
ggboxplot(test_lr, x = "DX", y = "Pseudotime_Normalized", fill = "DX", ylab = "Logit", xlab = "Diagnosis Group") +
  stat_compare_means(comparisons = my_comparisons, aes(label = ..p.signif..), method = "t.test", size = 5) +
  ggtitle("Logit Vs Diagnosis Group") +
  theme(standard_theme,  plot.title = element_text(hjust = 0.5)) 

dev.off()


# --------------------------- MLP ------------------------------------
dev.new(width=10, height=7.5) 

png("/Users/fahadmehfooz/Desktop/AE/SLOPE/Temp Files/Plot_train_boxplot_mlp.png")
par(mar = c(5, 5, 4, 5))

# Plot pseudotime across different diagnosis groups in the training dataset
ggboxplot(train_mlp, x = "DX", y = "Pseudotime_Normalized", fill = "DX", ylab = "Logit", xlab = "Diagnosis Group") +
  stat_compare_means(comparisons = my_comparisons, aes(label = ..p.signif..), method = "t.test", size = 5) +
  ggtitle("Logit Vs DX Diagnosis Group") +
  theme(standard_theme,  plot.title = element_text(hjust = 0.5)) 

dev.off()

dev.new(width=10, height=7.5) 
png("/Users/fahadmehfooz/Desktop/AE/SLOPE/Temp Files/Plot_test_boxplot_mlp.png")
par(mar = c(5, 5, 4, 5))

# Plot pseudotime across different diagnosis groups in the test dataset
ggboxplot(test_mlp, x = "DX", y = "Pseudotime_Normalized", fill = "DX", ylab = "Logit", xlab = "Diagnosis Group") +
  stat_compare_means(comparisons = my_comparisons, aes(label = ..p.signif..), method = "t.test", size = 5) +
  ggtitle("Logit Vs Diagnosis Group") +
  theme(standard_theme,  plot.title = element_text(hjust = 0.5)) 

dev.off()

# --------------------------- EN ------------------------------------
dev.new(width=10, height=7.5) 

png("/Users/fahadmehfooz/Desktop/AE/SLOPE/Temp Files/Plot_train_boxplot_en.png")

par(mar = c(5, 5, 4, 5))

# Plot pseudotime across different diagnosis groups in the training dataset
ggboxplot(train_en, x = "DX", y = "Pseudotime_Normalized", fill = "DX", ylab = "Logit", xlab = "Diagnosis Group") +
  stat_compare_means(comparisons = my_comparisons, aes(label = ..p.signif..), method = "t.test", size = 5) +
  ggtitle("Logit Vs Diagnosis Group")  +
  theme(standard_theme,  plot.title = element_text(hjust = 0.5)) 

dev.off()

dev.new(width=10, height=7.5) 

png("/Users/fahadmehfooz/Desktop/AE/SLOPE/Temp Files/Plot_test_boxplot_en.png" )

par(mar = c(5, 5, 4, 5))


# Plot pseudotime across different diagnosis groups in the test dataset
ggboxplot(test_en, x = "DX", y = "Pseudotime_Normalized", fill = "DX", ylab = "Logit", xlab = "Diagnosis Group") +
  stat_compare_means(comparisons = my_comparisons, aes(label = ..p.signif..), method = "t.test", size = 5) +
  ggtitle("Logit Vs Diagnosis Group")  +
  theme(standard_theme,  plot.title = element_text(hjust = 0.5)) 

dev.off()

# --------------------------- --------------------------- ------------------------------------

graphics.off()

