##load data
library(readr)
data <- read.csv("~/R/Coursera/MachineLearning_Project/pml-training.csv")
test_set_quiz <- read.csv("~/R/Coursera/MachineLearning_Project/pml-testing.csv")

## pre-process data
library(dplyr)
data <- data %>%
     mutate(classe = as.factor(classe))

set.seed(1234)
library(mlbench)
## slice training set into train (60%), test (20%), and validation (20%) sets
library(caret)
train_index <- createDataPartition(y = data$classe, p = 0.8, list = FALSE)
test_index <- createDataPartition(y = train_index, p = 0.25, list = FALSE)
training <- data[train_index[-test_index], ]
testing <- data[train_index[test_index], ]
validation <- data[-c(train_index, train_index[test_index]), ]

## remove columns with at least half of cells NA or blank
training <- training %>%
     select_if(~ sum(!is.na(.) & . != "") >= 19622/2) %>%
     select(-(2:5))
training <- training %>%
     select(-(1:3)) %>%
     select(-(31:33)) %>%
     select(-(8:10)) %>%
     select(-(4))

##looking at correlations
m <- abs(cor(training[ ,-46]))
diag(m) <- 0
which(m>.8, arr.ind = TRUE)
heatmap(m, 
        col = colorRampPalette(c("blue", "white", "red"))(20),
        main = "Correlation Heatmap",
        xlab = "Variables", ylab = "Variables")


# Identify variables with positive correlation
high_corr_vars <- which(m > 0.8 & m < 1, arr.ind = TRUE)

# Display the variables and their correlation values
for (i in 1:nrow(high_corr_vars)) {
     row_index <- high_corr_vars[i, 1]
     col_index <- high_corr_vars[i, 2]
     correlation_value <- m[row_index, col_index]
     
     cat("Variables:", rownames(m)[row_index], "and", colnames(m)[col_index], 
         "have a high correlation of", correlation_value, "\n")
}

##PCA
pca_result <- prcomp(training[,-46], scale. = TRUE)
summary(pca_result)
print(pca_result$rotation)

##Tree
tree <- train(classe~., method = "rpart", data = training)
fancyRpartPlot(tree$finalModel)
tree$finalModel

## set up parallel processing
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
fitControl <- trainControl(method = "cv",
                           number = 10,
                           allowParallel = TRUE)

##Random Forest
rf_parallel <- train(classe~., method= "rf", data = training, trControl = fitControl)
stopCluster(cluster)
registerDoSEQ()

##validating results (within training set)
confusionMatrix(rf_parallel)
plot(varImp(rf_parallel))

##looking at correlations between top variables
top_vars <- rownames(varImp(rf_parallel)$importance)[1:20]
top_vars_data <- training[, top_vars]
top_corr <- cor(top_vars_data)
highcorr_top<- which(top_corr > 0.75 & top_corr < 1, arr.ind = TRUE)
for (i in 1:nrow(highcorr_top)) {
     row_index <- highcorr_top[i, 1]
     col_index <- highcorr_top[i, 2]
     correlation_value <- top_corr[row_index, col_index]
     
     cat("Variables:", rownames(top_corr)[row_index], "and", colnames(top_corr)[col_index], 
         "have a high correlation of", correlation_value, "\n")
}

##pca of top variables
pca_top <- prcomp(top_vars_data, scale. = TRUE)


## predicting on test set
rf1_predict <- predict(rf_parallel, newdata=testing)
confusionMatrix(rf1_predict, testing$classe)

