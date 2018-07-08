library(e1071)
library(randomForest)
library(rpart)
library(gbm)
library(ggplot2)
library(psych)
library(reshape2)
library(xlsx)
library(fastDummies)

df_original <- read.xlsx('C:/Users/Dell/Downloads/Absenteeism_at_work.xls', 1)

num <- list('Distance.from.Residence.to.Work', 'Service.time', 'Age',
            'Work.load.Average.day.', 'Transportation.expense',
            'Hit.target', 'Son', 'Pet', 'Weight', 'Height', 
            'Body.mass.index', 'Absenteeism.time.in.hours')

cat <- setdiff(names(df_original),num)

# --------------------------------Generic Functions-----------------------------

outlier_removal <- function(un_data, cols){
  data <- un_data
  for (i in cols){
    print(i)
    percentiles <- quantile(data[,i], c(.75, .25))
    q75 <- percentiles[[1]]
    q25 <- percentiles[[2]]
    iqr <- q75 - q25
    min <- q25 - (1.5*iqr)
    max <- q75 + (1.5*iqr)
    print(paste(q75,q25,iqr,min,max))
    data <- data[!(data[,i] < min | data[,i] > max), ]
    print(nrow(data))
  }
  return(data)
}

standardize_data <- function(data, source){
  
  # data : data frame to be transformed
  # source : data frame whose mean and standard deviation will be 
  # used to transform the 'data' mentioned above
  
  if (ncol(data) != ncol(source)){
    print('Please make sure data and source have same number of columns')
    return
  }
  col_means <- sapply(source, mean)
  print(col_means)
  col_sd <- sapply(source, sd)
  print(col_sd)
  for (i in 1:ncol(data)){
    data[,i] <- (data[,i] - col_means[i])/col_sd[i] 
  }
  return(data)
}

Mode <- function(x){
  return(as.numeric(names(which.max(table(x)))))
}

impute_missing_values <- function(data, method){
  if (method == 'median'){
    data[is.na(data)] <- round(median(data, na.rm = T))
    return(data)
  }
  if (method == 'mode'){
    data[is.na(data)] <- Mode(data)
    return(data)
  }
  else{
    print("Please pass 'median' or 'mode' as argument in method parameter")
    return
  }
}

RMSE <- function(original_values, predicted_values){
  return(sqrt(mean((original_values - predicted_values)^2)))
}

# --------------------------Exploratory Data Analysis--------------------------

summary_df_original <- summary(df_original)

# To check number of missing values in each column
print(summary_df_original)

# Fill missing values
df_original[,as.character(num)] <- as.data.frame(
  sapply(df_original[,as.character(num)], impute_missing_values, 
         method='median'))

df_original[,cat] <- as.data.frame(
  sapply(df_original[,cat], impute_missing_values, 
         method='mode'))

# Split the data into train and test set
set.seed(1)

indexes <- sample(1:nrow(df_original), size=0.2*nrow(df_original))

train_data <- df_original[-indexes,]

test_data <- df_original[indexes,]

train_num <- train_data[,as.character(num)]

train_num_melt <- melt(train_num)

train_cat <- train_data[,cat]

multi.hist(train_num, main = '', dcol=c('white','black'), 
           dlty=c("solid", "solid"), bcol = 'white', density=TRUE)

cor_mat <- cor(train_num)

melted_cormat <- melt(cor_mat)

ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile() + theme(
    axis.text.x = element_text(
      angle = 45, vjust = 1, size = 10, hjust = 1), 
    axis.title.x = element_blank(), axis.title.y = element_blank())

ggplot(train_num_melt, aes(x = 0, y = value)) +
  geom_boxplot() + facet_wrap(~variable, scales='free') + 
  theme(axis.title.x=element_blank())

# -----------------------Predictive Modeling Experiments-----------------------
  
df_original[,cat] <- as.data.frame(sapply(df_original[,cat], 
                                          as.factor))

df_original <- dummy_cols(df_original)

df_original <- df_original[,!(names(df_original) %in% cat)]

# Split the data into train and test set
set.seed(1)

indexes <- sample(1:nrow(df_original), size=0.2*nrow(df_original))

train_data <- df_original[-indexes,]

test_data <- df_original[indexes,]

x_train <- train_data[,!(names(train_data) == 'Absenteeism.time.in.hours')]

y_train <- train_data$Absenteeism.time.in.hours

x_test <- test_data[,!(names(test_data) == 'Absenteeism.time.in.hours')]

y_test <- test_data$Absenteeism.time.in.hours

# ****************************Linear Regression******************************

lr <- lm(Absenteeism.time.in.hours ~ ., data = train_data)

lr_predictions_train <- predict(lr, x_train)

lr_training_error <- RMSE(y_train, lr_predictions_train)

lr_predictions_test <- predict(lr, x_test)

lr_test_error <- RMSE(y_test, lr_predictions_test)

sprintf('Training set error of Linear Regression is : %f', lr_training_error)

sprintf('Test set error of Linear Regression is : %f', lr_test_error)

# **************************Support Vector Machine*****************************

set.seed(1)

svr <- svm(Absenteeism.time.in.hours ~ ., data = train_data, kernel='radial')

svr_predictions_train <- predict(svr, x_train)

svr_training_error <- RMSE(y_train, svr_predictions_train)

svr_predictions_test <- predict(svr, x_test)

svr_test_error <- RMSE(y_test, svr_predictions_test)

sprintf('Training set error of Support Vector Regression is : %f', svr_training_error)

sprintf('Test set error of Support Vector Regression is : %f', svr_test_error)

# ***************************Gradient Boosted Tree*****************************

# Gradient Boosted Tree in R requires target to be integer rather 
# than factor that's why instead of using clean_train and clean_test
# itself, data has been prepared below for Gradient Boosted Tree only

# gbt_train <- clean_train
# 
# gbt_train$Churn <- as.integer(gbt_train$Churn)
# 
# gbt_train$Churn[gbt_train$Churn == 1] <- 0
# gbt_train$Churn[gbt_train$Churn == 2] <- 1
# 
# gbt_x_train <- gbt_train[,!(names(gbt_train) == 'Churn')]
# 
# gbt_y_train <- gbt_train$Churn
# 
# gbt_test <- clean_test
# 
# gbt_test$Churn <- as.integer(gbt_test$Churn)
# 
# gbt_test$Churn[gbt_test$Churn == 1] <- 0
# gbt_test$Churn[gbt_test$Churn == 2] <- 1
# 
# gbt_x_test <- gbt_test[,!(names(gbt_test) == 'Churn')]
# 
# gbt_y_test <- gbt_test$Churn

set.seed(1)

gbr <- gbm(Absenteeism.time.in.hours ~ ., data=train_data, n.trees=100, shrinkage=0.1, interaction.depth=2)

gbr_predictions_train <- predict(gbr, x_train, n.trees=100)

gbr_training_error <- RMSE(y_train, gbr_predictions_train)

gbr_predictions_test <- predict(gbr, x_test, n.trees=100)

gbr_test_error <- RMSE(y_test, gbr_predictions_test)

sprintf('Training set error of Gradient Boosted Regression is : %f', gbr_training_error)

sprintf('Test set error of Gradient Boosted Regression is : %f', gbr_test_error)

# ********************************Decision Tree********************************

set.seed(1)

dt <- rpart(Absenteeism.time.in.hours ~ ., data=train_data, method='anova')

dt_predictions_train <- predict(dt, x_train)

dt_training_error <- RMSE(y_train, dt_predictions_train)

dt_predictions_test <- predict(dt, x_test)

dt_test_error <- RMSE(y_test, dt_predictions_test)

sprintf('Training set error of Decision Tree Regression is : %f', dt_training_error)

sprintf('Test set error of Decision Tree Regression is : %f', dt_test_error)

# ********************************Random Forest********************************

set.seed(1)

rfr <- randomForest(Absenteeism.time.in.hours ~ ., data=train_data)

rfr_predictions_train <- predict(rfr, x_train)

rfr_training_error <- RMSE(y_train, rfr_predictions_train)

rfr_predictions_test <- predict(rfr, x_test)

rfr_test_error <- RMSE(y_test, rfr_predictions_test)

sprintf('Training set error of Random Forest is : %f', rfr_training_error)

sprintf('Test set error of Random Forest is : %f', rfr_test_error)

# *************************************End**************************************
