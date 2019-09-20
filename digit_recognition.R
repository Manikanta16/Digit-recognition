#######################
#Author: Guru Manikanta Innamuri
#Date: 23-06-2018
#Problem: Digit Recognition
#######################

#######################################################
#------------DIGIT RECOGNITION ASSIGNMENT-------------#
#######################################################


#This problem is aimed at identifying the digit from the pixel data available
#load the data
pixels <- read.csv('mnist_train.csv')
#check the number of rows and columns in the dataset
dim(pixels)
#summary of our data
summary(pixels)

#Rename the label column
colnames(pixels)[1] <- "label"

#Check for missing values
sum(sapply(pixels,function(x) sum(is.na(x)))) #no missing values in the data
pixels$label <- factor(pixels$label)
#######################################
####----MODEL BUILDING----####
#######################################
set.seed(3)

#import the packages
library(caret)
library(kernlab)
library(readr)

train_indices <- sample(1:nrow(pixels),size=5000)
train = pixels[train_indices,]
test = pixels[-train_indices,]

#linear SVM
linear_svm <- ksvm(label~. , data=train, scaled=FALSE, kernel="vanilladot")
linear_pred <- predict(linear_svm,test)

#confusion Matrix
confusionMatrix(linear_pred,test$label) 


#non-linear SVM
poly_svm <- ksvm(label~. , data=train, scaled=FALSE, kernel="polydot")
poly_pred <- predict(poly_svm,test)

#confusion Matrix
confusionMatrix(poly_pred,test$label) 


#RBF kernel
rbf_svm <- ksvm(label~., data=train, scaled=FALSE, kernel = "rbfdot")
rbf_pred <- predict(rbf_svm,test)

#confusion Matrix
confusionMatrix(rbf_pred,test$label)


############   Hyperparameter tuning and Cross Validation #####################



trainControl <- trainControl(method="cv", number=5)


# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"


set.seed(1)
grid <- expand.grid(.sigma=seq(0.01,0.05,0.01), .C=seq(1,5,by=1) )


#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(label~., data=train, method="svmRadial", scale = FALSE,metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

print(fit.svm)

plot(fit.svm)


test_data <- read.csv("mnist_test.csv")

digit <- predict(fit.svm,test_data)









