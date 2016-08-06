library(ggplot2)
library(caret)
library(AppliedPredictiveModeling)

set.seed(123)

all_training_data <- read.csv("pml-training.csv", na.strings = c("NA","","#DIV/0!"))
all_testing_data <- read.csv("pml-testing.csv", na.strings = c("NA","","#DIV/0!"))

na_pct <- apply(apply(all_training_data,2,is.na),2,sum)/nrow(all_training_data)
training <- all_training_data[,na_pct < 0.8]

#blank_pct <- colSums(training == "")/nrow(training)
#training <- training[ , blank_pct < 0.5]

ind <- createDataPartition(training$classe, p=.5, list=FALSE)
training <- training[ind,]
validation <- training[-ind,]

training <- subset(training, select = -c(X,
                                user_name,
                                raw_timestamp_part_1,
                                raw_timestamp_part_2,
                                cvtd_timestamp,
                                new_window,
                                num_window)
                   )

start.time <- Sys.time()
mod <- train(factor(classe) ~ ., 
             data=training,
             method='gbm',
             trControl=trainControl(method='cv'))
end.time <- Sys.time()
print(end.time - start.time)

confusionMatrix(validation$classe, predict(mod, validation))

testing <- all_testing_data[, colnames(subset(training, select= -classe))]
preds <- predict(mod, testing)

preds

#lin_model <- lm(factor(training$classe) ~ ., data=trainingX)

