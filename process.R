install.packages('randomForest')
require('randomForest')

convertToFactors <- function(data){
  data$SibSp.factor <- factor(data$SibSp)
  data$Pclass.factor <- factor(data$Pclass)
  data$Parch.factor <- factor(data$Parch)
  convertToFactors <- data
}

testSolution <- function(model, testset, cutoff = 0.5){
  values <- predict(model, newdata=testset,type = "response") 
  values[is.na(values)] <- 0
  values.bin <- (values>=cutoff)==testset$Survived
  testSolution <- sum(values.bin)/nrow(testset)
}

train <- read.csv('train.csv')
test <- read.csv('test.csv')

test <- convertToFactors(test)
train <- convertToFactors(train)
train$Survived.factor <- as.factor(train$Survived)

train.idx <- sample(nrow(train),ceiling(nrow(train)*0.7))
test.idx <- (1:nrow(train))[-train.idx]

mod.logit <- glm(Survived.factor~Pclass.factor*Sex*Age+Fare+Embarked, data = train[train.idx,], family = binomial(link = "logit"))

testSolution(mod.logit,train[test.idx,])

test$Survived <- predict(mod.logit, newdata=test,type = "response")
test$Survived.binary <- 0
test$Survived.binary[test$Survived>0.5] <- 1
test$Survived <- test$Survived.binary

write.table(test[,c("PassengerId","Survived")], file="predictionslogit.csv",sep = ",",row.names=FALSE,col.names = TRUE)