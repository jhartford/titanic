install.packages('randomForest')
require('randomForest')

convertToFactors <- function(data){
  data$SibSp.factor <- factor(data$SibSp)
  data$Pclass.factor <- factor(data$Pclass)
  data$Parch.factor <- factor(data$Parch)
  convertToFactors <- data
}

train <- read.csv('train.csv')
test <- read.csv('test.csv')

test <- convertToFactors(test)
train <- convertToFactors(train)
train$Survived.factor <- as.factor(train$Survived)

mod.logit <- glm(Survived.factor~Pclass.factor*Sex*Age+Fare+Embarked, data = train, family = binomial(link = "logit"))

test$Survived <- predict(mod.logit, newdata=test,type = "response")
test$Survived.binary <- 0
test$Survived.binary[test$Survived>0.5] <- 1
test$Survived <- test$Survived.binary

write.table(test[,c("PassengerId","Survived")], file="predictionslogit.csv",sep = ",",row.names=FALSE,col.names = TRUE)