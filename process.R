install.packages('randomForest')
install.packages('neuralnet')
install.packages('nnet')
require('neuralnet')
require('nnet')
require('randomForest')
require('stringr')
set.seed(2000)

missingfactor <- function(fact){
  temp <- as.character()
}

convertToFactors <- function(data){
  data$Title <- str_extract(data$Name,'(M[a-z]{1,})\\.')
  data$SibSp.factor <- factor(data$SibSp)
  data$Pclass.factor <- factor(data$Pclass)
  data$Parch.factor <- factor(data$Parch)
  data$Age.factors <- as.character(cut(data$Age,breaks=c(0.34,16.3,32.2,48.2,64.1,80.1)))
  data$Age.factors[is.na(data$Age.factors)] <- "Missing"
  data$Age.factors <- as.factor(data$Age.factors)
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

train.imputed <- rfImpute(Survived.factor ~ Pclass.factor + Sex + Age.factors+Fare+Embarked + Parch,train, ntree = 2000)
train.imputednn <- rfImpute(Survived ~ Pclass+ Sex + Age +Fare+Embarked + Parch,train, ntree = 2000)

#mod.logit1 <- glm(Survived.factor~Pclass.factor*Sex*Age.factors+Fare+Embarked + Parch*Age.factors, data = train[train.idx,], family = binomial(link = "logit"))
mod.logit2 <- glm(Survived.factor~Pclass.factor*Sex*Age.factors+Fare+Embarked + Parch + Title, data = train, family = binomial(link = "logit"))
mod.probit <- glm(Survived.factor~Pclass.factor*Sex*Age.factors+Fare+Embarked + Parch, data = train[train.idx,], family = binomial(link = "probit"))

mod.logitimpute <- glm(Survived.factor~Pclass.factor*Sex*Age.factors+Fare+Embarked, data = train.imputed[train.idx,], family = binomial(link = "logit"))
mod.randforest <- randomForest(Survived.factor ~ Pclass.factor + Sex + Age.factors, data = train.imputed[train.idx,], ntree = 2000)

mod.nnet <- neuralnet(Survived ~ Age + Pclass +Fare  + Parch, data = train.imputednn,hidden=5)
mod.nnet <- nnet(Survived ~ Age + Pclass +Fare  + Parch, data = train.imputednn,subset = train.idx,size = 15,decay=0.1,maxit=1000)

print(testSolution(mod.nnet,train[test.idx,],cutoff=0.5))

p <- as.numeric(predict(mod.nnet,train[test.idx,])>0.5) == train[test.idx,]$Survived

print(testSolution(mod.logit2,train[test.idx,],cutoff=0.5))
print(testSolution(mod.probit,train[test.idx,],cutoff=0.5))
print(testSolution(mod.logitimpute,train[test.idx,],cutoff=0.5))
print(testSolution(mod.randforest,train[test.idx,],cutoff=0.5))

test$Survived <- predict(mod.logit2, newdata=test,type = "response")
test$Survived.binary <- 0
test$Survived.binary[test$Survived>0.5] <- 1
test$Survived <- test$Survived.binary

write.table(test[,c("PassengerId","Survived")], file="predictionslogit.csv",sep = ",",row.names=FALSE,col.names = TRUE)

#create a Surv object
s <- with(train[train.idx,],Surv(Age,Survived))

#plot kaplan-meier estimate, per sex
fKM <- survfit(s ~ Sex ,data=train[train.idx,])
plot(fKM)

#plot Cox PH survival curves, per sex
sCox <- coxph(s ~ Sex,data=train[train.idx,])
lines(survfit(sCox,train[test,]),col='blue')
lines(survfit(sCox,newdata=data.frame(Sex='female',Pclass=1)),col='green')

#plot weibull survival curves, per sex,
sWei <- survreg(s ~ Sex + Pclass,dist='weibull',data=train[train.idx,])
sWei1 <- survreg(s ~ Pclass.factor*Sex*Age+Fare+Embarked,dist='weibull',data=train[train.idx,])

lines(predict(sWei, newdata=list(Sex='male', Pclass=3),type="quantile",p=seq(.01,.99,by=.01)),seq(.99,.01,by=-.01),col="red")
lines(predict(sWei, newdata=list(Sex='female'),type="quantile",p=seq(.01,.99,by=.01)),seq(.99,.01,by=-.01),col="red")