require('rpart') 
library('rpart.plot')   
train <- read.csv("./Documents/Academic/Cornell/CS5785/HW1/train.csv", header = TRUE, sep = ",")
test <- read.csv("./Documents/Academic/Cornell/CS5785/HW1/test.csv", header = TRUE, sep = ",")
drops <- c("PassengerId", "Survived", "Name", "Ticket", "Cabin")
full <- train[,!(names(train) %in% drops)]
full_m <- data.matrix(full)
drops_test <- c("PassengerId", "Name", "Ticket", "Cabin")
full_test <- test[,!(names(test) %in% drops)]

keeps <- c("Survived")
survived <- train[keeps]
survived_m <- data.matrix(survived)

fit <- rpart(train$Survived ~ ., data=full, method="class",control=rpart.control(xval=40))
pred_v <- predict(fit, newdata=full_test, type = "vector")
result <- integer()
for (i in pred_v) {
  if (i == 1) {
    result <- c(result, 0)
  } else {
    result <- c(result, 1)
  }
}
pid <- as.matrix(test[c("PassengerId")])
output <- cbind(pid, result)
#fit$cptable
#plotcp(fit)
rpart.plot(fit,type=4,cex=1.1, extra = 101,prefix="Survival: ",main="Titanic")
write.table(data.frame(output), file = "./Documents/Academic/Cornell/CS5785/HW1/rpart_xval40.csv",row.names=FALSE,col.names=FALSE, sep=",")