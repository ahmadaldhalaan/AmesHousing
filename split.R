ames <- read.csv("Ames_data.csv")

testIDs <- read.table("project1_testIDs.dat")
j <- 10
train <- ames[-testIDs[,j], ]
test <- ames[testIDs[,j], ]
test.y <- test[, c(1, 83)]
test <- test[, -83]
write.csv(train,"train.csv",row.names=FALSE)
write.csv(test, "test.csv",row.names=FALSE)
write.csv(test.y,"test_y.csv",row.names=FALSE)