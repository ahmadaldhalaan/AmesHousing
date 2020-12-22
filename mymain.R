library(glmnet)
library(gbm)
library(caret)

# pre-process training data
train <- read.csv("train.csv", stringsAsFactors = T)

train$Garage_Yr_Blt[is.na(train$Garage_Yr_Blt)] = 0
train[which(train$Year_Remod_Add == 1950),]$Year_Remod_Add = 0
train$Year_Remod_Add <- ifelse(train$Year_Remod_Add == train$Year_Built, 0, 1)
train$Total_Area <- train$Mas_Vnr_Area + train$Bsmt_Unf_SF + train$Total_Bsmt_SF + train$Gr_Liv_Area + 
  train$Wood_Deck_SF + train$Open_Porch_SF + train$Enclosed_Porch + train$Screen_Porch

remove.var <- c('Street', 'Utilities',  'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 
                'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude','Latitude')

train <- train[ , !(names(train) %in% remove.var)]

winsor.vars <- c("Total_Area", "Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", 
                 "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", 
                 "Wood_Deck_SF", "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", 
                 "Screen_Porch", "Misc_Val")

quan.value <- 0.95
for(var in winsor.vars){
  tmp <- train[, var]
  myquan <- quantile(tmp, probs = quan.value, na.rm = TRUE)
  tmp[tmp > myquan] <- myquan
  train[, var] <- tmp
}

train.x <- train[,-c(1,72)]
train.y <- log(train$Sale_Price)

dm <- dummyVars(" ~ .", data = train.x)
train.matrix <- data.frame(predict(dm, newdata = train.x))

# fit training data using lasso
cv.out <- cv.glmnet(as.matrix(train.matrix), as.matrix(train.y), alpha = 1)

# fit training data using gradient boosting machines
gbm.fit <- gbm(
  formula = log(Sale_Price) ~ .,
  distribution = "gaussian",
  data = train,
  n.trees = 1000,
  interaction.depth = 3,
  shrinkage = 0.1,
  bag.fraction = .5, 
  cv.folds = 5
)

# pre-process test data
test <- read.csv("test.csv", stringsAsFactors = T)

test$Garage_Yr_Blt[is.na(test$Garage_Yr_Blt)] = 0
test[which(test$Year_Remod_Add == 1950),]$Year_Remod_Add = 0
test$Year_Remod_Add <- ifelse(test$Year_Remod_Add == test$Year_Built, 0, 1)
test$Total_Area <- test$Mas_Vnr_Area + test$Bsmt_Unf_SF + 
  test$Total_Bsmt_SF + test$Gr_Liv_Area + test$Wood_Deck_SF + 
  test$Open_Porch_SF + test$Enclosed_Porch + test$Screen_Porch

test <- test[ , !(names(test) %in% remove.var)]

for(var in winsor.vars){
  tmp <- test[, var]
  myquan <- quantile(tmp, probs = quan.value, na.rm = TRUE)
  tmp[tmp > myquan] <- myquan
  test[, var] <- tmp
}

test.x <- test[,-1]

dm <- dummyVars(" ~ .", data = test.x)
test.matrix <- data.frame(predict(dm, newdata = test.x))

# add training data levels (not in test data) to test data and set as 0
zero.cols <- setdiff(colnames(train.matrix), colnames(test.matrix))
zeros <- matrix(0L, nrow = dim(test.matrix)[1], ncol = length(zero.cols))
colnames(zeros) <- zero.cols
test.matrix <- cbind(test.matrix, zeros)

# remove levels in test data not in train data
same.cols <- intersect(colnames(train.matrix), colnames(test.matrix))
test.matrix <- test.matrix[,same.cols]

# predict using lasso model
y.pred.linear <-predict(cv.out, s = cv.out$lambda.min, newx = as.matrix(test.matrix))

# predict using gbm model
y.pred.tree = predict(gbm.fit, test, n.trees = gbm.fit$n.trees)

# save predictions in file
mysubmission1 <- data.frame(PID = test[,"PID"], Sale_Price = exp(y.pred.linear))
colnames(mysubmission1) <- c("PID", "Sale_Price")
write.table(mysubmission1, file = "mysubmission1.txt", sep = ",", row.names = FALSE)

# save predictions in file
mysubmission2 <- data.frame(PID = test[,"PID"], Sale_Price = exp(y.pred.tree))
colnames(mysubmission2) <- c("PID", "Sale_Price")
write.table(mysubmission2, file = "mysubmission2.txt", sep = ",", row.names = FALSE)