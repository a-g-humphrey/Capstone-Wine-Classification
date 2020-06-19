#Please be aware that towards the end of the code there is a random forest implementation. On my machine this
#took about 10 minutes to run despite the relatively small data set


library(tidyverse)
library(readtext)
library(data.table)
library(caret)
library(corrplot)
library(purrr)
library(randomForest)
library(caretEnsemble)
library(knitr)
library(kableExtra)
library(rpart.plot)
library(Rborist)
#Download csv (semicolon deliminated) file for red
dl <-tempfile()
download.file(
  'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',dl)
winequality_red <- read_delim(dl, 
                                ";", escape_double = FALSE, trim_ws = TRUE)
#Download csv (semicolon deliminated) file for white
dl <-tempfile()
download.file(
  'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',dl)
winequality_white <- read_delim(dl, 
                              ";", escape_double = FALSE, trim_ws = TRUE)
#add the wine colour for classification purposes
winequality_red <- winequality_red %>% mutate(type = 'red')
winequality_white <- winequality_white %>% mutate(type = 'white')
#merge the data tables
data <- rbind(winequality_red,winequality_white)
data$type <- factor(data$type)
data$quality <- factor(as.character(data$quality))
#cleaning up column names to make data processing slightly easier
data <- data %>% mutate(fixed_acidity = `fixed acidity`, 
               volatile_acidity = `volatile acidity`,
               citric_acid = `citric acid`,
               residual_sugar = `residual sugar`,
               free_sulfur_dioxide = `free sulfur dioxide`,
               total_sulfur_dioxide = `total sulfur dioxide`
               )%>%
  select(-`fixed acidity`,-`volatile acidity`,-`citric acid`,
         -`free sulfur dioxide`,-`total sulfur dioxide`, -`residual sugar` )
#Iniial exploration of the data, is there some easy answers as two type (colour) or quality?
data %>%
  group_by(type)%>%
  summarise(n = n())%>%
  ggplot(aes(x = type, y = n))+
  geom_bar(stat="identity")+
  geom_text(aes(label=n), position=position_dodge(width=0.9), vjust=-0.25)+
  ylab("Count of observations")+
  xlab("Type of wine (colour)")+
  ggtitle(label =  "Count of wines by type in the dataset")+
  theme_bw()
data %>%
  group_by(quality)%>%
  summarise(n = n())%>%
  ggplot(aes(x = quality, y = n))+
  geom_bar(stat="identity")+
  geom_text(aes(label=n), position=position_dodge(width=0.9), vjust=-0.25)+
  ylab("Count of observations")+
  xlab("Quality")+
  ggtitle(label =  "Count of Wines by Quality in the dataset")+
  theme_bw()
data %>%
  gather(-type,-quality,-fixed_acidity, key = "var", value = "value") %>%
  ggplot(aes(x = value, y = fixed_acidity, colour = type)) +
  geom_point() +
  facet_wrap(~ var, scales = "free") +
  theme_bw()+
  ggtitle(label = "Wine Variables Comparison Plot - Type")
data %>%
  gather(-type,-quality,-fixed_acidity, key = "var", value = "value") %>%
  ggplot(aes(x = value, y = fixed_acidity, colour = quality)) +
  geom_point() +
  facet_wrap(~ var, scales = "free") +
  theme_bw()+
  ggtitle(label = "Wine Variables Comparison Plot - Quality")
data %>%
  gather(-type,-quality,key = "var", value = "value")%>%
  ggplot(aes(var,value,fill = type, color = type))+
  geom_boxplot()+
  facet_wrap(~var,scales = "free")+
  theme_bw()+ggtitle(label = "Wine Variables Comparison Plot - Boxplot by Type")
data %>%
  gather(-type,-quality,key = "var", value = "value")%>%
  ggplot(aes(var,value,fill = quality, color = quality))+
  geom_boxplot()+
  facet_wrap(~var,scales = "free")+
  theme_bw()+ggtitle(label = "Wine Variables Comparison Plot - Boxplot by Quality")
#data pre-processing
data <- data %>% filter(!quality %in% c(3,9))
data$quality <- factor(as.character(data$quality))
data <- data %>% select(type,everything())%>% mutate(index = row_number())
#creating a validation, test, train data set
set.seed(1)
vali_index <- createDataPartition(data$index,p = 0.2, times = 1, list = F)
vali <- data[vali_index,]
data_adjusted <- data[-vali_index,]
set.seed(1)
test_index <- createDataPartition(data_adjusted$index,p = 0.2, times = 1, list = F)
test <- data_adjusted[test_index,]
train <- data_adjusted[-test_index,]
vali <- vali %>% select(-index)
train <- train %>% select(-index)
test <- test %>% select(-index)
# Linear regression model Y = 1 for white and Y = 0 for red
lm_fit <- mutate(train, y = as.numeric(type == "white")) %>%
  lm(y ~ fixed_acidity, data = .)
p_hat <- predict(lm_fit,test)
y_hat <- ifelse(p_hat> 0.5, "white","red") %>% factor()
c <- confusionMatrix(y_hat,test$type)
Linear_Models <- data.frame(Variable = "fixed acidity", Accuracy = c$overall[1])
#Creating a vis to explain the initial model
train %>% 
  mutate(x = round(fixed_acidity)) %>%
  group_by(x) %>%
  filter(n() >= 10) %>%
  summarize(prop = mean(type == "white")) %>%
  ggplot(aes(x, prop)) +
  geom_point()+ 
  geom_abline(intercept = lm_fit$coef[1], slope = lm_fit$coef[2])
#Testing all of the variables for their accuracy to find the best linear regression model
a <- data %>% select(-type,-quality,-fixed_acidity) %>% colnames() %>% factor()
lm_fit <- mutate(train, y = as.numeric(type == "white")) %>%
  lm(y ~ chlorides, data = .)
p_hat <- predict(lm_fit,test)
y_hat <- ifelse(p_hat> 0.5, "white","red") %>% factor()
c <- confusionMatrix(y_hat,test$type)
Linear_Models <- bind_rows(Linear_Models,
                                data_frame( Variable = "chlorides", Accuracy = c$overall[1]))
lm_fit <- mutate(train, y = as.numeric(type == "white")) %>%
  lm(y ~ density, data = .)
p_hat <- predict(lm_fit,test)
y_hat <- ifelse(p_hat> 0.5, "white","red") %>% factor()
c <- confusionMatrix(y_hat,test$type)
Linear_Models <- bind_rows(Linear_Models,
                           data_frame( Variable = "density", Accuracy = c$overall[1]))
lm_fit <- mutate(train, y = as.numeric(type == "white")) %>%
  lm(y ~ pH, data = .)
p_hat <- predict(lm_fit,test)
y_hat <- ifelse(p_hat> 0.5, "white","red") %>% factor()
c <- confusionMatrix(y_hat,test$type)
Linear_Models <- bind_rows(Linear_Models,
                           data_frame( Variable = "pH", Accuracy = c$overall[1]))
lm_fit <- mutate(train, y = as.numeric(type == "white")) %>%
  lm(y ~ sulphates, data = .)
p_hat <- predict(lm_fit,test)
y_hat <- ifelse(p_hat> 0.5, "white","red") %>% factor()
c <- confusionMatrix(y_hat,test$type)
Linear_Models <- bind_rows(Linear_Models,
                           data_frame( Variable = "sulphates", Accuracy = c$overall[1]))
lm_fit <- mutate(train, y = as.numeric(type == "white")) %>%
  lm(y ~ alcohol, data = .)
p_hat <- predict(lm_fit,test)
y_hat <- ifelse(p_hat> 0.5, "white","red") %>% factor()
c <- confusionMatrix(y_hat,test$type)
Linear_Models <- bind_rows(Linear_Models,
                           data_frame( Variable = "alcohol", Accuracy = c$overall[1]))
lm_fit <- mutate(train, y = as.numeric(type == "white")) %>%
  lm(y ~ volatile_acidity, data = .)
p_hat <- predict(lm_fit,test)
y_hat <- ifelse(p_hat> 0.5, "white","red") %>% factor()
c <- confusionMatrix(y_hat,test$type)
Linear_Models <- bind_rows(Linear_Models,
                           data_frame( Variable = "volatile_acidity", Accuracy = c$overall[1]))
lm_fit <- mutate(train, y = as.numeric(type == "white")) %>%
  lm(y ~ citric_acid, data = .)
p_hat <- predict(lm_fit,test)
y_hat <- ifelse(p_hat> 0.5, "white","red") %>% factor()
c <- confusionMatrix(y_hat,test$type)
Linear_Models <- bind_rows(Linear_Models,
                           data_frame( Variable = "citric_acid", Accuracy = c$overall[1]))
lm_fit <- mutate(train, y = as.numeric(type == "white")) %>%
  lm(y ~ residual_sugar, data = .)
p_hat <- predict(lm_fit,test)
y_hat <- ifelse(p_hat> 0.5, "white","red") %>% factor()
c <- confusionMatrix(y_hat,test$type)
Linear_Models <- bind_rows(Linear_Models,
                           data_frame( Variable = "residual_sugar", Accuracy = c$overall[1]))
lm_fit <- mutate(train, y = as.numeric(type == "white")) %>%
  lm(y ~ free_sulfur_dioxide, data = .)
p_hat <- predict(lm_fit,test)
y_hat <- ifelse(p_hat> 0.5, "white","red") %>% factor()
c <- confusionMatrix(y_hat,test$type)
Linear_Models <- bind_rows(Linear_Models,
                           data_frame( Variable = "free_sulfur_dioxide", Accuracy = c$overall[1]))
lm_fit <- mutate(train, y = as.numeric(type == "white")) %>%
  lm(y ~ total_sulfur_dioxide, data = .)
p_hat <- predict(lm_fit,test)
y_hat <- ifelse(p_hat> 0.5, "white","red") %>% factor()
c <- confusionMatrix(y_hat,test$type)
Linear_Models <- bind_rows(Linear_Models,
                           data_frame( Variable = "total_sulfur_dioxide", Accuracy = c$overall[1]))
Linear_Models

Type_Models <-  data.frame(Model = "Linear Regression", Accuracy = c$overall[1])
#based on all of the models, total_sulfur dioxide was the best performing model, so ploting we see:
train %>% 
  mutate(x = round(total_sulfur_dioxide)) %>%
  group_by(x) %>%
  filter(n() >= 10) %>%
  summarize(prop = mean(type == "white")) %>%
  ggplot(aes(x, prop)) +
  geom_point()+ 
  geom_abline(intercept = lm_fit$coef[1], slope = lm_fit$coef[2])
#Linear model result in a straight line. What happens with non-linear regression?
glm_fit <- train %>% 
  mutate(y = as.numeric(type == "white")) %>%
  glm(y ~ total_sulfur_dioxide, data=., family = "binomial")
p_hat_logit <- predict(glm_fit, newdata = test, type = "response")
tmp <- train %>% 
  mutate(x = round(total_sulfur_dioxide)) %>%
  group_by(x) %>%
  filter(n() >= 10) %>%
  summarize(prop = mean(type == "white")) 
logistic_curve <- data.frame(x = seq(min(tmp$x), max(tmp$x))) %>%
  mutate(p_hat = plogis(glm_fit$coef[1] + glm_fit$coef[2]*x))
tmp %>% 
  ggplot(aes(x, prop)) +
  geom_point() +
  geom_line(data = logistic_curve,
            mapping = aes(x, p_hat), lty = 2)
y_hat_logit <- ifelse(p_hat_logit > 0.5, "white", "red") %>% factor
c<-confusionMatrix(y_hat_logit, test$type)

data.frame(x = seq(min(tmp$x), max(tmp$x))) %>%
  mutate(logistic = plogis(glm_fit$coef[1] + glm_fit$coef[2]*x),
         regression = lm_fit$coef[1] + lm_fit$coef[2]*x) %>%
  gather(method, p_x, -x) %>%
  ggplot(aes(x, p_x, color = method)) + 
  geom_line() +
  geom_hline(yintercept = 0.5, lty = 5)

Type_Models <-  bind_rows(Type_Models,
                          data.frame(Model = "Logistic Regression - 1 Predictor",
                                     Accuracy = c$overall[1]))
#multivariate glm - using all variables rather than one to find the type
glm_fit <- train %>% select(-quality)%>%
  mutate(y = as.numeric(type == "white")) %>%
  glm(y ~ . -type, data=., family = "binomial")
p_hat_logit <- predict(glm_fit, newdata = test, type = "response")
y_hat_logit <- ifelse(p_hat_logit > 0.5, "white", "red") %>% factor
c<-confusionMatrix(y_hat_logit, test$type)
Type_Models <-  bind_rows(Type_Models,
                          data.frame(Model = "Logistic Regression - 11 Predictors",
                                     Accuracy = c$overall[1]))
#Using KNN to undertake the same process
x<- train %>% select(-quality, -type)
y <- train$type 
knn_fit <- knn3(x,y)
z <- test %>% select(-type,-quality)
y_hat_knn <- predict(knn_fit, z, type = "class")
c<-confusionMatrix(data = y_hat_knn, reference = test$type)$overall["Accuracy"]
Type_Models <-  bind_rows(Type_Models,
                          data.frame(Model = "K Nearest Neighbours (k=5)",
                                     Accuracy = c))
#finding the best value for K 
k <- seq(1, 100, 1)
accuracy <- map_df(k, function(k){
  fit <- knn3(x,y,k = k)
  y_hat <- predict(fit, z, type = "class")
  test_error <- confusionMatrix(data = y_hat, reference = test$type)$overall["Accuracy"]
  tibble(k=k,accuracy = test_error)
})
accuracy %>%
  ggplot(aes(k,accuracy))+
  geom_line()+
  geom_point()
k_non_cross <- which.max(accuracy$accuracy)
knn_fit <- knn3(x,y,k_non_cross)
z <- test %>% select(-type,-quality)
y_hat_knn <- predict(knn_fit, z, type = "class")
c <- confusionMatrix(data = y_hat_knn, reference = test$type)$overall["Accuracy"]
Type_Models <-  bind_rows(Type_Models,
                          data.frame(Model = "K Nearest Neighbours (k=9)",
                                     Accuracy = c))
#Naive Bayes
params <- train %>% 
  group_by(type) %>% 
  summarize(avg = mean(total_sulfur_dioxide), sd = sd(total_sulfur_dioxide))
params

pi <- train %>% summarize(pi=mean(type=="white")) %>% pull(pi)
pi

x <- test$total_sulfur_dioxide

f0 <- dnorm(x, params$avg[1], params$sd[1])
f1 <- dnorm(x, params$avg[2], params$sd[2])

p_hat_bayes <- f1*pi / (f1*pi + f0*(1 - pi))

tmp <- data %>% 
  mutate(x = round(total_sulfur_dioxide)) %>%
  group_by(x) %>%
  summarize(prob = mean(type == "white")) 
naive_bayes_curve <- data.frame(x = seq(min(tmp$x), max(tmp$x))) %>%
  mutate(p_hat = dnorm(x, params$avg[2], params$sd[2])*pi/
           (dnorm(x, params$avg[2], params$sd[2])*pi +
              dnorm(x, params$avg[1], params$sd[1])*(1-pi)))
tmp %>% 
  ggplot(aes(x, prob)) +
  geom_point() +
  geom_line(data = naive_bayes_curve,
            mapping = aes(x, p_hat), lty = 3) 
### Controlling Prevalence

y_hat_bayes <- ifelse(p_hat_bayes > 0.5, "white", "red")
sensitivity(data = factor(y_hat_bayes), reference = factor(test$type))

specificity(data = factor(y_hat_bayes), reference = factor(test$type))

p_hat_bayes_unbiased <- f1 * 0.5 / (f1 * 0.5 + f0 * (1 - 0.5)) 
y_hat_bayes_unbiased <- ifelse(p_hat_bayes_unbiased> 0.5, "white", "red")

sensitivity(data = factor(y_hat_bayes_unbiased), reference = factor(test$type))
specificity(data = factor(y_hat_bayes_unbiased), reference = factor(test$type))

qplot(x, p_hat_bayes_unbiased, geom = "line") + 
  geom_hline(yintercept = 0.5, lty = 2) + 
  geom_vline(xintercept = 67, lty = 2)

#QDA example
params <- train %>% 
  group_by(type) %>% 
  summarize(avg_1 = mean(total_sulfur_dioxide), avg_2 = mean(volatile_acidity), 
            sd_1= sd(total_sulfur_dioxide), sd_2 = sd(volatile_acidity), 
            r = cor(total_sulfur_dioxide, volatile_acidity))
params

train %>% mutate(type = factor(type)) %>% 
  ggplot(aes(total_sulfur_dioxide, volatile_acidity, fill = type, color=type)) + 
  geom_point(show.legend = FALSE) + 
  stat_ellipse(type="norm", lwd = 1.5)
x<- train %>% select(-quality)
z <- test %>% select(-type,-quality)
train_qda <- train(type ~ ., method = "qda", data = x)
y_hat <- predict(train_qda, z)
c <- confusionMatrix(data = y_hat, reference = test$type)$overall["Accuracy"]
Type_Models <-  bind_rows(Type_Models,
                          data.frame(Model = "Quadratic Discriminant Analysis",
                                     Accuracy = c))
x<- train %>% select(-quality)
z <- test %>% select(-type,-quality)
train_qda <- train(type ~ ., method = "lda", data = x)
y_hat <- predict(train_qda, z)
c <- confusionMatrix(data = y_hat, reference = test$type)$overall["Accuracy"]
Type_Models <-  bind_rows(Type_Models,
                          data.frame(Model = "Linear Discriminant Analysis",
                                     Accuracy = c))
#Using a calssification (desicion tree) model
x<- train %>% select(-quality)
z <- test %>% select(-type,-quality)
train_rpart <- train(type ~ .,
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
                     data = x)
plot(train_rpart)
c <- confusionMatrix(predict(train_rpart,z), test$type)$overall["Accuracy"]
Type_Models <-  bind_rows(Type_Models,
                          data.frame(Model = "Classification (Desicion) Tree",
                                     Accuracy = c))
#Random forests
train_rf <- randomForest(type ~ ., data=x)
c <- confusionMatrix(predict(train_rf,z), test$type)$overall["Accuracy"]
Type_Models <-  bind_rows(Type_Models,
                          data.frame(Model = "Random Forest",
                                     Accuracy = c))
train_rf_2 <- train(type ~ .,
                    method = "Rborist",
                    tuneGrid = data.frame(predFixed = 2,
                                          minNode = c(3, 50)),
                    data = x)
plot(train_rf_2)
c <- confusionMatrix(predict(train_rf_2,z), test$type)$overall["Accuracy"]
Type_Models <-  bind_rows(Type_Models,
                          data.frame(Model = "Random Forest - Tuned Params",
                                     Accuracy = c))


##############
#Creating models for the quality of the wine - utilising caret package to auto build
x <- train %>% select(-type)
z <- test %>% select(-type,-quality)
train_knn <- train(quality ~ ., method = "knn",
                  tuneGrid = data.frame(k = seq(1,10,1))
                  ,data = x)
ggplot(train_knn)+theme_bw()
c <- confusionMatrix(predict(train_knn,z), test$quality)$overall["Accuracy"]
Quality_Models <-   data.frame(Model = "K Nearest Neighbours",
                                     Accuracy = c)
train_qda <- train(quality ~ ., method = "qda", data = x)
c <- confusionMatrix(predict(train_qda,z), test$quality)$overall["Accuracy"]
Quality_Models <-   bind_rows(Quality_Models,
                              data.frame(Model = "Quadratic Discriminant Analysis",
                                         Accuracy = c))
train_lda <- train(quality ~ ., method = "lda", data = x)
c <- confusionMatrix(predict(train_lda,z), test$quality)$overall["Accuracy"]
Quality_Models <-   bind_rows(Quality_Models,
                              data.frame(Model = "Linear Discriminant Analysis",
                                         Accuracy = c))
train_dtree <- train(quality ~ ., method = "rpart",
                     tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
                     data = x)
c <- confusionMatrix(predict(train_dtree,z), test$quality)$overall["Accuracy"]
Quality_Models <-  bind_rows(Quality_Models,
                             data.frame(Model = "Classification (Desicion) Tree",
                                        Accuracy = c))
train_rf <- randomForest(quality ~ ., data=x)
c <- confusionMatrix(predict(train_rf,z), test$quality)$overall["Accuracy"]
Quality_Models <-  bind_rows(Quality_Models,
                             data.frame(Model = "Random Forest",
                                        Accuracy = c))
#Final model run for both problems run against the validation dataset using the full training dataset
set.seed(1)
x <- data_adjusted %>% select(-quality,-index)
z <- vali %>% select(-type,-quality)
set.seed(1)
train_rf <- randomForest(type ~ ., method = "Rborist",
                         tuneGrid = data.frame(predFixed = 2,
                                               minNode = c(3, 50)),
                         data = x)
c <- confusionMatrix(predict(train_rf,z), vali$type)$overall["Accuracy"]
Final_Models <-  data.frame(Model = "Random Forest - Tuned Params for the classification (wine type) problem",
                                        Accuracy = c)
set.seed(1)
x <- data_adjusted %>% select(-type,-index)
z <- vali %>% select(-type,-quality)
set.seed(1)
train_rf <- randomForest(quality ~ ., method = "Rborist",
                         tuneGrid = data.frame(predFixed = 2,
                                               minNode = c(3, 50)),
                         data = x)
c <- confusionMatrix(predict(train_rf,z), vali$quality)$overall["Accuracy"]
Final_Models <-  bind_rows(Final_Models,
                           data.frame(Model = "Random Forest - Tuned Params for the classification (quality type) problem",
                            Accuracy = c))
#Presenting the final models 
Final_Models
