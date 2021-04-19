data(GermanCredit)

loan_default<-GermanCredit[,1:10] #Use only columns(features) 1 to 10

View(loan_default) # Run this to View the dataset

# The "Class" column in the dataset will be our target variable
# and a class with "bad" means a customer will defaulted on loan


# Now let us divide the data into train and test samples
# And make a machine learning algorithm to learn the pattern
# This give us a learned algorithm that can be use for future detection

len<-dim(loan_default)[1] #Get the size of our dataset (1000 rows)

len

train<- sample(1:len , 0.8*len) #Select 80% of the dataset

TrainData<-loan_default[train,] #Assign selected 80% as the TrainData

TestData<-loan_default[-train,] # Assign the remaining 20% as TestData

View(TrainData) #View Training Data (This would be use to train our algorithm)

View(TestData) #View Test Data (This would be use for future detection to test performance of our algorithm)

######## We are to seperate the features from the target ########
TrainData_features<-TrainData[-10]

TestData_features<-TestData[-10]

View(TrainData_features)#Train data features

View(TestData_features) #Test data features

####### MACHINE LEARNING #################
# let us build a classification model to learn the training data "features" and target
# with the "Random Forest" algorithm.

library(randomForest) #Load the Random Forest

classifier = randomForest(x = TrainData_features,
                          y = TrainData$Class,
                          ntree = 100, random_state = 0,proximity = TRUE)

print(classifier) 

plot(classifier)

#Evaluate variable importance
importance(classifier)

# Now, let us do classification on the test data applying our classifier 

loan_status<-predict(classifier,newdata=TestData_features)

head(loan_status,20) #View the prediction

View(loan_status)

#Construct the Confusion Matrix of Classifier Performance ####
table(loan_status, TestData$Class) 



