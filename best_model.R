# please uncomment following three lines if the packages have not been installed on your machine
# install.packages(LiblineaR)
# install.packages(Matrix)
# install.packages(SparseM)
library(LiblineaR)
library(Matrix)
library(SparseM)

# load training set
tr_dtm = as(readMM(file = './feature_vectors/training_combine_dtm.mtx'), 'matrix.csr')
tr_labels = read.csv('./feature_vectors/training_labels.csv')$Class

# learn the model
model.best = LiblineaR(data = tr_dtm, target = tr_labels, type = 2, cost = 0.1, epsilon = 1e-5)
# remove training to save memory
rm(tr_dtm)

# load the testing set
te_dtm = as(readMM(file = './feature_vectors/testing_combine_dtm.mtx'), 'matrix.csr')

# predict labels for testing data
te_predict = predict(model.best, te_dtm)$predictions
rm(te_dtm)

# store the predict results
te_predict.m <- matrix(, nrow=length(te_predict), ncol=2)

id <- c()
for (i in 1:length(te_predict)){
    id <- c(id, paste('te_doc_', i, sep = ''))
}

te_predict.m[,1] <- id
te_predict.m[,2] <- as.character(te_predict)

# store predict results into hard disk as the same directory as this file
write.table(te_predict.m, './data/testing_labels_pred.txt', append = F, row.names = F, col.names = F, quote = F)