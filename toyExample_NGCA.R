
library(steadyICA)
library(neuRosim)
source('jngcaFunctions.R')

#########
# Simulated Data:
# Spatial components used in Section 5 in Risk et al 2018 LNGCA paper 
simData = SimFMRI123(var.inactive = 0,snr=0.2) 
# the columns of simData$S are vectorized images:
# These are the true components:
par(mfrow=c(1,3))
image(matrix(simData$S[,1],33))
image(matrix(simData$S[,2],33))
image(matrix(simData$S[,3],33))

cov(simData$S)
cor(simData$S)
head(simData$S)
apply(simData$S,2,mean)

image(matrix(simData$whitened.X[,1],33,33))
image(matrix(simData$whitened.X[,2],33,33))
image(matrix(simData$whitened.X[,3],33,33))

#########################
# 1.  Unmix using the logistic non-linearity:
# this has 30 restarts:
# in other words, the model is estimate THIRTY times and then the estimate with the maximum objective
# function value among all estimates is retained:
a = Sys.time()
estX_logis = mlcaFP(xData = simData$whitened.X, n.comp = 3, whiten = "none", restarts.pbyd = 30, distribution='logistic')

estX_logis = mlcaFP(xData = simData$whitened.X, n.comp = 3, whiten = "none", W.list = list(diag(50)[,1:3]), distribution='logistic')

Sys.time() - a

# Ws is a semiorthogonal:
t(estX_logis$Ws)%*%estX_logis$Ws

image(matrix(estX_logis$S[,1],33,33))
image(matrix(estX_logis$S[,2],33,33))
image(matrix(estX_logis$S[,3],33,33))

# NOTE: LNGCA is only identifiable up to sign and permutation;
# you can match as follows:
temp = matchICA(S = estX_logis$S[,1:3], template = simData$S)
image(matrix(temp[,1],33,33))
image(matrix(temp[,2],33,33))
image(matrix(temp[,3],33,33))

# recover rows (time courses) of mixing matrix:
Mx_logis = est.M.ols(sData = estX_logis$S, xData = simData$X)
# discrepancy measure for sign and permutation invariance:
frobICA(Mx_logis, simData$Ms, standardize = TRUE)


frobICA(S1=estX_logis$S, S2=simData$S, standardize = TRUE)
#frobICA(S1=estX_logis$S, S2=simData$S)






###
# For comparison, here is the estimation using PCA+ICA:
est_infomaxICA = infomaxICA(X = simData$X,n.comp = 3, whiten = TRUE, restarts=30,eps = 1e-06)
# note: here W is square; it's also not constrained to be orthogonal in the infomax algorithm but the important difference from the previous algorithm is in the dimension reduction
est_infomaxICA$W
frobICA(S1=est_infomaxICA$S, S2=simData$S, standardize = TRUE)

par(mfrow=c(1,3))
image(matrix(est_infomaxICA$S[,1],33,33))
image(matrix(est_infomaxICA$S[,2],33,33))
image(matrix(est_infomaxICA$S[,3],33,33))
# as you can see, it is terrible:)


########################
#  2. JB: combination of third and fourth cumulants
a = Sys.time()
estX_JB = mlcaFP(xData = simData$whitened.X, n.comp = 3, whiten = "none", restarts.pbyd = 30, distribution='JB')
Sys.time() - a

Mx_JB = est.M.ols(sData = estX_JB$S, xData = simData$X)
frobICA(Mx_JB, simData$Ms, standardize=TRUE)

# Run the above code a few times to see how much the figures change
temp = matchICA(S = estX_JB$S[,1:3], template = simData$S)

par(mfrow=c(1,3))
image(matrix(temp[,1],33,33))
image(matrix(temp[,2],33,33))
image(matrix(temp[,3],33,33))
# these figures should not change if we are finding the argmax
