library(xts)
library(forecast)
library(cansim)
library(tidyverse)
library(lubridate)
require(tseries)
library(sarima)
library(astsa)
library(tseries)
library(aTSA)
# Plot quarterly & monthly GDP 
X2raw = get_cansim_vector( c( 
  "monthly GDP (basic prices)" = "v65201210" ,
  "quarterly GDP (market prices)" = "v62305752" ) ,
  start_time = "1900-01-01" ) %>% 
  normalize_cansim_values() 
# (note correct vector code "v62305752" for quarterly GDP)

X2raw %>% filter( Date >= "2010-01-01") %>% 
  ggplot( aes( x = Date, y = VALUE, col = label ) ) +
  geom_line() + geom_point() + ylab("Chained (2012) dollars")
# (note (basic prices) = (market prices) - (tax & subsidies), 
#  and that is why monthly data values are lower)

library(xts) # better ts objects, including quarterly series

# Quartely data 
Q = X2raw %>%  
  filter( VECTOR == "v62305752" ) %>% 
  # find year & quarter
  mutate( Y = year( Date ), Q = quarter( Date ),
          index = yearqtr( Y + Q/4 ) ) %>%  
  xts( x=.$VALUE, order.by =.$index) 
# plot(Q)

# Monthly data 
M = X2raw %>%  
  filter( VECTOR == "v65201210" ) %>% 
  # find year, quarter, month, and month-in-quarter
  mutate( Y = year( Date ), Q = quarter( Date ), 
          index = yearqtr( Y + Q/4 ), 
          M = month( Date ),
          MinQ = paste( "M", M%%3, sep="" ) ) %>%  
  # spread monthly data into 3 columns one for each month-in-quarter
  pivot_wider(id_cols = index, names_from = MinQ, 
              values_from = VALUE ) %>% 
  # take lag for M0
  mutate( M0 = lag(M0) ) %>% 
  xts( x=.[,c("M0","M1","M2")], order.by =.$index) 
#plot(M$M0); plot(M)

# combine & align quarterly series with expanded monthly data
X2 = merge(Q, M, join = "inner" )
plot(X2)

# ACF of differenced GDP seems 
acf( diff(X2$Q), na.action = na.pass )
pacf( diff(X2$Q), na.action = na.pass )

# model w/ past information only 
arima( x = X2$Q, order=c(2,1,1) )

# model w/ past information + M0 (start of quarter forecast)
arima( x = X2$Q, order=c(1,1,0),  xreg = X2$M0)

# model w/ past information + M0 + M1 (1st month nowcast)
arima( x = X2$Q, order=c(1,1,0),  xreg = X2[,c("M0","M1")])

# model w/ past information + M0 + M1 + M2 (2nd month nowcast)
arima( x = X2$Q, order=c(1,1,0),  xreg = X2[,c("M0","M1","M2")])

# non log version of the AIC values being compared
arima(x=X2$Q, order = c(1,2,1),xreg = X2[,c("M0","M1","M2")])

arima(x=X2$Q, order = c(1,1,1), xreg = X2[,c("M0","M1","M2")])

arima(x=X2$Q, order = c(1,4,1),xreg = X2[,c("M0","M1","M2")])

#arima(x=log(X2$Q), order = c(1,2,1), method = "CSS")


# compare AIC values
arima(x=log(X2$Q), order = c(1,2,1),xreg = X2[,c("M0","M1","M2")])

arima(x=log(X2$Q), order = c(1,1,1),xreg = X2[,c("M0","M1","M2")])

arima(x=log(X2$Q), order = c(1,3,1),xreg = X2[,c("M0","M1","M2")])

arima(x=log(X2$Q), order = c(1,0,1),xreg = X2[,c("M0","M1","M2")])

# compare AIC values for the transformed values
firstModel = arima(x=log(X2$Q), order = c(1,2,1),xreg = X2[,c("M0","M1","M2")])
Yhat = log(X2$Q) - firstModel$residuals #in-sample
arima(x=Yhat, order = c(1,2,1),xreg = X2[,c("M0","M1","M2")])

secondModel = arima(x=log(X2$Q), order = c(0,1,1),xreg = X2[,c("M0","M1","M2")])
Yhat2 = log(X2$Q) - secondModel$residuals #in-sample
arima(x=Yhat2, order = c(0,1,1),xreg = X2[,c("M0","M1","M2")])

thirdModel = arima(x=log(X2$Q), order = c(1,1,1),xreg = X2[,c("M0","M1","M2")])
Yhat3 = log(X2$Q) - thirdModel$residuals #in-sample
arima(x=Yhat3, order = c(1,1,1),xreg = X2[,c("M0","M1","M2")])

fourthModel = arima(x=log(X2$Q), order = c(1,2,2),xreg = X2[,c("M0","M1","M2")])
Yhat4 = log(X2$Q) - fourthModel$residuals #in-sample
arima(x=Yhat4, order = c(1,2,2),xreg = X2[,c("M0","M1","M2")])

fifthModel = arima(x=log(X2$Q), order = c(1,1,2),xreg = X2[,c("M0","M1","M2")])
Yhat5 = log(X2$Q) - fifthModel$residuals #in-sample
arima(x=Yhat5, order = c(1,1,2),xreg = X2[,c("M0","M1","M2")])

sixthModel = arima(x=log(X2$Q), order = c(2,1,1),xreg = X2[,c("M0","M1","M2")])
Yhat6 = log(X2$Q) - sixthModel$residuals #in-sample
arima(x=Yhat6, order = c(2,1,1),xreg = X2[,c("M0","M1","M2")])

seventhModel = arima(x=log(X2$Q), order = c(2,2,1),xreg = X2[,c("M0","M1","M2")])
Yhat7 = log(X2$Q) - seventhModel$residuals #in-sample
arima(x=Yhat7, order = c(2,2,1),xreg = X2[,c("M0","M1","M2")])

eighthModel = arima(x=log(X2$Q), order = c(1,0,1),xreg = X2[,c("M0","M1","M2")])
Yhat8 = log(X2$Q) - eighthModel$residuals #in-sample
arima(x=Yhat8, order = c(1,0,1),xreg = X2[,c("M0","M1","M2")])

ninethModel = arima(x=log(X2$Q), order = c(1,1,0),xreg = X2[,c("M0","M1","M2")])
Yhat9 = log(X2$Q) - ninethModel$residuals #in-sample
arima(x=Yhat9, order = c(1,1,0),xreg = X2[,c("M0","M1","M2")])


# when doing the log, the ARIMA model with parameters 1,1,1 has the lowest value at -811.08 so going forward
# we will take a closer look at how ARIMA(1,1,1) holds with the other ARIMA models

# when doing the log, the ARIMA model with parameters 1,1,1 has the lowest value at -811.08 so going forward
# we will take a closer look at how ARIMA(1,1,1) holds with the other ARIMA models

# forecast for ARIMA(1,1,1) model
sarima(log(X2$Q),1,1,1)

sarima(log(X2$Q),1,2,1)

sarima(log(X2$Q),1,3,1)

sarima(log(X2$Q),1,0,1)

# the ARIMA(1,1,1) model has the QQ-plot showing it has Normality except two points one which looks to be an
# outlier, the other falling just short on the outside of the line and the prediction shadow. There are no 
# significant auto correlations in the acf (the resiudals are uncorrelated), the p values after lag 5 are above 5% which suggests the WN null
# hypothesis is not rejected. It also tells us that the null which is also if the data is independently distributed, so any noise is due to randomness of how the data is sampled.
# The mean of the resiudals seem to also be centered at 0. Other than the start and 
# between 10 and 15 the values dip noticeably which looks the part as the QQ plot has outliers that account to this.

# the ARIMA(1,2,1) model has the p value being >5% around lag 10, further dips in the std resid plot and more outliers
# in the qq plot. The ARIMA(1,0,1) model seems to have one significant auto correlation, the mean for the std residual
# graph looks too distant from 0 (the spikes are  much larger than the the other two listed), and the p values dip over
# 5% after lag 10. The ARIMA(1,3,1) model has two significant auto correlations and the p-value never jumps over 5%.

# so far the ARIMA(1,1,1) model seems to have minor issues for the diagnostic test compared to the others.



forecast1 = sarima.for(log(X2$Q),n.ahead=20, 2,1,2)
forecast1$pred
forecast1$se
sarima.for(log(X2$Q),n.ahead=20, 1,1,1)
sarima.for(log(X2$Q),n.ahead=20, 1,2,1)
sarima.for(log(X2$Q),n.ahead=20, 1,0,1)
sarima.for(log(X2$Q),n.ahead=20, 1,3,1)

# to fully vaildate the ARIMA(1,1,1) model, the confidence interval is an increasing line with a narrow region for the region
# the ARIMA(1,2,1) has a wide region, the ARIMA(1,0,1) has a wide no decreasing region (levels off) and finally the confidence
# region of the ARIMA(1,3,1) model has the widest confidence interval region by a noticeable margin.

# the AIC, the diagnostics and the confidence region graphs all favour the ARIMA(1,1,1) model. Going forward the ARIMA(1,1,1)
# model will have the most accurate now cast compared to the others.

# the below code confirms AR(1) model should be used so the ARIMA model has AR(1) included is used
ar(X2$Q, order.max = 20,aic = TRUE)
# disregard the auto.arima
auto.arima(X2$Q)
# MA code to look at
ma(X2$Q,order(1))
MA(X2$Q)
# the p-value is quite large which could suggest white noise in our model that's mistaken as AR
# this also tells us thata unit vector doesnt exist and that the data is stationary.
adf.test(X2$Q)



# to be done
#Cross Validation
monthly = ts(prob2[["data"]][["VALUE"]][39:153],start=c(2010,1),frequency=12)
quarterly = ts(prob2[["data"]][["VALUE"]][1:38],start=c(2010,1),frequency=4)

out1=lm(monthly~time(monthly))
out2=lm(quarterly~time(quarterly))


Training.samples <- Q%>%createDataPartition(p=0.8,list=FALSE)
training <- Q[Training.samples,]
test <- Q[-Training.samples,]
model<- lm(training[,1]~.,data=training)
predict <- model%>%predict(test)
data.frame(R2 = R2(predict,test[,1]),
           RMSE = RMSE(predict,test[,1]),
           MAE = MAE(predict, test[,1]))
RMSE(predict,test[,1])/MAE(predict, test[,1])

train.con = trainControl(method = "LOOCV")
modelLOC <- train(Q[,1]~., data=Q, method="nb", trControl = train.con)


# plot differencing data 
Q.diff = diff(X2$Q)
plot(Q.diff)
acf(diff(log(X2$Q)), na.action = na.pass)
pacf(diff(log(X2$Q)), na.action = na.pass)


am.fit = arima(x=X2$Q/10^9, order=c(1,1,1)) # Get arima model

pred <- predict(am.fit, n.ahead = 1) # predictions

ts.plot(X2$Q, pred$pred, lty = c(1,3), col=c(4,2), main="ARIMA(1,1,1) with M0, M1, M2 vs Predictions") # plot the data vs predictions

# cross validation
# Prediction/Forecasting
d = X2$Q / 10^9
testingttt = Arima(d, order = c(1,2,1), include.drift  =TRUE)
forecast(testingttt, h =1)

# Out of sample predictions
pn=nrow(X2)
y = X2$Q/10^9
X = cbind( 1:n, X2[, c("M0","M1","M2")]/10^9)
nCV = 20 # number of cross validation errors
y_fe = y_fe. = rep(NA,n) # Y forecast error 
for(i in (n - nCV:1) ){
  out = arima( y[1:i], order = c(2,1,2), xreg = X[1:i,]) 
  y_fe[i+1] = predict( out, newxreg = X[i+1,])$pred

}
# Cross-Validated Performance 

(MSPE = mean( (y - y_fe)^2, na.rm = TRUE ))

(MAPE = mean( abs(y - y_fe)/abs(y), na.rm = TRUE ))
########################################################
# another analysis ( NOWCASTING)
# Plot quarterly & monthly GDP 
GDPraw = get_cansim_vector( c( 
  "monthly GDP (basic prices)" = "v65201210" ,
  "quarterly GDP (market prices)" = "v62305752" ) ,
  start_time = "1900-01-01" ) %>% 
  normalize_cansim_values() 
GDPraw %>% filter( Date >= "2010-01-01") %>% 
  ggplot( aes( x = Date, y = VALUE, col = label ) ) +
  geom_line() + geom_point() + ylab("Chained (2012) dollars")

Quaterly_raw = get_cansim_vector( c( 
  "Final domestic demand" = "v62305753",
  "Final consumption expenditure" = "v62305723",
  "Household final consumption expenditure" = "v62305724",
  "Services" = "v62305729",
  "Exports of goods" = "v62305746",
  "Imports of goods" = "v62305749") ,
  start_time = "1900-01-01" ) %>% 
  normalize_cansim_values() 
Quaterly_raw %>% filter( Date >= "2010-01-01") %>% 
  ggplot( aes( x = Date, y = VALUE, col = label ) ) +
  geom_line() + geom_point() + ylab("Chained (2012) dollars")

Monthly_raw = get_cansim_vector( c( 
  "monthly survey of large retailers (Unadjusted, value oriented)" = "v822787",
  "monthly building permit (Unadjusted, value oriented)" = "v121293394",
  "Central government operations" = "v86822777",
  "monthly export (customs-Seasonally adjusted)" = "v1001826959",
  "monthly import (customs-Seasonally adjusted)" = "v1001826347",
  "monthly labour force (15 years and over)" = "v2091072") ,
  start_time = "1900-01-01" ) %>% 
  normalize_cansim_values() 

# Quartely GDP data 
Q = GDPraw %>%  
  filter( VECTOR == "v62305752" ) %>% 
  mutate( Y = year( Date ), Q = quarter( Date ),
          index = yearqtr( Y + Q/4 ) ) %>%  
  xts( x=.$VALUE, order.by =.$index) 

# Monthly GDP data 
M = GDPraw %>%  
  filter( VECTOR == "v65201210" ) %>% 
  mutate( Y = year( Date ), Q = quarter( Date ), 
          index = yearqtr( Y + Q/4 ), 
          M = month( Date ),
          MinQ = paste( "M", M%%3, sep="" ) ) %>%  
  pivot_wider(id_cols = index, names_from = MinQ, 
              values_from = VALUE ) %>% 
  mutate( M0 = lag(M0) ) %>% 
  xts( x=.[,c("M0","M1","M2")], order.by =.$index) 


# Expenditure data
# Final domestic demand
Qua_FDD = Quaterly_raw %>%  
  filter( VECTOR == "v62305753" ) %>% 
  mutate( Y = year( Date ), Q = quarter( Date ),
          index = yearqtr( Y + Q/4 ) ) %>%  
  xts( x=.$VALUE, order.by =.$index) 

# Final consumption expenditure
Qua_FCE = Quaterly_raw %>%  
  filter( VECTOR == "v62305723" ) %>% 
  mutate( Y = year( Date ), Q = quarter( Date ),
          index = yearqtr( Y + Q/4 ) ) %>%  
  xts( x=.$VALUE, order.by =.$index) 

# Household final consumption expenditure
Qua_HFC = Quaterly_raw %>%  
  filter( VECTOR == "v62305724" ) %>% 
  mutate( Y = year( Date ), Q = quarter( Date ),
          index = yearqtr( Y + Q/4 ) ) %>%  
  xts( x=.$VALUE, order.by =.$index) 

# Services
Qua_Ser = Quaterly_raw %>%  
  filter( VECTOR == "v62305729" ) %>% 
  mutate( Y = year( Date ), Q = quarter( Date ),
          index = yearqtr( Y + Q/4 ) ) %>%  
  xts( x=.$VALUE, order.by =.$index) 

# Exports of goods
Qua_E = Quaterly_raw %>%  
  filter( VECTOR == "v62305746" ) %>% 
  mutate( Y = year( Date ), Q = quarter( Date ),
          index = yearqtr( Y + Q/4 ) ) %>%  
  xts( x=.$VALUE, order.by =.$index) 

# Imports of goods
Qua_I = Quaterly_raw %>%  
  filter( VECTOR == "v62305749" ) %>% 
  mutate( Y = year( Date ), Q = quarter( Date ),
          index = yearqtr( Y + Q/4 ) ) %>%  
  xts( x=.$VALUE, order.by =.$index) 

# Merging Quaterly datas
Qua = merge(Q, Qua_FDD, Qua_FCE, Qua_HFC, Qua_Ser, Qua_E, Qua_I)

# Survey from the retailers to calculate the consumption
Mon_C = Monthly_raw %>%  
  filter( VECTOR == "v822787" ) %>%
  mutate( Y = year( Date ), Q = quarter( Date ), 
          index = yearqtr( Y + Q/4 ), 
          M = month( Date ),
          MinQ = paste( "Mon_C", M%%3, sep="" ) ) %>%  
  pivot_wider(id_cols = index, names_from = MinQ, 
              values_from = VALUE ) %>% 
  mutate( Mon_C0 = lag(Mon_C0) ) %>% 
  xts( x=.[,c("Mon_C0","Mon_C1","Mon_C2")], order.by =.$index) 

# Building permits as part of investment
Mon_B = Monthly_raw %>%  
  filter( VECTOR == "v121293394" ) %>%
  mutate( Y = year( Date ), Q = quarter( Date ), 
          index = yearqtr( Y + Q/4 ), 
          M = month( Date ),
          MinQ = paste( "Mon_B", M%%3, sep="" ) ) %>%  
  pivot_wider(id_cols = index, names_from = MinQ, 
              values_from = VALUE ) %>% 
  mutate( Mon_B0 = lag(Mon_B0) ) %>% 
  xts( x=.[,c("Mon_B0","Mon_B1","Mon_B2")], order.by =.$index) 

# Central Government operations expenditure
Mon_G = Monthly_raw %>%  
  filter( VECTOR == "v86822777" ) %>%
  mutate( Y = year( Date ), Q = quarter( Date ), 
          index = yearqtr( Y + Q/4 ), 
          M = month( Date ),
          MinQ = paste( "Mon_G", M%%3, sep="" ) ) %>%  
  pivot_wider(id_cols = index, names_from = MinQ, 
              values_from = VALUE ) %>% 
  mutate(Mon_G0 = lag(Mon_G0) ) %>% 
  xts( x=.[,c("Mon_G0","Mon_G1","Mon_G2")], order.by =.$index) 

# Import
Mon_I = Monthly_raw %>%  
  filter( VECTOR == "v1001826347" ) %>%
  mutate( Y = year( Date ), Q = quarter( Date ), 
          index = yearqtr( Y + Q/4 ), 
          M = month( Date ),
          MinQ = paste( "Mon_I", M%%3, sep="" ) ) %>%  
  pivot_wider(id_cols = index, names_from = MinQ, 
              values_from = VALUE ) %>% 
  mutate( Mon_I0 = lag(Mon_I0) ) %>% 
  xts( x=.[,c("Mon_I0","Mon_I1","Mon_I2")], order.by =.$index) 

# Export
Mon_E = Monthly_raw %>%  
  filter( VECTOR == "v1001826959" ) %>%
  mutate( Y = year( Date ), Q = quarter( Date ), 
          index = yearqtr( Y + Q/4 ), 
          M = month( Date ),
          MinQ = paste( "Mon_E", M%%3, sep="" ) ) %>%  
  pivot_wider(id_cols = index, names_from = MinQ, 
              values_from = VALUE ) %>% 
  mutate( Mon_E0 = lag(Mon_E0) ) %>% 
  xts( x=.[,c("Mon_E0","Mon_E1","Mon_E2")], order.by =.$index) 

# Labor force as a somewhat outside factor
Mon_L = Monthly_raw %>%  
  filter( VECTOR == "v2091072" ) %>%
  mutate( Y = year( Date ), Q = quarter( Date ), 
          index = yearqtr( Y + Q/4 ), 
          M = month( Date ),
          MinQ = paste( "Mon_L", M%%3, sep="" ) ) %>%  
  pivot_wider(id_cols = index, names_from = MinQ, 
              values_from = VALUE ) %>% 
  mutate( Mon_L0 = lag(Mon_L0) ) %>% 
  xts( x=.[,c("Mon_L0","Mon_L1","Mon_L2")], order.by =.$index) 

# Merging Monthly datas
Mon = merge(M, Mon_C, Mon_B, Mon_G, Mon_I, Mon_E, Mon_L)

Merged = merge(Qua, Mon) # Merging the data

# model w/ GDP datas only 
arima( x = Merged$Q, order=c(1,1,1))
arima( x = Merged$Q, order=c(1,2,1),  xreg = Merged$M0)
arima( x = Merged$Q, order=c(1,2,1),  xreg = Merged[,c("M0","M1")])
arima( x = Merged$Q, order=c(1,2,1),  xreg = Merged[,c("M0","M1","M2")])

# model w/ other datas
# model w/ Quaterly Expenditure datas
a <- arima( x = Merged$Q, order=c(1,2,1),  xreg = Merged[,c("M0","M1","M2",
                                                            "Qua_E", "Qua_FCE", "Qua_FDD",
                                                            "Qua_HFC","Qua_I", "Qua_Ser")])
pred <- predict(a, n.ahead = 2, newxreg = Merged[,c("M0","M1","M2",
                                                    "Qua_E", "Qua_FCE", "Qua_FDD",
                                                    "Qua_HFC","Qua_I", "Qua_Ser")]) # predictions

ts.plot(Merged$Q, pred$pred, lty = c(1,3), col=c(4,2), main="Model with Quarterly Expenditure Series")
#pred$se
# model w/ Quaterly Expenditure datas
b = arima( x = Merged$Q, order=c(1,2,1),  xreg = Merged[,c("M0","M1","M2",
                                                           "Mon_B0","Mon_B1","Mon_B2",
                                                           "Mon_C0","Mon_C1","Mon_C2",
                                                           "Mon_E0","Mon_E1","Mon_E2",
                                                           "Mon_G0","Mon_G1","Mon_G2",
                                                           "Mon_I0","Mon_I1","Mon_I2",
                                                           "Mon_L0","Mon_L1","Mon_L2")])
pred2 <- predict(b, n.ahead = 2, newxreg = Merged[,c("M0","M1","M2",
                                                     "Mon_B0","Mon_B1","Mon_B2",
                                                     "Mon_C0","Mon_C1","Mon_C2",
                                                     "Mon_E0","Mon_E1","Mon_E2",
                                                     "Mon_G0","Mon_G1","Mon_G2",
                                                     "Mon_I0","Mon_I1","Mon_I2",
                                                     "Mon_L0","Mon_L1","Mon_L2")])

ts.plot(Merged$Q, pred2$pred, lty = c(1,3), col=c(4,2), main="Model with Monthly Expenditure Series")
#pred2
###############################################