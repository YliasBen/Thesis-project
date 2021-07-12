# Thesis-project

---
title: "Thesis_code"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Import data
```{r}
library(RWeka)
df_raw <- read.arff("C://Users//Ylias//Downloads//chronic_kidney_disease_full.arff")
```


## Data wrangling
```{r}
library(RWeka)
library ("dplyr")
library(mice)

#str(df_test)
init = mice(df_raw, maxit=0) 
meth = init$method
predM = init$predictorMatrix

 
meth[c("rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "appet", "pe", "ane" )]="logreg" 

set.seed(103)
imputed = mice(df_raw, method=meth, predictorMatrix=predM, m=5)
df<- complete(imputed)


#summary(df)
```
In this step the Mice package is used to impute values for the missing values. Imputing missing values results in less bias than removing missing values, thus a better approach. Summary is used to check if the imputed values are logical/realistic.


```{r}
#install.packages('fastDummies')
library(fastDummies)

df <- dummy_cols(df, select_columns = c('sg', 'al','su','rbc','pc','pcc','ba','htn','dm','cad','appet', 'pe','ane','class'),
                  remove_selected_columns = TRUE)


library(dplyr)
df <- select(df, -c("sg_1.005", "al_0", "su_0", "rbc_abnormal", "pc_abnormal", "pcc_notpresent", "ba_notpresent", "htn_no", "dm_no", "cad_no", "appet_poor", "pe_no", "ane_no", "class_notckd"))
```
In this step the nominal attributes of the dataset are transformed into dummy variables with the "class_ckd" as outcome. Furthermore, the dummy variables that are deleted are the dummy reference categories (K-1).


```{r}
df[1:11]<-scale(df[1:36])

#apply(df, 2, sd)
#apply(df, 2, mean)

dt= sort(sample(nrow(df), nrow(df)*.7))
df_train <- df[dt,]
df_test <- df[-dt,]


y_complete <- df[, "class_ckd"]
y_train <- df_train[, "class_ckd"]
y_test <- df_test[, "class_ckd"]

```
In this step the numerical values are scaled, in order to bring the data on the same scale (essential when fitting the model). After scaling, the mean of the continue variables are around zero and the standard deviation is 1, thus the standardization succeeded. Furthermore the data set is divided into a training set and a test set in order to check the generalizability of the model (train n=280 (70%) and test n= 120 (30%)). In addition, three Y data sets are created for the whole data, test and train. We will need these Y's later on to plot the fit against the data set (graphical posterior predictive check) 



## Data analysis

### Bayesian Lasso

```{r}
ndraws <- 1000
lasso_1_dist <- rep(NA, ndraws)
for(i in 1:ndraws){
  lasso_scale = 1
  lasso_df = 1
  lasso_inv_lambda <- rchisq(1, df = lasso_df)
  lasso_1_dist[i] <- rmutil::rlaplace(1, m=0, s=lasso_scale * lasso_inv_lambda)
 }

plot(density(lasso_1_dist))


ndraws <- 1000
lasso_2_dist <- rep(NA, ndraws)
for(i in 1:ndraws){
  lasso_scale = 0.1
  lasso_df = 1
  lasso_inv_lambda <- rchisq(1, df = lasso_df)
  lasso_2_dist[i] <- rmutil::rlaplace(1, m=0, s=lasso_scale * lasso_inv_lambda)
}
plot(density(lasso_2_dist))


ndraws <- 1000
lasso_3_dist <- rep(NA, ndraws)
for(i in 1:ndraws){
  lasso_scale = 1
  lasso_df = 5
  lasso_inv_lambda <- rchisq(1, df = lasso_df)
  lasso_3_dist[i] <- rmutil::rlaplace(1, m=0, s=lasso_scale * lasso_inv_lambda)
}
plot(density(lasso_3_dist))

myData1 <- data.frame(lasso_1=lasso_1_dist,
                     lasso_2=lasso_2_dist)
                     

myData2 <- data.frame(lasso_1=lasso_1_dist,                    
                      lasso_3=lasso_3_dist)

myData3 <- data.frame(lasso1=lasso_1_dist,
                      lasso2=lasso_2_dist,
                      lasso3=lasso_3_dist)


library(ggplot2);library(reshape2)
data<- melt(myData1) 
ggplot(data) + aes(x=value, lines=variable, col = variable) + geom_density(alpha=0.25) + xlim(-10, 10)

data<- melt(myData2) 
ggplot(data) + aes(x=value, lines=variable, col = variable) + geom_density(alpha=0.25) + xlim(-10, 10)


data<- melt(myData3) 
p<- ggplot(data) + aes(x=value, lines=variable, col = variable) + geom_density(alpha=0.25) + xlim(-8, 8) + ylim(0, 14)
p + labs(title = "Prior distributions", col = "Priors")
```
In this step we plot the lasso distribution based on different df and scale. The purpose of this step is to understand how the distribution changes based on these parameters. For the hyperlasso, the degrees of freedom need to be specified with smaller degrees of freedom resulting in a heavier-tailed prior.The scale parameter influences how spread out the prior will be (chapter 4)

#### lasso with brms
```{r}
library(brms)
library(rstantools)
library(jtools)
library(ggstance)
library(broom.mixed)
library(bayesplot)
library(ggplot2)
library(rstanarm)

fit_L1 = brm(formula = class_ckd ~ age + bp + bgr + bu + sc + sod+ pot + hemo+ pcv + wbcc+ rbcc + sg_1.010 + sg_1.015 + sg_1.020 + sg_1.025 + al_1 + al_2 + al_3 + al_4 + al_5 + su_1 + su_2 + su_3 + su_4 + su_5 + rbc_normal + pc_normal + pcc_present + ba_present + htn_yes + dm_yes + cad_yes + appet_good + pe_yes + ane_yes ,
             family = bernoulli(link = "logit"),
             data=df_train, 
             prior=set_prior('lasso(df = 1, scale = 1)', class = 'b'),
             algorithm= 'sampling', iter= 4000, warmup = 500)

  summary(fit_L1)
  save("fit_L1", file = "fit_L1.RData")
  #load("fit1.RData")
  #plot_summs(fit_L1) 
  #yrep_L1 <- posterior_predict(fit_L1, draws = 50)
  #ppc_dens_overlay(y_train, yrep_L1 )
  
  
  
fit_L2 = brm(formula = class_ckd ~ age + bp + bgr + bu + sc + sod+ pot + hemo+ pcv + wbcc+ rbcc + sg_1.010 + sg_1.015 + sg_1.020 + sg_1.025 + al_1 + al_2 + al_3 + al_4 + al_5 + su_1 + su_2 + su_3 + su_4 + su_5 + rbc_normal + pc_normal + pcc_present + ba_present + htn_yes + dm_yes + cad_yes + appet_good + pe_yes + ane_yes ,
             family = bernoulli(link = "logit"),
             data=df_train, 
             prior=set_prior('lasso(df = 1, scale = 0.1)', class = 'b'),
             algorithm= 'sampling', iter= 4000, warmup = 500)

  summary(fit_L2)
  save("fit_L2", file = "fit_L2.RData")
  #load("fit_L2.RData")

  
  
fit_L3 =  brm(formula = class_ckd ~ age + bp + bgr + bu + sc + sod+ pot + hemo+ pcv + wbcc+ rbcc + sg_1.010 + sg_1.015 + sg_1.020 + sg_1.025 + al_1 + al_2 + al_3 + al_4 + al_5 + su_1 + su_2 + su_3 + su_4 + su_5 + rbc_normal + pc_normal + pcc_present + ba_present + htn_yes + dm_yes + cad_yes + appet_good + pe_yes + ane_yes ,
             family = bernoulli(link = "logit"),
             data=df_train, 
             prior=set_prior('lasso(df = 5, scale = 1)', class = 'b'),
             algorithm= 'sampling', iter= 4000, warmup = 500)

  summary(fit_L3)
  save("fit_L3", file = "fit_L3.RData")
  #load("fit_L3.RData")    
  
  
```
In the step above, three Bayesian lasso models are created and saved based on different hyperparameter values. Number of iterations are the number of samples of the MCMC sampling. 




```{r}
visModel <- function(varNum, varName){
    post1 <- as.matrix(fit_L1)
    post2 <- as.matrix(fit_L2)
    post3 <- as.matrix(fit_L3)
    Lasso1 <- post1[, varNum]
    Lasso2 <- post2[, varNum]
    Lasso3 <- post3[, varNum]
    x <- data.frame(Lasso1, Lasso2, Lasso3)
    data<- melt(x)
    p <- ggplot(data,aes(x=value, color=variable))+ labs(title = "Variable distribution", subtitle= (varName)) + geom_density(alpha=0.25)
    p + labs(col = "Models")
}



visModel(27, "rbc_normal") #largest coefficient
visModel(22, "su_1") #smallest coefficient 
```
The function visModel above is a function to visualize the density distribution for specific coefficients per lasso model. This model gives insight concerning the amount of shrinkage based on the distribution form.


####projpred
``` {r}
library(projpred)

#lasso 1
cv_vs <- cv_varsel(fit_L1)
suggest_size(cv_vs)
plot_L1 <- plot(cv_vs, stats = c('rmse'), main = "Lasso 1 - rmse")
plot_L1
summary(cv_vs)
#solution_terms(cv_vs)
# Visualise the projected five most relevant variables
proj <- project(cv_vs, nterms = 9, ns = 500)
mcmc_areas(as.matrix(proj)) +
coord_cartesian(xlim = c(-10, 10))



#lasso 2
cv_vs2 <- cv_varsel(fit_L2)
solution_terms(cv_vs2)
suggest_size(cv_vs2)
plot_L2 <- plot(cv_vs2, stats = c('rmse'))
plot_L2



#lasso 3
cv_vs3 <- cv_varsel(fit_L3)
suggest_size(cv_vs3)
plot_L3 <- plot(cv_vs3, stats = c('rmse'))
plot_L3


```
projpred-package, which implements the projective variable selection (Goutis and Robert, 1998; Dupuis and Robert, 2003) is used for variable selection. We plot some statistics computed on the training data, such as the sum of log predictive densities (ELPD) and root mean squared error (RMSE) as the function of number of variables added. By default, the statistics are shown on absolute scale, but with deltas=T the plot shows results relative to the full model. Furthermore the five most relevant values are listed.

```{r}

library(Metrics)

#smallest model
Lasso_small<- brm(formula = class_ckd ~  hemo, family = "bernoulli"(link='logit'), data = df_train, algorithm= 'sampling', iter= 4000, warmup = 500)

Lasso_small_test <- predict(Lasso_small, newdata = df_test)
rmse(Lasso_small_test, df_test$class_ckd)



#largest model
Lasso_large<- brm(formula = class_ckd ~ age + bp + bgr + bu + sc + sod+ pot + hemo + pcv + wbcc+ rbcc + sg_1.010 + sg_1.015 + sg_1.020 + sg_1.025 + al_1 + al_2 + al_3 + al_4 + al_5 + su_1 + su_2 + su_3 + su_4 + su_5 + rbc_normal + pc_normal + pcc_present + ba_present + htn_yes + dm_yes + cad_yes + appet_good + pe_yes + ane_yes, family = "bernoulli"(link='logit'), data = df_train, algorithm= 'sampling', iter= 5000, warmup = 1000)

Lasso_large_test <- predict(Lasso_large, newdata = df_test)
rmse(Lasso_large_test, df_test$class_ckd)



#LASSO_1
Lasso1<- brm(formula = class_ckd ~  hemo + pcv + rbc_normal + rbcc + dm_yes + htn_yes + sg_1.025 + bgr + sg_1.010, family = "bernoulli"(link='logit'), data = df_train, algorithm= 'sampling', iter= 5000, warmup = 1000)

Lasso1_test <- predict(Lasso1, newdata = df_test)
rmse(Lasso1_test, df_test$class_ckd)



#LASSO_2
Lasso2<- brm(formula = class_ckd ~  hemo + pcv + rbcc + rbc_normal + dm_yes + htn_yes + sg_1.025 + bgr + sg_1.010, family = "bernoulli"(link='logit'), data = df_train, algorithm= 'sampling', iter= 4000, warmup = 500)

Lasso2_test <- predict(Lasso2, newdata = df_test)
rmse(Lasso2_test, df_test$class_ckd)



#LASSO_3
Lasso3<- brm(formula = class_ckd ~  hemo + pcv + rbc_normal + rbcc  + dm_yes + htn_yes + sg_1.025 + bgr + sg_1.010, family = "bernoulli"(link='logit'), data = df_train, algorithm= 'sampling', iter= 4000, warmup = 500)

Lasso3_test <- predict(Lasso3, newdata = df_test)
rmse(Lasso3_test, df_test$class_ckd)


``` 
In the step above, the predicted y-values are calculated with the rmse values based on the selected variables using the plot of projpredict.



```{r}
#LASSO_1
Lasso1.1<- brm(formula = class_ckd ~  hemo + pcv + rbc_normal + rbcc + dm_yes + htn_yes + sg_1.025 + bgr + sg_1.010 + sg_1.020 + sod + pc_normal + appet_good + sg_1.015, family = "bernoulli"(link='logit'), data = df_train, algorithm= 'sampling', iter= 2000, warmup = 1000)

Lasso1.1_test <- predict(Lasso1.1, newdata = df_test)
rmse(Lasso1.1_test, df_test$class_ckd)



#LASSO_2
Lasso2.2<- brm(formula = class_ckd ~  hemo + pcv + rbcc + rbc_normal + dm_yes + htn_yes + sg_1.025 + bgr + sg_1.010 + sg_1.020 + sod + pc_normal + 
sg_1.015 + appet_good + pe_yes + al_4 + su_4 + al_2, family = "bernoulli"(link='logit'), data = df_train, algorithm= 'sampling', iter= 2000, warmup = 1000)

Lasso2.2_test <- predict(Lasso2.2, newdata = df_test)
rmse(Lasso2.2_test, df_test$class_ckd)



#LASSO_3
Lasso3.3<- brm(formula = class_ckd ~  hemo + pcv + rbc_normal + rbcc + dm_yes + htn_yes + sg_1.025 + bgr + sg_1.010 + sg_1.020 + sod + pc_normal + appet_good + sg_1.015, family = "bernoulli"(link='logit'), data = df_train, algorithm= 'sampling', iter= 2000, warmup = 1000)

Lasso3.3_test <- predict(Lasso3.3, newdata = df_test)
rmse(Lasso3.3_test, df_test$class_ckd)


```




### Regularized horseshoe

```{r eval=FALSE, include=FALSE}
library (extraDistr)
library(invgamma)
library(LaplacesDemon)

reg.hs <- rep(NA, ndraws)

for(i in 1:ndraws){
c2 <- rinvgamma(1, shape=0.5, scale=1)
lambda <- rhalfcauchy(1, scale=0.1)  #local parameter
tau <- rhalfcauchy(1, scale=0.1)   #global parameter
lambda2_tilde <- c2 * lambda^2/(c2 + tau^2*lambda^2)
reg.hs[i] <- rnorm(1, 0, sqrt(tau^2*lambda2_tilde))
}

plot(density(reg.hs))
```
In this step we plot the regularized horseshoe distribution based on different parameters. The regularized horseshoe has the most flexibility in terms of tuning. scale_global and global_df determines the general shrinkage for all coefficients simultaneously. The scale influences how wide the peak is and defaults to 1. A smaller scale leads to more overall shrinkage of all coefficients.The global degrees of freedom parameter determines the tail behavior and defaults to 1, with larger values leading to lighter tails.

For the local shrinkage parameters (which allow truly large coefficients to escape the global shrinkage), only the degrees of freedom
(local_df) need to be specified, with 1 as default and larger values resulting in lighter tails. Finally, the regularized horseshoe differs from the horseshoe prior by asserting some shrinkage on large coefficients. This shrinkage is determined by a t-distribution with some scale (slab_scale) and degrees of freedom (slab_df). Both default to 1






```{r}
library("bayesplot")
library("ggplot2")
library("rstanarm")

fit_H1 =  brm(formula = class_ckd ~ age + bp + bgr + bu + sc + sod+ pot + hemo+ pcv + wbcc+ rbcc + sg_1.010 + sg_1.015 + sg_1.020 + sg_1.025 + al_1 + al_2 + al_3 + al_4 + al_5 + su_1 + su_2 + su_3 + su_4 + su_5 + rbc_normal + pc_normal + pcc_present + ba_present + htn_yes + dm_yes + cad_yes + appet_good + pe_yes + ane_yes,family = bernoulli(link = "logit"),
              data=df_train, prior=set_prior(horseshoe(df = 1,  scale_global = 1,  df_global = 1,  scale_slab = 2,  df_slab = 4), class = 'b'),
              algorithm= 'sampling', iter= 4000, warmup = 500)

  save("fit_H1", file = "fit_H1.RData")
  #plot_summs(fit_H1) 
  summary(fit_H1)



fit_H2 =  brm(formula = class_ckd ~ age + bp + bgr + bu + sc + sod+ pot + hemo+ pcv + wbcc+ rbcc + sg_1.010 + sg_1.015 + sg_1.020 + sg_1.025 + al_1 + al_2 + al_3 + al_4 + al_5 + su_1 + su_2 + su_3 + su_4 + su_5 + rbc_normal + pc_normal + pcc_present + ba_present + htn_yes + dm_yes + cad_yes + appet_good + pe_yes + ane_yes,               family = bernoulli(link = "logit"),
              data=df_train, prior=set_prior(horseshoe(df = 1,  scale_global = 0.1,  df_global = 1,  scale_slab = 2,  df_slab = 4), class = 'b'),
              algorithm= 'sampling', iter= 4000, warmup = 500)

  save("fit_H2", file = "fit_H2.RData")
  #plot_summs(fit_H2) 
  summary(fit_H2)


    
fit_H3 =  brm(formula = class_ckd ~ age + bp + bgr + bu + sc + sod+ pot + hemo+ pcv + wbcc+ rbcc + sg_1.010 + sg_1.015 + sg_1.020 + sg_1.025 + al_1 + al_2 + al_3 + al_4 + al_5 + su_1 + su_2 + su_3 + su_4 + su_5 + rbc_normal + pc_normal + pcc_present + ba_present + htn_yes + dm_yes + cad_yes + appet_good + pe_yes + ane_yes,               family = bernoulli(link = "logit"),
              data=df_train, prior=set_prior(horseshoe(df = 1,  scale_global = 1,  df_global = 1,  scale_slab = 0.1,  df_slab = 4), class = 'b'),
              algorithm= 'sampling', iter= 4000, warmup = 500)

  save("fit_H3", file = "fit_H3.RData")
  #plot_summs(fit_H3) 
  summary(fit_H3)

 
```
Three models with regularized horseshoe priors are created with different hyperparameter values.

```{r}

visModel2 <- function(varNum, varName){
  post1 <- as.matrix(fit_H1)
  post2 <- as.matrix(fit_H2)
  post3 <- as.matrix(fit_H3)
  RegHor1 <- post1[, varNum]
  RegHor2 <- post2[, varNum]
  RegHor3<- post3[, varNum]
  x <- data.frame(RegHor1, RegHor2, RegHor3)
  data<- melt(x)
  ggplot(data,aes(x=value, color=variable))+ labs(subtitle=varName) + geom_density(alpha=0.25)
    }


visModel2(27, "rbc_normal")
visModel2(22, "su_1")
```
The function visModel above is a function to visualize the density distribution for specific coefficients per regularized horseshoe model. This model gives insight concerning the amount of shrinkage based on the distribution form. Furthermore, a plot is given wherein the intervals between lasso and regularized horseshoe are compared


```{r}

plot_coefs(fit_L2, fit_L3, fit_H2, model.names = c("Lasso_2", "Lasso_3", "Reghs2"))






```

In this step, two interval plots are visualized to compare the shrinkage of variables between lasso and regularized horseshoe.

```{r}
#RH1
cv_vsH1 <- cv_varsel(fit_H1)
solution_terms(cv_vsH1)
#summary(cv_vsH1)
plot_H1 <-plot(cv_vsH1, stats = c('rmse'))
plot_H1
suggest_size(cv_vsH1)



#RH2
cv_vsH2 <- cv_varsel(fit_H2)
#summary(cv_vsH2)
plot_H2 <-plot(cv_vsH2, stats = c('rmse'))
plot_H2
solution_terms(cv_vsH2)
suggest_size(cv_vsH2)


#RH3
cv_vsH3 <- cv_varsel(fit_H3)
plot_H3 <-plot(cv_vsH3, stats = c('rmse'))
plot_H3
solution_terms(cv_vsH3)
suggest_size(cv_vsH3)




```
Variable selection with projpredict.



```{r}
#RegHor1
reghor1<- brm(formula = class_ckd ~  hemo + pcv + rbc_normal + rbcc + dm_yes + htn_yes + sg_1.025 + bgr + sg_1.010, family = "bernoulli"(link='logit'), data = df_train, algorithm= 'sampling', iter= 2000, warmup = 1000)

reghor1_test <- predict(reghor1, newdata = df_test)
rmse(reghor1_test, df_test$class_ckd) 

#Reghor2 and 3 have the same model thus same output.


#reghor 1.1
reghor1.1<- brm(formula = class_ckd ~  hemo + pcv + rbcc + rbc_normal + dm_yes + htn_yes + sg_1.025 + bgr + sg_1.010 + sg_1.020 + sod + sg_1.015 + pc_normal + appet_good + pe_yes + al_4 + su_4 + al_2 + al_1  , family = "bernoulli"(link='logit'), data = df_train, algorithm= 'sampling', iter= 2000, warmup = 1000)

reghor1.1_test <- predict(reghor1.1, newdata = df_test)
rmse(reghor1.1_test, df_test$class_ckd) 




#reghor 2.2
reghor2.2<- brm(formula = class_ckd ~  hemo + pcv + rbcc + rbc_normal + dm_yes + htn_yes + sg_1.025 + bgr + sg_1.010 + sg_1.020 + sod + sg_1.015 + pc_normal + appet_good + pe_yes + su_4 + al_4, family = "bernoulli"(link='logit'), data = df_train, algorithm= 'sampling', iter= 2000, warmup = 1000)

reghor2.2_test <- predict(reghor2.2, newdata = df_test)
rmse(reghor2.2_test, df_test$class_ckd) 



#Reghor 3.3
reghor3.3<- brm(formula = class_ckd ~  hemo + pcv + rbcc + rbc_normal + dm_yes + htn_yes + sg_1.025 + bgr + sg_1.010 + sg_1.020 + sod + sg_1.015 + pc_normal + appet_good + pe_yes + su_4 + al_4 + al_2, family = "bernoulli"(link='logit'), data = df_train, algorithm= 'sampling', iter= 2000, warmup = 1000)

reghor3.3_test <- predict(reghor3.3, newdata = df_test)
rmse(reghor3.3_test, df_test$class_ckd) 


```
