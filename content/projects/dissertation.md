---
title: "Bayesian Analysis of Efficacy of the ChAd0x1 nCoV-19 (AZD1222) Vaccine"
date: 2022-09-06T22:00:32+05:30
draft: true
author: "Roudranil"
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: false
UseHugoToc: true
hideSummary: true
---

## Synopsis

2020 saw the onset of the onset of the Covid-19 pandemic cause by the SARS-CoV-2 virus. With the number of daily affected people and number of deaths due to the pandemic climbing sharply, a vaccine was the need of the hour. In this paper we consider one such vaccine, the ChAdOx1 nCoV-19 (AZD1222) vaccine (known as Covishield in India), and we investigate its efficacy based on studies of four randomised controlled trials held in Brazil, South Africa, and the United Kingdom. We use the Bayesian paradigm to model the posterior distribution of the vaccine efficacy and calculate its credible interval.

_Methods_: In order to perform Bayesian inference, we impose prior distributions on the incidence rate of infection in the 
vaccine and the control cohorts. In another model we impose prior distributions of the parameters of the infection process in both cohorts. The likelihood of observed data is then chosen suitably, and the data is collected from “Safety and efficacy of the ChAdOx1 nCoV-19 vaccine (AZD1222) against SARS-CoV-2: an interim analysis of four randomised controlled trials in Brazil,   South Africa, and the UK”, by Voysey et al. Vaccine efficacy is then modelled as a function of these parameters (i.e., either the incidence rates or parameters of infection processes). Using Bayes’ Theorem, we then compute the posterior densities of the parameters and vaccine efficacy consequently.

_Results_: Using the posterior density of vaccine efficacy we can compute the predicted vaccine efficacy, and also the 
corresponding 95% Bayesian Credible Intervals. The values obtained by Bayesian methods are in excellent agreement with those obtained in the paper.

*Concluding remarks*: From the results of our models, we can conclude that a Bayesian analysis of vaccine clinical trials 
data is also a viable method to compute the efficacy.

## Link to the paper

The full text of the project paper is [available here](/pdf/dissertation.pdf).

The code associated with the paper (as well as the paper itself) can be found in this [repo](https://github.com/Roudranil/Bayesian-analysis-of-efficacy-of-the-ChAdOx1-nCoV-19-AZD1222-vaccine).
