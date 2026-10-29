---
title: "What I learned doing Data Science Interviews in the 2020s"
date: "2023-12-18"
description: "Notes on preparation, technical questions, and the interview process from both sides of the table."
category: "Learning"
tags: ["interview-preparation", "data-science", "cv", "interview-questions"]
original_url: "https://medium.com/@lukas.hauzenberger/what-i-learned-doing-data-science-interviews-in-the-2020s-219c9517cffa"
---

![](<images/7dec99dc8b377fd1.jpeg>)



Few disciplines have experienced hype periods comparable to that of Data Science (DS) and Machine Learning (ML) since approximately 2016. The demand for new data scientists has soared paralleled by the continuous stream of new talent and people looking for a career switch entering into the field. Due to the growing importance of DS for businesses and the higher volume of applicants for DS positions, interview processes have gotten more refined and challenging.



Nevertheless, I believe that effective preparation can provide a substantial competitive advantage for many open positions, despite maybe having only a basic understanding of Statistics and ML. In this article, I will share key insights I gained from participating in a significant double-digit number of DS interviews over the past few years — both as an applicant and as part of the hiring team.



## First tip: pimp your CV!



This might be obvious but having a good CV that makes you stand out vs hundreds of other applicants is essential. Without being invited for an interview you cannot apply any of the other tips I will give you. A good CV combines effective formatting/design with compelling content.



![](<images/6913c1c4544d6d04.png>)

*LaTeX template “Awesome CV”*



For formatting I highly recommend this [LaTeX template](<https://www.overleaf.com/latex/templates/awesome-cv/dfnvtnhzhhbm>) which can be used for both a CV as well as a cover letter (if you want to bother with that). In case you dont know LaTeX I highly encourage you to learn it as its incredibly useful to write any kind of longer technical documents. For anyone with some coding knowledge, which I would assume you have, its pretty easy to pick up.



For optimizing the CV content I wont go too much in detail since there is already a lot of specialized guides available but the three tips I would give you are the following:



- **Tell a consistent story.** The different roles you list in your CV should make sense in the sequence you present them. If you list a non-senior position after a senior position you will likely raise question marks. As job titles are anyway not really comparable across companies (everyone has their own naming schema) I would suggest to tweak the labels you give to the roles on your CV to tell a consistent story of career growth.
- **Emphasize proficiency** in programming languages, data manipulation, and machine learning frameworks. Provide concrete examples of impactful projects to demonstrate the practical application of those skills. Also quantify achievements wherever possible, using metrics to highlight the impact of your contributions.
- **Tailor your CV to the specific job** by aligning your skills and experiences with the requirements of the role. I cannot state the advantage of this enough. Recruiters often hire for a position in a specific team therefore they often know which technologies, frameworks and algorithms you will be working with and will skim CVs for keywords related to that. Its usually easy to extract information on that from the job posting.



## The case study



A case study is a popular method to weed out impostors that might be able to present themselves well but do not have the skills to back it up. In my experience it often comes after a first screening call where your technical knowledge is never tested and before the second round interview where usually a SME is present who can challenge your DS knowledge.



Typically a case study involves a dataset with company pseudo data and some DS problem statement. The goal of such assignments is almost never to achieve the best model performance and somehow amaze with an extremely low error metric but to get some insight in your way of working and if you follow generally accepted best practices for the development process. In my opinion there are some key points you should consider when working on such a case study:



- **Clarify open questions early:** We all like to procrastinate so it does happen quite often that an applicant starts working on on the evening of the deadline. It is fine if you do this but make sure you at least quickly read through the case study as soon as you get it to have enough time to clarify any important points with your company contact person.
- **Make your assumptions transparent:** If your approach assumes anything that is not clearly stated in the case description you should list those assumptions when you present your results.
- **Exploratory data analysis is important:** I have experienced multiple times that the dataset you get contains some traps such as features that are just random noise, extreme outliers or highly imbalanced data. Things like this are easily detected by just generating some basic plots about the data distribution. Doing this requires little effort but skipping this step can quickly disqualify you in the eyes of the interviewer.
- **Train / Validation / Test split:** Everyone who has done some basic crash course in data science knows about doing a train-test split but many often forget that when running different models and/or trying different values for hyperparameters you run into danger of overfitting on the test data. Therefore it is important to keep a third subset of the data that you do the final evaluation on once you have selected your best model and hyperparameters. This is generally referred to as the test set while the dataset you perform your intermediate evaluations on is the validation set. If you do not follow this pattern there is a high chance it will be addressed during the interview.
- **Create a baseline model:** many make the mistake of directly jumping to xgboost or a neural network for something were linear / logistic regression might have provided similar or even better performance. For timeseries problems make sure you check the performance of just predicting the last value. I once had a candiate explain to me he selected an LSTM for a time series case study because its a simple model. He was then unable to explain to me how an LSTM works. This leads me to my next point.
- **Know the algorithm you are using!** Chances are you will be asked to explain how it works and why you chose it. If you are aware of this before the interview it is relatively easy to prepare good answers. Therefore be aware of the advantages and disadvantages of the algorithm and know about the most important hyperparameters.
- **Feature Importance:** It is not essential but generally well received if you somehow visualise the importance of different features for you model. Some model implementation have built in methods that give out feature importances so it is very little effort to apply this. An alternative would be to use [Shapely values](<https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html>).



## Technical Interview



![](<images/8c205a5c699bca8f.png>)



In addition to being questioned about the case study, for which I hope you are well-prepared using the tips provided in the preceding section, I’d like to highlight some popular topics that often serve as sources for data science interview questions.



- [**Bias Variance Tradeoff**](<https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff>)**:** Know about the dangers of overfitting and that it visually equates to the training error decreasing while the validation error increases. Also have in mind that overfitting equates to high variance while underfitting equates to high bias which is usually the case with simple models (e.g. the mean which is essentially a very simple statistical model). Ideally also know how the mean squared error can be represented with bias and variance.
- [**Linear Regression**](<https://en.wikipedia.org/wiki/Linear_regression>) **and** [**OLS**](<https://en.wikipedia.org/wiki/Ordinary_least_squares>)**:** Linear regression is one of the simplest models and it is a popular topic for interviews to see if you know the basics. Therfore you should be able to explain the model in detail. You should also know OLS, in particular its [assumptions](<https://www.albert.io/blog/key-assumptions-of-ols-econometrics-review/>).
- [**Maximum Likelihood Estimation:**](<https://en.wikipedia.org/wiki/Maximum_likelihood_estimation>) It involves maximizing the likelihood function for a specific statistical model (e.g. Gaussian Distribution). The goal is to optimize the parameters of the statistical model so that they make the observed data most probable. The likelihood function quantifies the probability of observing the given data given the parameter values. Also be aware that for linear models maximum likelihood estimation is equivalent to OLS.
- [**Bayes Theorem**](<https://en.wikipedia.org/wiki/Bayes%27_theorem>)**:** Be able to explain what bayes theorem says and what its components are. For more challenging interviews be prepared to solve a simple mathematical example such as [this](<https://www.mathsisfun.com/data/bayes-theorem.html>) one using bayes theorem.
- [**Confusion Matrix**](<https://en.wikipedia.org/wiki/Confusion_matrix>) **and** [**F1 Score**](<https://en.wikipedia.org/wiki/F-score>)**:** A confusion matrix helps visulalise the performance of a classification algorithm. You should know what precision and recall are and why the F1 score is suitable for imablanced classification but accuracy is not.
- **Imbalanced classification**: How to deal with imbalanced classes (sampling techniques such as boostrapping, random duplication, SMOTE, stratified train-test split or weighted loss functions). Have an idea about which algorithms perform well out of the box on imbalanced data (GBM) and suitable evaluation metrics (F1 score).
- **Unsupervised learning**: Know about examples of unsupervised learning (Clustering, Anomaly Detection, Dimensionality Reduction). Know about popular algorithms (K-Means, DB-SCAN, Expectation Maximisation for clustering; Isolation Forest, One-Class SVM for anomaly detection; t-SNE, UMAP for dimensionality reduction). For clustering you should also know about the [elbow method](<https://en.wikipedia.org/wiki/Elbow_method_(clustering)>).
- [**Imputation:**](<https://en.wikipedia.org/wiki/Imputation_(statistics)>) Be able to explain what imputation is and in what cases it makes sense. Name at least one imputation technique: Mean imputation, Hot/Cold deck imputation, Regression Imputation
- **Data Drift:** Know what data drift is and how you can measure it (KL divergence, Wasserstein distance)
- **Neural Networks:** Be able to explain how a neural network works and know about widely used neural network components (though I feel like this is not asked too frequently in detail)  
  **Optimizers:** SGD, SGD with Momentum, Adam  
  **Regularization Techniques:** L1, L2, Dropout  
  **Activation Functions:** Tanh, ReLU  
  **Special Layers:** Embeddings, attention, residual / skip connections  
  **The Vanishing Gradient Problem** and how to fix it (e.g. with Residual Layers, Batch/Layer Norm, ReLU)
- **Know your interviewer:** Before an interview you will usually get a calendar invite or an email where the names of the people that will participate in the meeting are already listed. You can use this to check online (LinkedIn, personal website etc.) what projects/topics they have already worked on in the past and what their interests are. This can be invaluable information when preparing for an interview because based on the job description and the interviews interests you can often get a good idea what likely questions could be and which topics you should refresh your knowledge on.  
  Also be aware that if the interviewers interests align with something that you mentioned in your CV or cover letter it is very likely they will ask you about it.



## Final thoughts



In this article I tried to give an overview of the most important components of a DS interview. From my experience if you know the topics that are listed here well and follow my suggestions you should be very well prepared for 95 percent of DS interview processes. Of course for very specialised roles or research related positions the process and questions might look very different. I would be very interested to hear your thoughts on this and if there is topics / tips I missed which from your experience should be included in this guide.



If you would like a additional guidance for mastering DS interviews or have more specific questions you are also welcome to send me a message at [lukas.hauzenberger.medium@gmail.com](<mailto:lukas.hauzenberger.medium@gmail.com>)
