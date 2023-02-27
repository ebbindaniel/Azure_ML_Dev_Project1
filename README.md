# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.


## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**

The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. The marketing data consists of various attributes like age,job, education, etc to access if a bank term deposit for a client would be subscribed or not. 

Here we try to train our data on the attributes and we seek to Classify if a bank term deposit for a client would be subscribed or not. 


**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

The Best performing Model was Voting Ensamble Model with a 91.64% accuracy acquired during AutoML run.

| AutoML Best Run Model | |
| :---: | :---: |
| id | AutoML_c36df95d-a166-416b-9a5b-96d555378323_30 |
| Accuracy | 0.9164491654021244 |
| AUC_weighted | 0.9472247723046079 |
| Algortithm | VotingEnsemble |


## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

In this project, I had to build and optimize an Azure ML pipeline using two Classification experiments using the Python SDK and find the best experiment based on metrics from each run. The two Classification approaches/ experiments used to evaluate the best model were:

1. Manual Hyperdrive ML Parameter optimization and model selection
2. Auto ML based Parameter optimization and model selection


Below is the main illustration of the general Pipeline architecture: 

![pipeline](img/pipeline.JPG?raw=true "Pipeline")


**data**

Here the data used is the bank marketing campaign data retrieved as a Tabular dataset and then cleaned and split into training and testing as per the Train.py script. 

**Manual Hyperdrive Pipeline Steps**

Here we perform the following pipeline steps with the details:

_Step1: Azure ML Workspace connection_ 

Connect to the Azure ML workspace using config.json

_Step2: Azure ML Experiment & Compute_

Create a new Experiment & Compute resource

_Step3: Scikit Learn Model_

Here we specify a Logistic regression model in 'Train.py' where we specify the arguments 'C' and 'Max_Iter' using argparse and get the accuracy details logged. 

_Step4: Parameter Sampler_

Here we randomly sample hyperparameter values of 'C' where smaller values specify stronger regularization and 'Max_Iter' to provide the iteration values to sample.

_Step5: Early Termination_

This is used to automatically terminate poorly performing runs thus improving computational efficiency.

_Step6: Set up Conda environment with dependencies(conda_dependencies.yml)_

Initiates base packages dependent for running.

_Step7: Get the details from the Training Job (ScriptRunConfig)_

Here we pass the initial Train.py as our script to iterate and run based on our hyperparameters

_Step8: hyperdrive_run_config_

The hyperdrive handles  parameter sampling and reuses the ScriptRunConfig to run different experiments on (Train.py) with different parameter values passed as arguments 'C' and 'Max_iter'. Aside from this, hyperdrive also keeps track of metrics from the Logs from each experiment according to our Primary Metric goal which in our case is 'Accuracy'.

_Step9: Save & Submit the Best Hyperdrive run and view the results_

Once the Hyperdrive run is completed we can view the results and save the best model by primary metric we defined earlier. 


**Hyperparameter Configuration and Classification Algorithim**

Here the Classification algorithim used was Logistic Regression. 

The Hyperparameters used are:

```
ps = RandomParameterSampling(
    {
        '--C' : choice(0.001,0.01,0.1,1,10,20,50,100,200,500,1000),
        '--max_iter': choice(50,100,300)
    }
)
```
_C_ is the Regularization while _max_iter_ is the maximum number of iterations.




**What are the benefits of the parameter sampler you chose?**


C would train and test based on the regularization strengths and with respect to various number of iterations providing a model with better chance to evalute and produce a higher accuracy model.  

_RandomParameterSampling_ was chosen as it is faster and supports early termonation which can save computational costs. 


**What are the benefits of the early stopping policy you chose?**

An early stopping policy is used to automatically terminate poorly performing runs thus improving computational efficiency. 

The policy i chose was the BanditPolicy as follows:

```
policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)
```

evaluation_interval: This is optional and represents the frequency for applying the policy. Each time the training script logs the primary metric counts as one interval.

slack_factor: The amount of slack allowed with respect to the best performing training run. This factor specifies the slack as a ratio.

Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated. This means that with this policy, the best performing runs will execute until they finish.




## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

Here i used Azure AutoML package which can be used to run multiple classification Models and retrieve the best Model based on the primary metric defined instead of manually tuning the hyperparamets or manually performing model selection to find the best model and its associated metrics. 


**AutoML Pipeline Steps** 

_Step1: Azure ML Workspace connection_ 

Connect to the Azure ML workspace using config.json

_Step2: Azure ML Experiment & Compute_

Create a new Experiment & Compute resource

_Step3: Data setup_

The Data from the Bank Marketing dataset is obtained and converted to a tabular dataset. It is then cleaned and registered as a Azure Dataset which can be later  retrieved. 

_Step4: Auto ML  Config_

Here we configure the Auto ML parameters that we would need to pass to run the AutoML Experiment and their primary metric. 

_Step5: Save & Submit the Best AutoML run and view the results_

Once the AutoML run is completed we can view the results and save the best model by primary metric we defined earlier. 


**AutoML Configurations**

The following were the configuration for the AutoML run:

```
automl_config = AutoMLConfig(
    compute_target = compute_target,
    experiment_timeout_minutes=30,
    task='classification',
    primary_metric= 'accuracy',
    training_data=Main_dataset,
    label_column_name='Label_col', 
    n_cross_validations= 2)
```
_experiment timeout minutes=30_

The default experiment was set to 30 mins. 

_task='classification'_

This defines the experiment type which in this case is classification.

_primary metric='accuracy'_

accuracy was chosen as the primary metric.

_training data and label column name_ 

Here the Main dataset was was provided as training data and the label column was the Predictor column  - bank term deposit for a client subscribed or not (0 or 1)


_n_cross_validations=2_

This parameter sets how many cross validations to perform, based on the same number of folds (number of subsets). In this case i choose to have 2 Cross Validation subsets to reduce any overfitting as the metrics would be the average of 2 subset outputs generated. 


## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

| HyperDrive Best Run Model | |
| :---: | :---: |
| id | HD_887a2b34-8ac1-4850-aec3-53c04a5ca594_5 |
| ML Model | Logistic Regression | 
| Regularization Strength | 200.0 |
| Max iterations | 300 |
| Accuracy | 0.9141122913505311 |

![best_run_hyp](img/best_run_hyp.jpg?raw=true "Hyperdrive Best Run")

![Hyperdrive Accuracy PLot](img/acc_hyp.JPG?raw=true "Hyperdrive Accuracy PLot")


| AutoML Best Run Model | |
| :---: | :---: |
| id | AutoML_c36df95d-a166-416b-9a5b-96d555378323_30 |
| Accuracy | 0.9164491654021244 |
| AUC_weighted | 0.9472247723046079 |
| Algortithm | VotingEnsemble |

![Auto ML Accuracy](img/auto_ml.JPG?raw=true "Auto ML Accuracy")

Here we see based on the Hyperparamters we provided the HyperDrive was able to get a model with 91.41% accuracy with around 18mins to execute However the AutoML took around 27 mins to complete with a little better accuracy of 91.64% using a Voting Ensamble Model. 

Hyperparameter only used the Logistic regression in the Train.py and iterated it based on the arguments we provided to find the best model. However Auto ML ran the data set with all the classification models listed under the package to provide the best model with the best metric. 

AutoML was computationally more exhaustive but it ran a lot of models with auto hyperpatrameter tuning to find the best model. In this example Voting Ensamble provided the best model. This would have been an exhaustive research and run manually but was made easy by the AutoML package. 



## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**


During the AutoML run, it detected an imbalance class counts

![Imbalance Class info](img/aml_improvements.JPG?raw=true "Imbalance Class info")

![Imbalance Class](img/aml_improvements2.JPG?raw=true "Imbalance Class")

Here we see there are only 3.6k 1s as compared to the overall dataset of 32k. This could be accounted for in the next iteration to improve the accuracy. 

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**

The Cluster was deleted in the code.

![Deleted Cluster](img/del.JPG?raw=true "Deleted Cluster")


## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)