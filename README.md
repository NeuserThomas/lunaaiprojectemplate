# Demo: Publish a regression model service

## Overview
In this article, we will show you how to use Luna service to publish an AI service to train regression model and use it to do batch inference or real time scoring.

## Deploy Luna Service
You can deploy Luna service by following the instruction [here](aka.ms/projectluna). If you want to use our pre-deployed demo environment, please contact xiwu@microsoft.com

## Clone Luna project template
Clone the template from [here](https://aka.ms/lunaprojecttemplate)

## Update AML environment
### Update AML workspace info
Update the value of "Scope" property in ./.cloud/.azureml/test_workspace.json with your AML workspace Azure resource Id

### Update AML compte targets
Update the value of "target" property in ./.cloud/.azureml/aml_run_config.yml with the compte cluster name. The compute cluster should be already created and registered in your AML workspace.

### Update your deployment targets
Update the value of "deployment_target" in ./.cloud/.azureml/compute.yml. Right now we support aci (Azure Container Instance) or aks (Azure Kubernetes Service). If you choose AKS as deployment target, you also need to update the value of "aks_cluster" to the name of AKS cluster you created and registerd in AML.

