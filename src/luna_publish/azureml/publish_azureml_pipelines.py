from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.core.runconfig import RunConfiguration
from azureml.core import Workspace

from luna.utils import ProjectUtils

import argparse

import os


if __name__ == "__main__":

    parser=argparse.ArgumentParser(description="Publish AML pipelines") 

    parser.add_argument('-training_pipeline_name', 
                        '--training_pipeline_name', 
                        help="The name of training pipeline", 
                        default="mytrainingpipeline",
                        type=str)  
                        
    parser.add_argument('-batch_inference_pipeline_name', 
                        '--batch_inference_pipeline_name', 
                        help="The name of batch inference pipeline", 
                        default="mybatchinferencepipeline",
                        type=str) 
                        
    parser.add_argument('-deployment_pipeline_name', 
                        '--deployment_pipeline_name', 
                        help="The name of deployment pipeline", 
                        default="mydeploymentpipeline",
                        type=str)
                        

    args=parser.parse_args()
    
    training_pipeline_name = args.training_pipeline_name
    batch_inference_pipeline_name = args.batch_inference_pipeline_name
    deployment_pipeline_name = args.deployment_pipeline_name

    utils = ProjectUtils(luna_config_file='luna_config.yml', run_mode='default')

    print('publishing training pipeline')
    endpoint = utils.PublishAMLPipeline('train', training_pipeline_name, 'The training pipeline')
    print('training pipeline published with endpoint {}'.format(endpoint))
    
    print('publishing batchinference pipeline')
    endpoint = utils.PublishAMLPipeline('batchinference', batch_inference_pipeline_name, 'The batch inference pipeline')
    print('batchinference pipeline published with endpoint {}'.format(endpoint))
    
    print('publishing deployment pipeline')
    endpoint = utils.PublishAMLPipeline('deploy', deployment_pipeline_name, 'The deployment pipeline')
    print('deployment pipeline published with endpoint {}'.format(endpoint))
