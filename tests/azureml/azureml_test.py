from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.core import Workspace
from azureml.pipeline.core.graph import PipelineParameter
from azureml.core import Experiment
from azureml.core.webservice import AciWebservice, AksWebservice, Webservice

from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from uuid import uuid4

from luna.utils import ProjectUtils
import os
from uuid import uuid4
import requests
import json
import pathlib
import argparse
import time

modelId = '00000000-0000-0000-0000-000000000000'
endpointId = '00000000-0000-0000-0000-000000000000'
operationId = '00000000-0000-0000-0000-000000000000'
subscriptionId = '00000000-0000-0000-0000-000000000000'

experimentName = 'xiwumlflowTest'
userId = 'xiwu@microsoft.com'
productName = 'eddi'
deploymentName = 'westus'
apiVersion = 'v1.0'
ws = None
dns_name_label = 'testlabel'
utils = None

def init(test_data_file):
    global modelId, endpointId, operationId, subscriptionId, dns_name_label, test_data, utils, ws
    utils = ProjectUtils(luna_config_file='luna_config.yml', run_mode='default')
    ws = Workspace.from_config(path=utils.luna_config['azureml']['workspace_config_path'], _file_name=utils.luna_config['azureml']['workspace_config_file_name'])
    modelId = str('a' + uuid4().hex[1:])
    endpointId = str('a' + uuid4().hex[1:])
    operationId = str('a' + uuid4().hex[1:])
    subscriptionId = str('a' + uuid4().hex)[1:]
    dns_name_label = str('a' + uuid4().hex[1:])
    print('modelId {}'.format(modelId))
    print('endpointId {}'.format(endpointId))
    print('operationId {}'.format(operationId))
    print('subscriptionId {}'.format(subscriptionId))
    print('dns_name_label {}'.format(dns_name_label))
    if test_data_file == "default":
        test_data_file = os.path.join(pathlib.Path(__file__).parent.absolute(), "test_data.json")
    with open(test_data_file) as f:
        test_data = json.load(f)

def trainModel(userInput='{}'):
    run_id = utils.RunProject(azureml_workspace = ws, 
                                    entry_point = 'train', 
                                    experiment_name = experimentName, 
                                    parameters={'userId': userId,
                                                'userInput': userInput, 
                                                'operationId': modelId,
                                                'productName': productName,
                                                'deploymentName': deploymentName,
                                                'apiVersion': apiVersion,
                                                'subscriptionId': subscriptionId,
                                                'predecessorOperationId': 'na'}, 
                                    tags={'userId': userId, 
                                            'productName': productName, 
                                            'deploymentName': deploymentName, 
                                            'apiVersion': apiVersion,
                                            'modelId': modelId,
                                            'subscriptionId': subscriptionId})

    print(run_id)
    return run_id

def batchInference(userInput='{}'):
    run_id = utils.RunProject(entry_point = 'batchinference', 
                                    experiment_name = experimentName, 
                                    parameters={'userId': userId,
                                                'userInput': userInput, 
                                                'operationId': operationId,
                                                'productName': productName,
                                                'deploymentName': deploymentName,
                                                'apiVersion': apiVersion,
                                                'subscriptionId': subscriptionId,
                                                'predecessorOperationId': modelId}, 
                                    tags={'userId': userId, 
                                            'productName': productName, 
                                            'deploymentName': deploymentName, 
                                            'apiVersion': apiVersion,
                                            'modelId': modelId,
                                            'subscriptionId': subscriptionId,
                                            'operationId': operationId})
    print(run_id)
    return run_id

def deploy(userInput='{"dns_name_label":"testlabel"}'):
    run_id = utils.RunProject(entry_point = 'deploy', 
                                    experiment_name = experimentName, 
                                    parameters={'userId': userId,
                                                'userInput': userInput, 
                                                'operationId': endpointId,
                                                'productName': productName,
                                                'deploymentName': deploymentName,
                                                'apiVersion': apiVersion,
                                                'subscriptionId': subscriptionId,
                                                'predecessorOperationId': modelId}, 
                                    tags={'userId': userId, 
                                            'productName': productName, 
                                            'deploymentName': deploymentName, 
                                            'apiVersion': apiVersion,
                                            'modelId': modelId,
                                            'subscriptionId': subscriptionId,
                                            'endpointId': endpointId})
    print(run_id)
    return run_id

def test_deployed_endpoint(data, expected_output):
    
    tags = [['userId', userId], ['endpointId', endpointId], ['subscriptionId', subscriptionId]]
    endpoints = Webservice.list(ws, tags = tags)
    endpoint = endpoints[0]
    headers = {'Content-Type': 'application/json'}
    headers['Authorization'] = 'Bearer '+endpoint.get_keys()[0]

    test_sample = json.dumps(data)

    response = requests.post(
    endpoint.scoring_uri, data=test_sample, headers=headers)
    if response.status_code != 200:
        raise Exception('The service return non-success status code: {}'.format(response.status_code))
    if response.json() != expected_output:
        raise Exception('The scoring result is incorrect: {}'.format(response.json()))

if __name__ == "__main__":
    
    parser=argparse.ArgumentParser(description="Test AML pipelines") 

    parser.add_argument('-test_data_file_path', 
                        '--test_data_file_path', 
                        help="The file path of test data", 
                        default="default",
                        type=str)  

    args=parser.parse_args()

    init(args.test_data_file_path)
    
    run_id = trainModel(userInput=json.dumps(test_data['training_user_input']))

    utils.WaitForRunCompletionByTags(experimentName, tags={'modelId': modelId, 'userId': userId, 'subscriptionId': subscriptionId})
    run_id = batchInference(userInput=json.dumps(test_data['batch_inference_input']))
    utils.WaitForRunCompletionByTags(experimentName, tags={'operationId': operationId, 'userId': userId, 'subscriptionId': subscriptionId})

    userInput = '{{"dns_name_label":"{}"}}'.format(dns_name_label)
    run_id = deploy(userInput=userInput)
    utils.WaitForRunCompletionByTags(experimentName, tags={'endpointId': endpointId, 'userId': userId, 'subscriptionId': subscriptionId})

    test_deployed_endpoint(data=test_data['real_time_scoring_input'], expected_output=test_data['real_time_scoring_expected_output'])