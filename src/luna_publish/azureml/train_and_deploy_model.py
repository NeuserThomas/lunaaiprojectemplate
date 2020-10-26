from azureml.core import model
from azureml.core.webservice.aci import AciWebservice
from azureml.core.webservice.aks import AksWebservice
from luna.utils import ProjectUtils
from azureml.core import Workspace, Experiment, Run
from azureml.core.webservice import Webservice
from uuid import uuid4
import json
import os
import pathlib
import argparse

if __name__ == "__main__":

    parser=argparse.ArgumentParser(description="Train and publish model") 

    parser.add_argument('-experiment_name', 
                        '--experiment_name', 
                        help="The name of the experiment", 
                        default="train_and_deploy_model",
                        type=str)  
                        
    parser.add_argument('-model_id', 
                        '--model_id', 
                        help="The id of the model", 
                        default="default",
                        type=str)  
                        
    parser.add_argument('-endpoint_id', 
                        '--endpoint_id', 
                        help="The id of the service endpoint", 
                        default="default",
                        type=str)  

    parser.add_argument('-dns_name_label', 
                        '--dns_name_label', 
                        help="The name of DNS name label", 
                        default="default",
                        type=str)  

    parser.add_argument('-input_data_file_path', 
                        '--input_data_file_path', 
                        help="The data file for training input", 
                        default="default",
                        type=str) 

    args=parser.parse_args()

    experimentName = args.experiment_name
    modelId = args.model_id
    endpointId = args.endpoint_id
    serviceEndpointDnsNameLabel = args.dns_name_label
    input_data_file = args.input_data_file_path

    if modelId == "default":
        modelId = str('a' + uuid4().hex[1:])
    
    if endpointId == "default":
        endpointId = str('a' + uuid4().hex[1:])

    if serviceEndpointDnsNameLabel == "default":
        serviceEndpointDnsNameLabel = str('a' + uuid4().hex[1:])
    
    if input_data_file == "default":
        input_data_file = os.path.join(pathlib.Path(__file__).parent.absolute(), "training_input.json")

    with open(input_data_file) as f:
        trainingUserInput = f.read()
    trainingUserInput = trainingUserInput.replace(" ", "").replace("\n", "").replace("\r", "")
    print(trainingUserInput)
    
    deploymentUserInput = json.dumps({"dns_name_label": serviceEndpointDnsNameLabel})
        
    utils = ProjectUtils(luna_config_file='luna_config.yml', run_mode='default')

    run_id = utils.RunProject(entry_point = 'train', 
                                    experiment_name = experimentName, 
                                    parameters={'operationId': modelId, 
                                                'userInput': trainingUserInput}, 
                                    tags={})
    
    utils.WaitForRunCompletion(run_id, experimentName)
    
    run_id = utils.RunProject(entry_point = 'deploy', 
                                    experiment_name = experimentName, 
                                    parameters={'predecessorOperationId': modelId, 
                                                'userInput': deploymentUserInput, 
                                                'operationId': endpointId}, 
                                    tags={})
    
    utils.WaitForRunCompletion(run_id, experimentName)
    
    webservice = utils.GetServiceEndpoint(endpointId)
    
    if (webservice.compute_type == 'ACI'):
        aciWebservice = utils.GetAciServiceEndpoint(endpointId)
        print("The model {} was deployed to a ACI service endpoint.".format(modelId))
        print("The scoring URL is: {}".format(aciWebservice.scoring_uri))
        print("The primary authentication key is: {}".format(aciWebservice.get_keys()[0]))
    elif (webservice.compute_type == 'AKS'):
        aksWebservice = utils.GetAksServiceEndpoint(endpointId)
        print("The model {} was deployed to a AKS service endpoint.".format(modelId))
        print("The scoring URL is: {}".format(aksWebservice.scoring_uri))
        print("The primary authentication key is: {}".format(aksWebservice.get_keys()[0]))