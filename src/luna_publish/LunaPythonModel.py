from mlflow.pyfunc import PythonModel, PythonModelContext
from luna.lunaUtils import LunaUtils
from azureml.contrib.services.aml_response import AMLResponse
import json


class LunaPythonModel(PythonModel):
    def load_context(self, context):
        ## DO NOT CHANGE! Set mlflow as default run mode
        if (self._run_mode != 'azureml'):
            self._run_mode = 'mlflow'

        ## DO NOT CHANGE! Get the model path
        model_path = LunaUtils.GetModelPath(run_mode = self._run_mode, context = context)
        
        # This function will only be called if user deploy the model to a real time service endpoint
        # This function will be called every time when the container instance is started
        # DO NOT use context otherwise it won't work on Azure ML
        # You model is the following directory: model_path/<model_path you specified in train method>

        return

    def predict(self, context, model_input):
        ## DO NOT CHANGE! Get the model path
        model_path = LunaUtils.GetModelPath(run_mode = self._run_mode, context = context)
        
        # Add your scoring code here. You model is the following directory: model_path/<model_path you specified in train method>
        # DO NOT use context otherwise it won't work on Azure ML
        # Update the scoring_result with the real result
        
        user_input = json.loads(model_input)
        
        scoring_result = json.dumps({"result": ""})
        return AMLResponse(scoring_result, 200)

    def train(self, args, user_input, logger):  
        # train your model here
        # userInput is a dictionary, for example userInput['source']

        # Update the model_path if your model is saved in a different folder. 
        # All files under model_path will be saved and registered as a part of the model
        # Update the description for your model

        # Logging example:
        # 1. Log metric: logger.log_metric("accuracy, 0.89)
        # 2. Log metrics: logger.log_metrics({"accuracy": 0.89, "execution_time_in_sec": 100})
        # 3. Upload a file or artifact: logger.upload_artifacts(local_file_name, upload_file_name)
        # 4. Upload files or artifacts: logger.upload_artifacts(local_directory_name, upload_directory_name)

        model_path = "models"
        description = "this is my model"

        return model_path, description

    def batch_inference(self, args, user_input, model_path, logger):
        # Do your batch inference here. You model is the following directory: model_path/<model_path you specified in train method>
        # userInput is a dictionary, for example userInput['source'] or userInput['hyper_parameters]['epocs']
        # The return value will be ignored. You should ask user to provide a output data source as user input and write the result

        # Logging example:
        # 1. Log metric: logger.log_metric("accuracy, 0.89)
        # 2. Log metrics: logger.log_metrics({"accuracy": 0.89, "execution_time_in_sec": 100})
        # 3. Upload a file or artifact: logger.upload_artifacts(local_file_name, upload_file_name)
        # 4. Upload files or artifacts: logger.upload_artifacts(local_directory_name, upload_directory_name)
        

        return

    ## DO NOT CHANGE
    def set_run_mode(self, run_mode):
        self._run_mode = run_mode
