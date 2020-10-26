from luna.lunaUtils import LunaUtils
import os

from LunaPythonModel import LunaPythonModel

utils = LunaUtils.Create()
args = utils.args
user_input = utils.user_input

model_path = utils.DownloadModel()
print(model_path)

python_model = LunaPythonModel()

python_model.batch_inference(args, user_input, model_path, utils.logger)
