from luna.lunaUtils import LunaUtils
import os
from LunaPythonModel import LunaPythonModel
import json
from datetime import datetime

utils = LunaUtils.Create()
args = utils.args
user_input = utils.user_input

python_model = LunaPythonModel()

model_path, description = python_model.train(args, user_input, utils.logger)


utils.RegisterModel(model_path = model_path,
                       description = description,
                       luna_python_model=LunaPythonModel())