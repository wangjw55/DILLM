import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

from transformers import AutoTokenizer, AutoModel
from utils_chatglm import load_model_on_gpus
tokenizer = AutoTokenizer.from_pretrained("/home/wsco/DILLM/r2r_src/THUDM/chatglm3-6b", trust_remote_code=True)
# model = AutoModel.from_pretrained("/home/wsco/DILLM/r2r_src/THUDM/chatglm3-6b", trust_remote_code=True).quantize(4).half().cuda()
model = load_model_on_gpus("/home/wsco/DILLM/r2r_src/THUDM/chatglm3-6b", num_gpus=2)
model = model.eval()
import json
from tqdm import tqdm

# file_name = "R2R_train" # R2R_train, R2R_val_seen, R2R_val_unseen
# test:1391 train:4675 val_seen:340 val_unseen:782
prompt = "Break the following instruction into multiple detailed sub-instructions and list them numerically, each sub-instruction must be written in English and end with a period, do not output other information: "

for file_name in ["R2R_val_seen", "R2R_val_unseen", "R2R_test", "R2R_train"]:
    with open(file_name+'.json', 'r') as f:
        data = json.load(f)

    for j in tqdm(range(len(data))):
        data[j]["subgoals"] = []
        for i in range(3):
            response, _ = model.chat(tokenizer, prompt+data[j]["instructions"][i], history=[])
            # print(response)
            index = response.find("1")
            if index == -1:
                # 1 not find
                index = 0
            response = response[index:]
            response = response.split('\n')
            response = [item[3:] for item in response]
            # print(response)
            data[j]["subgoals"].append(response)

    with open(file_name+'_subgoals.json', 'w') as f:
        json.dump(data, f, indent=2)