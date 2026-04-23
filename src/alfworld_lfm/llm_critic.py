from huggingface_hub import snapshot_download
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

# from mistral_inference.transformer import Transformer
# from mistral_inference.generate import generate


import torch

class LLM_Critic:
    def __init__(self):
        self.pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")

    def query_llm(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant for completing tasks in alfworld. Only answer the given prompt. Do not add unnecessary detail."},
            {"role": "user", "content": prompt},
        ]
        
        response = self.pipe(messages, max_new_tokens=256)
        return response[0]['generated_text'][2]['content']



def test(pipe):
    prompt = 'You will be shown a playthrough for solving a task. Afterwards, answer some questions about how you performed.\nTask: put some alarmclock on desk.\nBefore: -= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, and a garbagecan 1.\n\n----------\nStep 1\nYour action: look\nResult: You are in the middle of a room. Looking quickly around you, you see nothing.\n----------\nStep 2\nYour action: go to dresser 1\nResult: You arrive at dresser 1. On the dresser 1, you see a alarmclock 2, a alarmclock 1, a creditcard 2, and a creditcard 1.\n----------\nStep 3\nYour action: take alarmclock 1 from dresser 1\nResult: You pick up the alarmclock 1 from the dresser 1.\n----------\nStep 4\nYour action: go to desk 1\nResult: You arrive at desk 1. On the desk 1, you see a book 1, a cd 2, a desklamp 1, a keychain 2, a keychain 1, and a pencil 1.\n----------\nStep 5\nYour action: move alarmclock 1 to desk 1\nResult: You move the alarmclock 1 to the desk 1.\n----------\nDid you see anything that makes you believe you are on the right track to solve the task? Answer yes or no. If yes, then list ALL the steps when you saw helpul things by the step number. List only in bullet form.'
    messages = [
        {"role": "system", "content": "You are a helpful assistant for completing tasks in alfworld. Only answer the given prompt. Do not add unnecessary detail."},
        {"role": "user", "content": prompt},
    ]
    
    test = pipe(messages)
    print(test)

def testing():
    llm = LLM_Critic()
    llm.query_llm()




if __name__ == "__main__":
    testing()
#     pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")
#     test(pipe)
    # model_path = Path.home().joinpath('Desktop', 'COS435_Assignments', 'COS435_FinalProject', 'mistral_models', '7B-Instruct-v0.3')
    # if not model_path.is_dir():
    #     download_model()

    # MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

    # gpu_available = torch.cuda.is_available()

    # if gpu_available:
    #     print("gpu")
    #     model, tokenizer = load_model(model_path, MODEL_NAME, gpu_available)
    #     result = query_llm_gpu("what is your name?")
    # else:
    #     print("no gpu")
    #     model, tokenizer = load_model(model_path, MODEL_NAME, gpu_available)
    #     print("finished loading model")
    #     result = query_llm_cpu_2(model, tokenizer, "what is your name?")
    #     print(result)

    