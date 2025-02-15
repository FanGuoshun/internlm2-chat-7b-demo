import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from openxlab.model import download

# base_path = './internlm2-chat-7b'

# download(model_repo='OpenLMLab/internlm2-chat-7b',output=base_path)

# tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()

def chat(message,history):
    base_path = './internlm2-chat-7b'

    download(model_repo='OpenLMLab/internlm2-chat-7b',output=base_path)

    tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="InternLM2-Chat-7B",
                description="""
InternLM is mainly developed by Shanghai AI Laboratory.  
                 """,
                 ).queue(1).launch()
