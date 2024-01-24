import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from openxlab.model import download


download(model_repo='OpenLMLab/internlm2-chat-7b',output='/home/xlab-app-center/internlm/internlm2-chat-7b')

tokenizer = AutoTokenizer.from_pretrained("/home/xlab-app-center/internlm/internlm2-chat-7b",trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/home/xlab-app-center/internlm/internlm2-chat-7b",trust_remote_code=True, torch_dtype=torch.float16).cuda()

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="InternLM2-Chat-7B",
                description="""
InternLM is mainly developed by Shanghai AI Laboratory.  
                 """,
                 ).queue(1).launch()
