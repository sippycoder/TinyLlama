import torch

from config import ShadowCloneManager
from config import ShadowCloneConfig
from model import ShadowCloneForCausalLM

# from transformers import MixtralConfig, MixtralForCausalLM

config = ShadowCloneConfig.from_pretrained(".")
device = "cuda:0"
model = ShadowCloneForCausalLM(config)
model = model.to(device)

print ("model init")

x = torch.randint(1, 32000, (1, 1024)).to(device)
labels = torch.randint(1, 32000, (1, 1024)).to(device)

mgr = ShadowCloneManager.get_instance()

mgr.current_scale = 0
out = model(x, labels=labels, output_router_logits=True)

mgr.current_scale += 1
out = model(x, labels=labels, output_router_logits=True)

mgr.current_scale += 1
out = model(x, labels=labels, output_router_logits=True)

print (out.logits.shape)
