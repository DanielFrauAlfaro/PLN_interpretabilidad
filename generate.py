import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


set_seed(3407)

use_mingpt = True # use minGPT or huggingface/transformers model?
model_type = 'gpt2'
device = 'cpu'

if use_mingpt:
    model = GPT.from_pretrained(model_type)
else:
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.config.pad_token_id = model.config.eos_token_id # suppress a warning

# ship model to device and set to eval mode
model.to(device)
model.eval();

import numpy as np

def generate(prompt='', num_samples=10, steps=20, do_sample=False, store = False, patch = False, patch_layer = 0, patch_embed = 0):
        
    # tokenize the input prompt into integer input sequence
    # if use_mingpt:
    #     tokenizer = BPETokenizer()
    #     if prompt == '':
    #         # to create unconditional samples...
    #         # manually create a tensor with only the special <|endoftext|> token
    #         # similar to what openai's code does here https://github.com/openai/gpt-2/blob/master/src/generate_unconditional_samples.py
    #         x = torch.tensor([[tokenizer.encoder.encoder['<|endoftext|>']]], dtype=torch.long)
    #     else:
    #         x = tokenizer(prompt).to(device)

    # else:
    #     tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    #     if prompt == '': 
    #         # to create unconditional samples...
    #         # huggingface/transformers tokenizer special cases these strings
    #         prompt = '<|endoftext|>'
    #     encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
    #     x = encoded_input['input_ids']
    
    # we'll process all desired num_samples in a batch, so expand out the batch dim
    x = prompt.expand(num_samples, -1)

    # forward the model `steps` times to get samples, in a batch
    y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, 
                       top_k=None, store = store, patch = patch, 
                       patch_layer = patch_layer, patch_embed = patch_embed)

    return y



model.eval()

# Frases de input
input1 = "Michelle Jones was a top-notch student. Michelle"
input2 = "Michelle Smith was a top-notch student. Michelle"

# tokenize the input prompt into integer input sequence
tokenizer = BPETokenizer()

# Tokens de la primera frase
x = tokenizer(input1)
tokens_str = [tokenizer.decode(torch.tensor([token])) for token in x[0]]

# Genera la salida original
y, original_output = generate(prompt=x, num_samples=1, steps=1, store = True)

names = tokenizer.decode(y.cpu().squeeze())

index_smith = names.index("Jones")
or_smith = original_output[0][index_smith]

# Tokens de la segunda frase
tokens = tokenizer(input2)
input_length = tokens[0].shape[-1]

# Matriz resultado
res = np.zeros((12, 6))

# Para cada layer y cada embedding
for i in range(12):
    print("Layer ", i)

    for j in range(input_length):
        print("--- Embed.: ", j)

	# Genera la salida corrputa
        y, out_np = generate(prompt=tokens, num_samples=1, steps=1, store = False, patch=True,
                          patch_layer=i, patch_embed = j)

	
	# Destokeniza la imagen
        names = tokenizer.decode(y.cpu().squeeze())
	
	# Obtiene los índices de los nombres
        index_smith = names.index("He")
        index_jones = names.index("She")

	# Obtiene sus logits
        smith = out_np[0][index_smith]
        jones = out_np[0][index_jones]
		
	# Distancia
        distancia = (index_smith - index_jones)
        res[i][j] = distancia


# Crear un DataFrame de pandas si tus datos no están ya en uno.
# Los índices y columnas deben coincidir con los de tu matriz real.
df = pd.DataFrame(res, index=[f'layer {i}' for i in range(res.shape[0])],
                  columns=[f'(embed {i}) {tokens_str[i]}' for i in range(res.shape[1])])

# Ajustar la paleta de colores para que coincida con tu imagen.
# Puedes jugar con diferentes paletas de seaborn o matplotlib para obtener los colores deseados.
cmap = sns.color_palette("coolwarm", as_cmap=True)

# Crear el mapa de calor.
plt.figure(figsize=(10, 8))
sns.heatmap(df, annot=False, fmt="f", cmap=cmap, cbar_kws={'label': 'Magnitude'})

# Ajustar la visualización aquí para que coincida con tu imagen.
plt.xticks(rotation=45, ha='right')  # Rota las etiquetas del eje x si es necesario.
plt.yticks(rotation=0)
plt.tight_layout()  # Asegura que todo encaje bien en la figura.

# Mostrar el mapa de calor.
plt.show()

