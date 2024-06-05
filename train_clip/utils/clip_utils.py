from flags import device
from modules.utils import gen_shape_description_simple
import torch
import clip

from utils.dbe import dbe

def gen_word_objs_embeddings(obj, clip_model, context_length=77):
    # shape_description = gen_shape_description(obj)
    shape_description = gen_shape_description_simple(obj)

    # Tokenize text with default clip model
    # text = clip.tokenize(shape_description, context_length=context_length).to(device)
    text = clip.tokenize(obj, context_length=context_length).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text)

    return text_features


def gen_word_objs_embeddings_batch(objs, clip_model, context_length=77):
    # Generate descriptions for the whole batch
    shape_descriptions = [gen_shape_description_simple(obj) for obj in objs]

    # Concatenate the single lists into strings
    shape_descriptions = [''.join(desc) for desc in shape_descriptions]

    # Tokenize the batch of text descriptions
    text = clip.tokenize(shape_descriptions, context_length=context_length).to(device)
    text = text.long()

    with torch.no_grad():
        text_features = clip_model.encode_text(text)

    return text_features