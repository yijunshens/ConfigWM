"""Text encoder module for ConfigurableEncoder."""


def create_text_encoder(type, model_name, **kwargs):
    if type == "sentence_transformer":
        from .sentence_transformer import SentenceTransformerTextEncoder

        return SentenceTransformerTextEncoder(model_name, **kwargs)
    if type == "pe_core":
        from .pe_core import PECoreTextEncoder

        return PECoreTextEncoder(model_name, **kwargs)
    raise ValueError(f"Unsupported text encoder type: {type}")
