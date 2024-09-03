from typing import Any, Dict, List

from transformers import AutoTokenizer, pipeline
from trl import AutoModelForCausalLMWithValueHead


class PRM:
    """
    PRM model class.
    
    Args:
        model_name (str): Name of the main model.
        ref_model_name (str): Name of the reference model.
        reward_model_name (str): Name of the reward model.
        device (int or str): Device to run the model on ('cpu' or 'cuda').
        
    Examples:
        >>> prm_model = PRM(
        ...     model_name="lvwerra/gpt2-imdb-pos-v2",
        ...     ref_model_name="lvwerra/gpt2-imdb",
        ...     reward_model_name="lvwerra/distilbert-imdb",
        ...     device=device,
        ... )
        >>> prm_model.generate_responses(
        ...     queries, gen_len=10, gen_kwargs=gen_kwargs
        ... )
        ['Sample response 1', 'Sample response 2']
        >>> prm_model.score_responses(responses, sent_kwargs)
        [0.0, 0.0]
    
    """ 
    def __init__(
        self,
        model_name: str = "lvwerra/gpt2-imdb-pos-v2",
        ref_model_name: str = "lvwerra/gpt2-imdb",
        reward_model_name: str = "lvwerra/distilbert-imdb",
        device=None,
    ):
        """
        Initialize the PRM model with specified models and tokenizer.

        Args:
            model_name (str): Name of the main model.
            ref_model_name (str): Name of the reference model.
            reward_model_name (str): Name of the reward model.
            device (int or str): Device to run the model on ('cpu' or 'cuda').
        """
        self.model_name = model_name
        self.ref_model_name = ref_model_name
        self.reward_model_name = reward_model_name
        self.device = device

        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name
        ).to(device)

        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            ref_model_name
        ).to(device)

        self.reward_pipe = pipeline(
            "sentiment-analysis", model=reward_model_name, device=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(ref_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_responses(
        self, queries: List[str], gen_len: int, gen_kwargs: Dict[str, Any]
    ) -> List[str]:
        """
        Generate responses for a batch of queries.

        Args:
            queries (list of str): List of query strings.
            gen_len (int): Length of the generated response.
            gen_kwargs (dict): Additional keyword arguments for generation.

        Returns:
            list of str: Generated responses.
        """
        responses = []
        for query in queries:
            input_ids = self.tokenizer.encode(query, return_tensors="pt").to(
                self.device
            )
            output_ids = self.model.generate(
                input_ids, max_new_tokens=gen_len, **gen_kwargs
            )
            response = self.tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )
            responses.append(response)
        return responses

    def score_responses(
        self, responses: List[str], sent_kwargs: Dict[str, Any]
    ) -> List[float]:
        """
        Score a batch of responses using the reward pipeline.

        Args:
            responses (list of str): List of response strings.
            sent_kwargs (dict): Additional keyword arguments for sentiment analysis.

        Returns:
            list of float: Scores for each response.
        """
        scores = [
            output[0]["score"]
            for output in self.reward_pipe(responses, **sent_kwargs)
        ]
        return scores

import os

from dotenv import load_dotenv
from swarms.models import OpenAIChat

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# LLM initialization
llm = OpenAIChat(api_key=api_key)


class MathDataGenerator:
    """
    Math data generator for the LLM.

    Args:
        llm (OpenAIChat): LLM model.
        num_iters (int): Number of iterations to run the LLM.

    Returns:
        list of dict: Generated samples.

    Examples:
        >>> llm = OpenAIChat(api_key=api_key)
        >>> mdg = MathDataGenerator(llm, num_iters=10)
        >>> mdg.generate_samples("1 + 1 = 2")
        [{'query': '1 + 1 = 2', 'response': '1 + 1 = 2', 'score': 0.0, 'reward': 0.0}]

    """

    def __init__(self, llm, num_iters):
        self.llm = llm
        self.num_iters = num_iters

    def generate_samples(self, task: str):
        """Generate samples for a given task.

        Args:
            task (str): _description_

        Returns:
            _type_: _description_
        """
        memory = []
        for _ in range(self.num_iters):
            results = self.llm(task)
            memory.append(results)
        return memory

import torch
from zeta.structs import (
    AutoregressiveWrapper,
    Decoder,
    Transformer,
)


class GPT4(torch.nn.Module):
    """
    GPT4 is a transformer architecture that uses a ViT encoder and a transformer decoder.

    Args:

        image_size (int): Size of the image.
        patch_size (int): Size of the patch.
        encoder_dim (int): Dimension of the encoder.
        encoder_depth (int): Depth of the encoder.
        encoder_heads (int): Number of heads in the encoder.
        num_tokens (int): Number of tokens.
        max_seq_len (int): Maximum sequence length.
        decoder_dim (int): Dimension of the decoder.
        decoder_depth (int): Depth of the decoder.
        decoder_heads (int): Number of heads in the decoder.
        alibi_num_heads (int): Number of heads in the alibi attention.
        attn_kv_heads (int): Number of heads in the attention key-value projection.
        use_abs_pos_emb (bool): Whether to use absolute positional embeddings.
        cross_attend (bool): Whether to cross attend in the decoder.
        alibi_pos_bias (bool): Whether to use positional bias in the alibi attention.
        rotary_xpos (bool): Whether to use rotary positional embeddings.
        attn_flash (bool): Whether to use attention flash.
        qk_norm (bool): Whether to normalize the query and key in the attention layer.

    Returns:

            torch.Tensor: The output of the model.

    Usage:

            >>> img = torch.randn(1, 3, 256, 256)
            >>> text = torch.randint(0, 20000, (1, 1024))
            >>> model = GPT4()
            >>> output = model(img, text)
            >>> print(output)

    """

    def __init__(
        self,
        num_tokens=20000,
        max_seq_len=1024,
        decoder_dim=512,
        decoder_depth=6,
        decoder_heads=8,
        alibi_num_heads=4,
        attn_kv_heads=2,
        use_abs_pos_emb=False,
        alibi_pos_bias=True,
        rotary_xpos=True,
        attn_flash=True,
        qk_norm=True,
    ):
        super(GPT4, self).__init__()
        # palm model architecture
        self.decoder = Transformer(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            use_abs_pos_emb=use_abs_pos_emb,
            attn_layers=Decoder(
                dim=decoder_dim,
                depth=decoder_depth,
                heads=decoder_heads,
                alibi_pos_bias=alibi_pos_bias,
                alibi_num_heads=alibi_num_heads,
                rotary_xpos=rotary_xpos,
                attn_kv_heads=attn_kv_heads,
                attn_flash=attn_flash,
                qk_norm=qk_norm,
            ),
        )

        # autoregressive wrapper to enable generation of tokens
        self.decoder = AutoregressiveWrapper(self.decoder)

    def forward(self, text: torch.Tensor):
        """Forward pass of the model."""
        try:
            return self.decoder(text)
        except Exception as error:
            print(f"Failed in forward method: {error}")
            raise
