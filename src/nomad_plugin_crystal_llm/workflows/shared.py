from dataclasses import dataclass


@dataclass
class InferenceUserInput:
    """
    User input data for the inference workflow.

    Attributes:
    - input_file: Path to a file containing the input prompt.
    - raw_input: Raw input string to use as a prompt.
    - generate_cif: If True, the model will generate CIF files.
    """

    input_file: str | None = None
    raw_input: str | None = None
    generate_cif: bool = False


@dataclass
class InferenceModelInput:
    """
    Model input data for the inference workflow.

    Attributes:

    - model_path: Path to the model file.
    - model_url: URL to download the model if not available locally.
    - raw_input: Raw input string to use as a prompt.
    - num_samples: Number of samples to draw during inference.
    - max_new_tokens: Maximum number of tokens to generate in each sample.
    - temperature: Controls the randomness of predictions (1.0 = no change, < 1.0 =
        less random, > 1.0 = more random).
    - top_k: Retain only the top_k most likely tokens, clamp others to have 0
        probability.
    - seed: Random seed for reproducibility.
    - dtype: Data type for the model (e.g., 'float32', 'bfloat16', 'float16').
    - compile: Whether to use PyTorch 2.0 to compile the model for faster inference.
    """

    raw_input: str
    model_path: str = "models/crystallm_v1_small/ckpt.pt"
    model_url: str = (
        "https://zenodo.org/records/10642388/files/crystallm_v1_small.tar.gz"
    )
    num_samples: int = 2
    max_new_tokens: int = 3000
    temperature: float = 0.8
    top_k: int = 10
    seed: int = 1337
    dtype: str = "bfloat16"
    compile: bool = False


@dataclass
class InferenceResultsInput:
    """
    CIF Results input data for the inference workflow.

    Attributes:
    - generate_cif: If True, the model will generate CIF files.
    - generated_samples: List to store generated samples from the model.
    """

    generated_samples: list[str]
    generate_cif: bool
