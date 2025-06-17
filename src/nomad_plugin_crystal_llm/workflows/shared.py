from dataclasses import dataclass


@dataclass
class InferenceUserInput:
    """
    User input data for the inference workflow.

    Attributes:
    - raw_input: Raw input string to use as a prompt.
    - user_id: User making the request
    - upload_id: If `generate_cif` is set to True, save CIF files to this upload.
    - generate_cif: If True, the model will generate CIF files.
    """

    upload_id: str
    user_id: str
    raw_input: str
    generate_cif: bool = False
    model_path: str = 'models/crystallm_v1_small/ckpt.pt'
    model_url: str = (
        'https://zenodo.org/records/10642388/files/crystallm_v1_small.tar.gz'
    )
    num_samples: int = 2
    max_new_tokens: int = 3000
    temperature: float = 0.8
    top_k: int = 10
    seed: int = 1337
    dtype: str = 'bfloat16'
    compile: bool = False


@dataclass
class InferenceInput:
    """
    User input data for the inference workflow.

    Attributes:
    - raw_input: Raw input string to use as a prompt.
    - user_id: User making the request
    - upload_id: If `generate_cif` is set to True, save CIF files to this upload.
    - generate_cif: If True, the model will generate CIF files.
    - model_path: Path to the model file.
    - model_url: URL to download the model if not available locally.
    - num_samples: Number of samples to draw during inference.
    - max_new_tokens: Maximum number of tokens to generate in each sample.
    - temperature: Controls the randomness of predictions. Lower values make the
        model more deterministic, while higher values increase randomness.
    - top_k: Retain only the top_k most likely tokens, clamp others to have 0
        probability.
    - seed: Random seed for reproducibility.
    - dtype: Data type for the model (e.g., 'float32', 'bfloat16', 'float16').
    - compile: Whether to compile the model for faster inference.
    """

    upload_id: str
    user_id: str
    raw_input: str
    generate_cif: bool = False
    model_path: str = 'models/crystallm_v1_small/ckpt.pt'
    model_url: str = (
        'https://zenodo.org/records/10642388/files/crystallm_v1_small.tar.gz'
    )
    num_samples: int = 2
    max_new_tokens: int = 3000
    temperature: float = 0.8
    top_k: int = 10
    seed: int = 1337
    dtype: str = 'bfloat16'
    compile: bool = False


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
    - temperature: Controls the randomness of predictions. Lower values make the
        model more deterministic, while higher values increase randomness.
    - top_k: Retain only the top_k most likely tokens, clamp others to have 0
        probability.
    - seed: Random seed for reproducibility.
    - dtype: Data type for the model (e.g., 'float32', 'bfloat16', 'float16').
    - compile: Whether to compile the model for faster inference.
    """

    raw_input: str
    model_path: str = 'models/crystallm_v1_small/ckpt.pt'
    model_url: str = (
        'https://zenodo.org/records/10642388/files/crystallm_v1_small.tar.gz'
    )
    num_samples: int = 2
    max_new_tokens: int = 3000
    temperature: float = 0.8
    top_k: int = 10
    seed: int = 1337
    dtype: str = 'bfloat16'
    compile: bool = False


@dataclass
class InferenceResultsInput:
    """
    CIF Results input data for the inference workflow.

    Attributes:
    - upload_id: If generate_cif, write the generate CIF files to the upload.
    - user_id: User making the request
    - generate_cif: If True, the model will generate CIF files.
    - generated_samples: List to store generated samples from the model.
    - cif_dir: Directory to save CIF files. If empty, uses the upload's raw directory.
    - cif_prefix: Prefix for the generated CIF files: <cif_prefix>_<index>.cif
    """

    upload_id: str
    user_id: str
    generated_samples: list[str]
    generate_cif: bool
    model_data: InferenceModelInput
    cif_dir: str = ''  # empty string means the upload's raw directory
    cif_prefix: str = 'sample'
