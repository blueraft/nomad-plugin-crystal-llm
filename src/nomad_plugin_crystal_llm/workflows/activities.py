from temporalio import activity

from nomad_plugin_crystal_llm.workflows.shared import (
    InferenceModelInput,
    InferenceResultsInput,
    InferenceUserInput,
)


@activity.defn
async def get_model(data: InferenceModelInput):
    from .llm import download_model

    await download_model(data.model_path, data.model_url)


@activity.defn
async def construct_model_input(data: InferenceUserInput) -> str:
    # also validates that the input is not empty
    if not data.raw_input and data.input_file:
        with open(data.input_file, "r", encoding="utf-8") as f:
            model_input = f.read()
            return model_input
    if not data.raw_input:
        raise ValueError("Input data cannot be empty.")
    return data.raw_input


@activity.defn
async def run_inference(data: InferenceModelInput) -> list[str]:
    from .llm import evaluate_model

    return evaluate_model(data)


@activity.defn
async def write_results(data: InferenceResultsInput) -> None:
    """
    Write the inference results to a file.
    """
    from .llm import write_cif_files

    write_cif_files(data)
