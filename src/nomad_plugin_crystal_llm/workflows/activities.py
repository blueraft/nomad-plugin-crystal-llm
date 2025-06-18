import os

from nomad.orchestrator.util import workflow_artifacts_dir
from temporalio import activity

from nomad_plugin_crystal_llm.workflows.shared import (
    InferenceInput,
    InferenceModelInput,
    InferenceResultsInput,
)


@activity.defn
async def get_model(data: InferenceModelInput):
    from .llm import download_model

    model_path = os.path.join(workflow_artifacts_dir(), data.model_path)
    await download_model(model_path, data.model_url)


@activity.defn
async def construct_model_input(data: InferenceInput) -> str:
    # validates that the input is not empty
    if not data.raw_input:
        raise ValueError('Input data cannot be empty.')
    return data.raw_input


@activity.defn
async def run_inference(data: InferenceModelInput) -> list[str]:
    from .llm import evaluate_model

    data.model_path = os.path.join(workflow_artifacts_dir(), data.model_path)
    return evaluate_model(data)


@activity.defn
async def write_results(data: InferenceResultsInput) -> None:
    """
    Write the inference results to a file.
    """
    from .llm import write_cif_files, write_entry_archive

    cif_paths = write_cif_files(data)
    if not cif_paths:
        raise ValueError('No CIF files were generated.')
    write_entry_archive(cif_paths, data)
