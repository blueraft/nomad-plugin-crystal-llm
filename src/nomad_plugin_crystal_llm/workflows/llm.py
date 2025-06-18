import asyncio
import json
import os
import shutil
import tarfile
import tempfile
from contextlib import nullcontext

import aiohttp
import torch
from crystallm import (
    GPT,
    CIFTokenizer,
    GPTConfig,
)
from nomad.app.v1.routers.uploads import get_upload_with_read_access
from nomad.datamodel import User
from nomad.orchestrator.util import get_upload_files

from nomad_plugin_crystal_llm.schemas.schema import (
    CrystaLLMInferenceResult,
    InferenceSettings,
)
from nomad_plugin_crystal_llm.workflows.shared import (
    InferenceModelInput,
    InferenceResultsInput,
)

BLOCK_SIZE = 1024


async def download_model(model_path: str, model_url: str | None = None) -> dict:
    """
    Checks if the model file exists locally, and if not, downloads it from the
    provided URL.
    """
    # Check if file exists asynchronously
    exists = await asyncio.to_thread(os.path.exists, model_path)
    if not exists and not model_url:
        raise FileNotFoundError(
            f'Model file "{model_path}" does not exist and `model_url` is not provided.'
        )
    elif exists and model_url:
        return {
            'model_path': model_path,
            'model_url': model_url,
        }
    elif exists:
        return {'model_path': model_path}

    # Download the model from the URL and copy the model file to the model_path
    with tempfile.TemporaryDirectory() as tmpdir:
        async with aiohttp.ClientSession() as session:
            async with session.get(model_url) as response:
                response.raise_for_status()
                # Download in chunks
                tmp_zipfile = os.path.join(tmpdir, model_url.split('/')[-1])
                loop = asyncio.get_running_loop()
                with open(tmp_zipfile, 'wb') as f:
                    async for chunk in response.content.iter_chunked(BLOCK_SIZE):
                        await loop.run_in_executor(None, f.write, chunk)
        # Unpack the model zip
        with tarfile.open(tmp_zipfile, 'r:gz') as tar:
            tar.extractall(tmpdir)
        tmp_zipdir = tmp_zipfile.split('.')[0]
        # Check if '.pt' file exists in the extracted directory
        model_files = [f for f in os.listdir(tmp_zipdir) if f.endswith('.pt')]
        if not model_files:
            raise FileNotFoundError(
                'No ".pt" file found in the extracted directory '
                f'"{os.path.dirname(model_path)}".'
            )
        # Move over the first .pt file found to the model_path
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        shutil.move(os.path.join(tmp_zipdir, model_files[0]), model_path)

    return {'model_path': model_path, 'model_url': model_url}


def evaluate_model(inference_state: InferenceModelInput) -> list[str]:
    """
    Evaluate the model with the given parameters.
    Adapted from https://github.com/lantunes/CrystaLLM
    """
    torch.manual_seed(inference_state.seed)
    torch.cuda.manual_seed(inference_state.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device = (
        'cuda' if torch.cuda.is_available() else 'cpu'
    )  # for later use in torch.autocast
    ptdtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }[inference_state.dtype]
    ctx = (
        nullcontext()
        if device == 'cpu'
        else torch.amp.autocast(device_type=device, dtype=ptdtype)
    )

    tokenizer = CIFTokenizer()
    encode = tokenizer.encode
    decode = tokenizer.decode

    checkpoint = torch.load(inference_state.model_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    if inference_state.compile:
        model = torch.compile(model)

    # encode the beginning of the prompt
    prompt = inference_state.raw_input
    start_ids = encode(tokenizer.tokenize_cif(prompt))
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # run generation
    generated = []
    with torch.no_grad():
        with ctx:
            for k in range(inference_state.num_samples):
                y = model.generate(
                    x,
                    inference_state.max_new_tokens,
                    temperature=inference_state.temperature,
                    top_k=inference_state.top_k,
                )
                generated.append(decode(y[0].tolist()))

    return generated


def write_cif_files(result: InferenceResultsInput) -> None:
    """
    Write the generated CIFs to the specified target (console or file).
    """
    if not result.generate_cif:
        return
    upload_files = get_upload_files(result.upload_id, result.user_id)
    if not upload_files:
        raise ValueError(
            f'No upload files found for upload_id "{result.upload_id}" '
            f'and user_id "{result.user_id}".'
        )
    cif_paths = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for k, sample in enumerate(result.generated_samples):
            fname = os.path.join(tmpdir, f'{result.cif_prefix}_{k + 1}.cif')
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(sample)
            upload_files.add_rawfiles(fname, target_dir=result.cif_dir)
            cif_paths.append(
                os.path.join(result.cif_dir, f'{result.cif_prefix}_{k + 1}.cif')
            )
    return cif_paths


def write_entry_archive(cif_paths, result: InferenceResultsInput) -> str:
    """
    Create an entry for the inference results and add it to the upload.
    """

    # upload_files = get_upload_files(result.upload_id, result.user_id)
    upload = get_upload_with_read_access(
        result.upload_id,
        User(user_id=result.user_id),
        include_others=True,
    )
    inference_result = CrystaLLMInferenceResult(
        prompt=result.model_data.raw_input,
        workflow_id=result.cif_dir,
        generated_cifs=cif_paths,
        inference_settings=InferenceSettings(
            model=result.model_data.model_url.rsplit('/', 1)[-1].split('.tar.gz')[0],
            num_samples=result.model_data.num_samples,
            max_new_tokens=result.model_data.max_new_tokens,
            temperature=result.model_data.temperature,
            top_k=result.model_data.top_k,
            seed=result.model_data.seed,
            dtype=result.model_data.dtype,
            compile=result.model_data.compile,
        ),
    )
    fname = os.path.join('inference_result.archive.json')
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump({'data': inference_result.m_to_dict(with_root_def=True)}, f, indent=4)
    upload.process_upload(
        file_operations=[
            dict(op='ADD', path=fname, target_dir=result.cif_dir, temporary=True)
        ],
        only_updated_files=True,
    )
