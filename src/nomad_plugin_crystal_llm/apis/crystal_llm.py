import uuid
from fastapi import Depends, FastAPI, HTTPException
from nomad.config import config
from nomad.app.v1.models import User
from nomad.app import main
from nomad.orchestrator.shared.constant import TaskQueue
from nomad.app.v1.routers.auth import create_user_dependency

from nomad_plugin_crystal_llm.workflows.shared import InferenceInput, InferenceUserInput

crystal_llm_api_entrypoint = config.get_plugin_entry_point(
    "nomad_plugin_crystal_llm.apis:crystal_llm_api"
)

root_path = f"{config.services.api_base_path}/{crystal_llm_api_entrypoint.prefix}"
print(root_path)

app = FastAPI(root_path=root_path)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/start-inference-task")
async def start_inference_task(
    data: InferenceUserInput,
    user: User = Depends(create_user_dependency(required=True)),
):
    workflow_id = f"crystal-llm-workflow-{user.user_id}-{uuid.uuid4()}"
    client = main.temporal_client()
    workflow_data = InferenceInput(
        raw_input=data.raw_input,
        generate_cif=data.generate_cif,
        upload_id=data.upload_id,
        user_id=user.user_id,
    )
    try:
        await client.start_workflow(
            "InferenceWorkflow", workflow_data, id=workflow_id, task_queue=TaskQueue.GPU
        )
        return {"workflow_id": workflow_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
