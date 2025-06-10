from pydantic import BaseModel
from nomad.orchestrator.base import BaseWorkflowHandler
from nomad.orchestrator.shared.constant import TaskQueue


class CrystalLLMEntryPoint(BaseModel):
    entry_point_type: str = "workflow"

    def load(self):
        from nomad_plugin_crystal_llm.workflows.workflow import (
            InferenceWorkflow,
        )
        from nomad_plugin_crystal_llm.workflows.activities import (
            get_model,
            construct_model_input,
            run_inference,
            write_results,
        )

        return BaseWorkflowHandler(
            task_queue=TaskQueue.GPU,
            workflows=[InferenceWorkflow],
            activities=[get_model, construct_model_input, run_inference, write_results],
        )


crystal_llm = CrystalLLMEntryPoint()
