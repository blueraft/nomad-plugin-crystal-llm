from nomad.orchestrator.base import BaseWorkflowHandler
from nomad.orchestrator.shared.constant import TaskQueue
from pydantic import BaseModel


class CrystalLLMEntryPoint(BaseModel):
    entry_point_type: str = 'workflow'

    def load(self):
        from nomad_plugin_crystal_llm.workflows.activities import (
            construct_model_input,
            get_model,
            run_inference,
            write_results,
        )
        from nomad_plugin_crystal_llm.workflows.workflow import InferenceWorkflow

        return BaseWorkflowHandler(
            task_queue=TaskQueue.GPU,
            workflows=[InferenceWorkflow],
            activities=[get_model, construct_model_input, run_inference, write_results],
        )


crystal_llm = CrystalLLMEntryPoint()
