from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad_plugin_crystal_llm.workflows.activities import (
        construct_model_input,
        get_model,
        run_inference,
        write_results,
    )
    from nomad_plugin_crystal_llm.workflows.shared import (
        InferenceInput,
        InferenceModelInput,
        InferenceResultsInput,
    )


@workflow.defn(name="nomad_plugin_crystal_llm.workflows.InferenceWorkflow")
class InferenceWorkflow:
    @workflow.run
    async def run(self, data: InferenceInput) -> list[str]:
        raw_input = await workflow.execute_activity(
            construct_model_input,
            data,
            start_to_close_timeout=timedelta(seconds=60),
        )
        model_data = InferenceModelInput(raw_input=raw_input)
        await workflow.execute_activity(
            get_model,
            model_data,
            start_to_close_timeout=timedelta(seconds=120),
        )
        generated_samples = await workflow.execute_activity(
            run_inference,
            model_data,
            start_to_close_timeout=timedelta(seconds=120),
        )
        await workflow.execute_activity(
            write_results,
            InferenceResultsInput(
                user_id=data.user_id,
                upload_id=data.upload_id,
                generated_samples=generated_samples,
                generate_cif=data.generate_cif,
            ),
            start_to_close_timeout=timedelta(seconds=60),
        )
        return generated_samples
