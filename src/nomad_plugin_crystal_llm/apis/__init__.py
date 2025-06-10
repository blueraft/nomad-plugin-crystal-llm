from nomad.config.models.plugins import APIEntryPoint


class CrystalLLMAPIEntryPoint(APIEntryPoint):
    def load(self):
        from nomad_plugin_crystal_llm.apis.crystal_llm import app

        return app


crystal_llm_api = CrystalLLMAPIEntryPoint(
    prefix="crystal-llm",
    name="CrystalLLM API",
    description="Crystal LLM custom API.",
)
