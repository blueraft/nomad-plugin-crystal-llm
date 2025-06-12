from nomad.config.models.plugins import SchemaPackageEntryPoint


class CrystallmSchemaPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_plugin_crystal_llm.schema import m_package

        return m_package


crystal_llm_schemas = CrystallmSchemaPackageEntryPoint(
    name='Crystallm Schema',
    description='Schema for running Crystal LLM on NOMAD deployments.',
)
