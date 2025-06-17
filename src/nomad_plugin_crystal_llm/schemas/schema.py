from ase.io import read
from matid import SymmetryAnalyzer
from nomad.app.v1.routers.uploads import get_upload_with_read_access
from nomad.datamodel.data import ArchiveSection, EntryData, EntryDataCategory
from nomad.datamodel.metainfo.annotations import (
    ELNAnnotation,
    ELNComponentEnum,
    SectionProperties,
)
from nomad.datamodel.results import Material, Results, SymmetryNew, System
from nomad.metainfo import Category, MEnum, Quantity, SchemaPackage, Section, SubSection
from nomad.normalizing.common import nomad_atoms_from_ase_atoms
from nomad.normalizing.topology import add_system, add_system_info
from nomad.orchestrator import util as orchestrator_utils
from nomad.orchestrator.shared.constant import TaskQueue

from nomad_plugin_crystal_llm.workflows.shared import InferenceUserInput

m_package = SchemaPackage()


class InferenceCategory(EntryDataCategory):
    """Category for inference workflows."""

    m_def = Category(label='Inference Workflows', categories=[EntryDataCategory])


class RunWorkflowAction(ArchiveSection):
    """Abstract section to run inference workflows"""

    trigger_run_workflow = Quantity(
        type=bool,
        description='Starts an asynchronous workflow for running the inference.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.ActionEditQuantity,
            label='Run Inference Workflow',
        ),
    )

    def run_workflow(self, archive, logger=None):
        """Run the workflow with the provided archive."""
        raise NotImplementedError('This method should be implemented in subclasses.')

    def normalize(self, archive, logger=None):
        """Normalize the section to ensure it is ready for processing."""
        super().normalize(archive, logger)
        if self.trigger_run_workflow:
            try:
                self.run_workflow(archive, logger)
            except Exception as e:
                logger.error(f'Error running workflow: {e}. ')
            self.trigger_run_workflow = False


class InferenceSettings(ArchiveSection):
    """Settings for CrystaLLM inference workflows."""

    model = Quantity(
        type=MEnum(
            [
                'crystallm_v1_small',
                'crystallm_v1_large',
            ]
        ),
        description="""
        Model used for inference.
        | Available models                                 | Description                                                                         |
        |--------------------------------------------------|-------------------------------------------------------------------------------------|
        | **crystallm_v1_small  (25.36M parameters)**      | Downloadable at https://zenodo.org/records/10642388/files/crystallm_v1_small.tar.gz |
        | **crystallm_v1_large  (1.2B parameters)**        | Downloadable at https://zenodo.org/records/10642388/files/crystallm_v1_large.tar.gz |
        """,  # noqa: E501
    )
    num_samples = Quantity(
        type=int,
        description='Number of samples to draw during inference.',
    )
    max_new_tokens = Quantity(
        type=int,
        description='Maximum number of tokens to generate in each sample.',
    )
    temperature = Quantity(
        type=float,
        description='Controls the randomness of predictions. Lower values make the '
        'model more deterministic, while higher values increase randomness.',
    )
    top_k = Quantity(
        type=int,
        description='Retain only the top_k most likely tokens, clamp others to have 0 '
        'probability.',
    )
    seed = Quantity(
        type=int,
        description='Random seed for reproducibility.',
    )
    dtype = Quantity(
        type=MEnum(['float32', 'bfloat16', 'float16']),
        description='Data type for the model (e.g., "float32", "bfloat16", "float16").',
    )
    compile = Quantity(
        type=bool,
        description='Whether to compile the model for faster inference.',
    )


class InferenceSettingsForm(ArchiveSection):
    """Settings used for CrystaLLM inference workflows with editable fields."""

    model = InferenceSettings.model.m_copy(deep=True)
    model.default = 'crystallm_v1_small'
    model.m_annotations['eln'] = ELNAnnotation(
        component=ELNComponentEnum.EnumEditQuantity,
    )

    num_samples = InferenceSettings.num_samples.m_copy(deep=True)
    num_samples.default = 1
    num_samples.m_annotations['eln'] = ELNAnnotation(
        component=ELNComponentEnum.NumberEditQuantity,
    )

    max_new_tokens = InferenceSettings.max_new_tokens.m_copy(deep=True)
    max_new_tokens.default = 3000
    max_new_tokens.m_annotations['eln'] = ELNAnnotation(
        component=ELNComponentEnum.NumberEditQuantity,
    )

    temperature = InferenceSettings.temperature.m_copy(deep=True)
    temperature.default = 0.8
    temperature.m_annotations['eln'] = ELNAnnotation(
        component=ELNComponentEnum.NumberEditQuantity,
    )

    top_k = InferenceSettings.top_k.m_copy(deep=True)
    top_k.default = 10
    top_k.m_annotations['eln'] = ELNAnnotation(
        component=ELNComponentEnum.NumberEditQuantity,
    )

    seed = InferenceSettings.seed.m_copy(deep=True)
    seed.default = 1337
    seed.m_annotations['eln'] = ELNAnnotation(
        component=ELNComponentEnum.NumberEditQuantity,
    )

    dtype = InferenceSettings.dtype.m_copy(deep=True)
    dtype.default = 'bfloat16'
    dtype.m_annotations['eln'] = ELNAnnotation(
        component=ELNComponentEnum.EnumEditQuantity,
    )

    compile = InferenceSettings.compile.m_copy(deep=True)
    compile.default = False
    compile.m_annotations['eln'] = ELNAnnotation(
        component=ELNComponentEnum.BoolEditQuantity,
    )


class InferenceStatus(ArchiveSection):
    """Section to fetch the status of an inference workflow."""

    m_def = Section(
        a_eln=ELNAnnotation(
            overview=True,
            order=['workflow_id', 'trigger_workflow_status'],
        )
    )
    workflow_id = Quantity(
        type=str,
        description='ID of the `temporalio` workflow.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity),
    )
    status = Quantity(
        type=str,
        description='Status of the inference workflow.',
    )
    generated_entry = Quantity(
        type=str,
        description='Reference to the generated entry after the workflow completes.',
    )

    def normalize(self, archive, logger=None):
        """Normalize the section to ensure it is ready for processing."""
        super().normalize(archive, logger)
        if not self.status or self.status == 'RUNNING':
            try:
                status = orchestrator_utils.get_workflow_status(self.workflow_id)
                if status:
                    self.status = status.name
            except Exception as e:
                logger.error(f'Error getting workflow status: {e}. ')
            # if self.status == 'COMPLETED':
            #     self.generated_entry = get_reference_from_mainfile(
            #         archive,
            #         mainfile=os.path.join(
            #             self.workflow_id, 'crystallm_inference.archive.json'
            #         ),
            #     )


class InferenceForm(RunWorkflowAction):
    """Settings form for CrystaLLM inference workflows with editable fields."""

    m_def = Section(
        a_eln=ELNAnnotation(
            overview=True,
            order=[
                'prompt',
                'trigger_run_workflow',
                'inference_settings',
            ],
        )
    )

    prompt = Quantity(
        type=str,
        description='Prompt to be used for inference.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity),
    )
    inference_settings = SubSection(
        section_def=InferenceSettingsForm,
        description='Settings for the CrystaLLM inference workflow.',
    )

    def run_workflow(self, archive, logger=None):
        """
        Run the CrystaLLM inference workflow with the provided archive.
        Uses the first author's credentials to run the workflow.
        """
        if not self.prompt:
            logger.warn(
                'No prompt provided for the CrystaLLM inference workflow. '
                'Cannot run the workflow.'
            )
            return
        if not archive.metadata.authors:
            logger.warn(
                'No authors found in the archive metadata. '
                'Cannot run CrystaLLM inference workflow.'
            )
            return
        if not self.inference_settings:
            self.inference_settings = InferenceSettingsForm()
        input_data = InferenceUserInput(
            user_id=archive.metadata.authors[0].user_id,
            upload_id=archive.metadata.upload_id,
            raw_input=self.prompt,
            generate_cif=True,
            model_path=f'models/{self.inference_settings.model}/ckpt.pt',
            model_url=(
                'https://zenodo.org/records/10642388/files/'
                f'{self.inference_settings.model}.tar.gz'
            ),
            num_samples=self.inference_settings.num_samples,
            max_new_tokens=self.inference_settings.max_new_tokens,
            temperature=self.inference_settings.temperature,
            top_k=self.inference_settings.top_k,
            seed=self.inference_settings.seed,
            dtype=self.inference_settings.dtype,
            compile=self.inference_settings.compile,
        )
        workflow_name = 'nomad_plugin_crystal_llm.workflows.InferenceWorkflow'
        workflow_id = orchestrator_utils.run_workflow(
            workflow_name=workflow_name, data=input_data, task_queue=TaskQueue.GPU
        )
        # TODO: Add inference status section to the archive


class CrystaLLMInferenceResult(EntryData):
    """Result of a CrystaLLM inference workflow."""

    m_def = Section(
        label='CrystaLLM Inference Result',
        a_eln=ELNAnnotation(
            properties=SectionProperties(
                order=[
                    'prompt',
                    'workflow_id',
                    'status',
                    'trigger_workflow_status',
                    'generated_cifs',
                    'generated_structures',
                    'inference_settings',
                ]
            ),
        ),
    )
    prompt = Quantity(
        type=str,
        description='Prompt to be used for inference.',
    )
    workflow_id = Quantity(
        type=str,
        description='ID of the `temporalio` workflow.',
    )
    generated_cifs = Quantity(
        type=str,
        shape=['*'],
        description='Path to the CIF generated by the LLM.',
    )
    generated_structures = Quantity(
        type=System,
        shape=['*'],
        description='Reference to the system normalized based on the generated CIF.',
    )
    inference_settings = SubSection(
        section_def=InferenceSettings,
        description='Settings used for the CrystaLLM inference workflow.',
    )

    def normalize(self, archive, logger=None):
        """Normalize the section to ensure it is ready for processing."""
        super().normalize(archive, logger)
        # TODO: Add logic to normalize material and topology sections


class CrystaLLMInference(EntryData):
    """
    Section for running CrystaLLM inference workflows.
    """

    m_def = Section(
        label='CrystaLLM Inference',
        categories=[InferenceCategory],
        a_eln=ELNAnnotation(
            properties=SectionProperties(
                order=[
                    'name',
                    'description',
                    'inference_form',
                    'results',
                ]
            ),
        ),
    )
    name = Quantity(
        type=str,
        description='Name of the inference workflow.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity),
    )
    description = Quantity(
        type=str,
        description='Description of the inference workflow.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.RichTextEditQuantity,
            props=dict(height=150),
        ),
    )
    inference_form = SubSection(
        section_def=InferenceForm,
        description='Settings for the CrystaLLM inference workflow.',
    )
    inference_statuses = SubSection(
        section_def=InferenceStatus,
        description='Results of the inference workflow.',
        repeats=True,
    )

    def process_generated_cifs(self, archive, logger):
        """
        Process the generated CIFs in `archive.data.results` and populates
        `archive.results.material`.
        """
        if not self.results:
            return
        elements = set()
        topologies = {}
        for result in self.results:
            if not result.generated_cifs:
                continue
            # get the cifs from the results
            system_references = []
            for cif in result.generated_cifs:
                if not cif:
                    continue
                with archive.m_context.raw_file(cif) as file:
                    try:
                        ase_atoms = read(file.name)
                    except Exception as e:
                        logger.error(
                            f'Unable to read cif: {cif}. Encounter the error: {e}'
                        )
                        continue
                    # populate elemets from a set of all the elemsts in ase_atoms
                elements.update(ase_atoms.get_chemical_symbols())
                symmetry = SymmetryNew()
                symmetry_analyzer = SymmetryAnalyzer(ase_atoms, symmetry_tol=1)
                symmetry.bravais_lattice = symmetry_analyzer.get_bravais_lattice()
                symmetry.space_group_number = symmetry_analyzer.get_space_group_number()
                symmetry.space_group_symbol = (
                    symmetry_analyzer.get_space_group_international_short()
                )
                symmetry.crystal_system = symmetry_analyzer.get_crystal_system()
                symmetry.point_group = symmetry_analyzer.get_point_group()

                label = (
                    f'{ase_atoms.get_chemical_formula()}-{symmetry.space_group_number}'
                )
                system = System(
                    atoms=nomad_atoms_from_ase_atoms(ase_atoms),
                    label=label,
                    description='Structure generated by CrystaLLM with workflow_id: '
                    f'"{result.workflow_id}"',
                    structural_type='bulk',
                    dimensionality='3D',
                    symmetry=symmetry,
                )
                add_system_info(system, topologies)
                add_system(system, topologies)

                # `system.system_id` is same as the reference to the topology index
                # `#/results/material/topology/{topology_iter}`
                system_references.append(f'#/{system.system_id}')
            result.generated_structures = system_references

        # reset results
        archive.results = Results(material=Material())
        archive.results.material.elements = list(elements)
        archive.results.material.topology = list(topologies.values())

    def normalize(self, archive, logger=None):
        """
        Normalize the CrystaLLM inference section.
        This method ensures that the section is ready for processing.
        """
        if not self.name:
            self.name = archive.metadata.mainfile.split('.', 1)[0]
        self.m_setdefault('inference_form/inference_settings')

        super().normalize(archive, logger)


m_package.__init_metainfo__()
