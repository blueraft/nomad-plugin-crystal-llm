from ase.io import read
from matid import SymmetryAnalyzer
from nomad.datamodel.data import ArchiveSection, EntryData, EntryDataCategory
from nomad.datamodel.metainfo.annotations import (
    ELNAnnotation,
    ELNComponentEnum,
    SectionProperties,
)
from nomad.datamodel.results import Material, SymmetryNew, System
from nomad.metainfo import Category, MEnum, Quantity, SchemaPackage, Section, SubSection
from nomad.normalizing.common import nomad_atoms_from_ase_atoms
from nomad.normalizing.topology import add_system, add_system_info
from nomad.orchestrator import util as orchestrator_utils
from nomad.orchestrator.shared.constant import TaskQueue

from nomad_plugin_crystal_llm.workflows.shared import InferenceInput

m_package = SchemaPackage()


class InferenceCategory(EntryDataCategory):
    """Category for inference workflows."""

    m_def = Category(
        label='Inference Workflows',
        categories=[EntryDataCategory],
    )


class WorkflowSection(ArchiveSection):
    """Abstract section to run inference workflows"""

    trigger_run_workflow = Quantity(
        type=bool,
        description='Trigger to run the workflow.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.ActionEditQuantity, label='Run Inference'
        ),
    )
    trigger_workflow_status = Quantity(
        type=bool,
        description='Trigger to get the status of the workflow.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.ActionEditQuantity, label='Get Workflow Status'
        ),
    )

    def run_workflow(self, archive, logger=None):
        """Run the workflow with the provided archive."""
        raise NotImplementedError('This method should be implemented in subclasses.')

    def workflow_status(self, archive, logger=None):
        """Get the status of the workflow."""
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

        if self.trigger_workflow_status:
            try:
                self.workflow_status(archive, logger)
            except Exception as e:
                logger.error(f'Error getting workflow status: {e}. ')
            self.trigger_workflow_status = False


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
    prompt = Quantity(
        type=str,
        description='Prompt to be used for inference.',
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


class InferenceSettingsFixed(InferenceSettings):
    """Settings for CrystaLLM inference workflows with non-editable fields."""


class InferenceSettingsForm(InferenceSettings):
    """Settings form for CrystaLLM inference workflows with editable fields."""

    model = InferenceSettings.model.m_copy()
    model.m_annotation = ELNAnnotation(
        component=ELNComponentEnum.EnumEditQuantity,
        default='crystallm_v1_small',
    )
    prompt = InferenceSettings.prompt.m_copy()
    prompt.m_annotation = ELNAnnotation(
        component=ELNComponentEnum.StringEditQuantity,
    )
    num_samples = InferenceSettings.num_samples.m_copy()
    num_samples.m_annotation = ELNAnnotation(
        component=ELNComponentEnum.NumberEditQuantity,
        default=1,
    )
    max_new_tokens = InferenceSettings.max_new_tokens.m_copy()
    max_new_tokens.m_annotation = ELNAnnotation(
        component=ELNComponentEnum.NumberEditQuantity,
        default=3000,
    )
    temperature = InferenceSettings.temperature.m_copy()
    temperature.m_annotation = ELNAnnotation(
        component=ELNComponentEnum.NumberEditQuantity,
        default=0.8,
    )
    top_k = InferenceSettings.top_k.m_copy()
    top_k.m_annotation = ELNAnnotation(
        component=ELNComponentEnum.NumberEditQuantity,
        default=10,
    )
    seed = InferenceSettings.seed.m_copy()
    seed.m_annotation = ELNAnnotation(
        component=ELNComponentEnum.NumberEditQuantity,
        default=1337,
    )
    dtype = InferenceSettings.dtype.m_copy()
    dtype.m_annotation = ELNAnnotation(
        component=ELNComponentEnum.EnumEditQuantity,
        default='bfloat16',
    )
    compile = InferenceSettings.compile.m_copy()
    compile.m_annotation = ELNAnnotation(
        component=ELNComponentEnum.BoolEditQuantity,
        default=False,
    )


class InferenceResult(ArchiveSection):
    """Result of a CrystaLLM inference workflow."""

    workflow_id = Quantity(
        type=str,
        description='ID of the `temporalio` workflow that generated this result.',
    )
    status = Quantity(
        type=str,
        description='Status of the inference result.',
    )
    generated_cif = Quantity(
        type=str,
        description='Path to the CIF generated by the LLM.',
    )
    generated_structure = Quantity(
        type=System,
        description='Reference to the system normalized based on the generated CIF.',
    )
    inference_settings = SubSection(
        section_def=InferenceSettingsFixed,
        description='Settings used for the CrystaLLM inference workflow.',
    )


class CrystaLLMInference(WorkflowSection, EntryData):
    """
    Section for running CrystaLLM inference workflows.
    """

    m_def = Section(
        categories=[InferenceCategory],
        a_eln=ELNAnnotation(
            properties=SectionProperties(
                order=[
                    'name',
                    'description',
                    'prompts',
                    'trigger_run_inference',
                    'inference_settings',
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
        a_eln=ELNAnnotation(component=ELNComponentEnum.RichTextEditQuantity),
    )
    inference_form = SubSection(
        section_def=InferenceSettingsForm,
        description='Settings for the CrystaLLM inference workflow.',
    )
    results = SubSection(
        section_def=InferenceResult,
        description='Results of the inference workflow.',
        repeats=True,
    )

    def run_workflow(self, archive, logger=None):
        """
        Run the CrystaLLM inference workflow with the provided archive.
        Uses the first author's credentials to run the workflow.
        """
        self.results = []
        if not archive.metadata.authors:
            logger.warn(
                'No authors found in the archive metadata. '
                'Cannot run CrystaLLM inference workflow.'
            )
            return
        input_data = InferenceInput(
            user_id=archive.metadata.authors[0].user_id,
            upload_id=archive.metadata.upload_id,
            raw_input='',
            generate_cif=True,
        )
        workflow_name = 'nomad_plugin_crystal_llm.workflows.InferenceWorkflow'
        for prompt in self.prompts:
            input_data.raw_input = prompt
            workflow_id = orchestrator_utils.run_workflow(
                workflow_name=workflow_name, data=input_data, task_queue=TaskQueue.GPU
            )
            self.results.append(
                InferenceResult(
                    workflow_id=workflow_id,
                    prompt=prompt,
                )
            )

    def workflow_status(self, archive, logger=None):
        """
        Get the status of the CrystaLLM inference workflow.
        This method should be implemented to retrieve the status of the workflow.
        """
        for result in self.results:
            status = orchestrator_utils.get_workflow_status(result.workflow_id)
            if status:
                result.status = status.name

    def process_generated_cif(self, archive, logger):
        """
        Process the CIF file in `archive.data.results` and populates
        `archive.results.material.topology`.
        """
        # Read the reference CIF files and convert them into ase atoms
        if not self.results:
            return
        ase_atoms_list = []
        for result in self.results:
            cif_file = result.cif_file
            if not cif_file or not cif_file.endswith('.cif'):
                logger.warn(
                    f'Cannot parse structure file: {cif_file}. '
                    'Should be a "*.cif" file.'
                )
                continue
            with archive.m_context.raw_file(cif_file) as file:
                try:
                    ase_atoms_list.append(read(file.name))
                except RuntimeError:
                    logger.warn(f'Cannot parse cif file: {cif_file}.')

        # Let's save the composition and structure into archive.results.material
        if not archive.results.material:
            archive.results.material = Material()

        # populate elemets from a set aof all the elemsts in ase_atoms
        elements = set()
        for ase_atoms in ase_atoms_list:
            elements.update(ase_atoms.get_chemical_symbols())
        archive.results.material.elements = list(elements)

        # Create a System: this is a NOMAD specific data structure for
        # storing structural and chemical information that is suitable for both
        # experiments and simulations.
        topology = {}
        labels = []
        for ase_atoms in ase_atoms_list:
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
                f'{cif_file.rsplit(".", 1)[0]}-{ase_atoms.get_chemical_formula()}'
                f'-{symmetry.space_group_number}'
            )
            labels.append(label)
            system = System(
                atoms=nomad_atoms_from_ase_atoms(ase_atoms),
                label=label,
                description='Structure generated by CrystaLLM',
                structural_type='bulk',
                dimensionality='3D',
                symmetry=symmetry,
            )
            add_system_info(system, topology)
            add_system(system, topology)

        archive.results.material.topology = list(topology.values())
        topology_m_proxies = dict()
        for i, system in enumerate(archive.results.material.topology):
            topology_m_proxies[system.label] = f'#/results/material/topology/{i}'

        # connect `data.reference_structures[i].system` and
        # `results.material.topology[j]` using the label
        for i, label in enumerate(labels):
            self.results[i].system = topology_m_proxies[label]

    def normalize(self, archive, logger=None):
        """
        Normalize the CrystaLLM inference section.
        This method ensures that the section is ready for processing.
        """
        super().normalize(archive, logger)
        self.process_generated_cif(archive, logger)


m_package.__init_metainfo__()
