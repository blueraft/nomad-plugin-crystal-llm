from abc import ABCMeta, abstractmethod

from nomad.datamodel.data import ArchiveSection, EntryData, EntryDataCategory
from nomad.datamodel.metainfo.annotations import (
    ELNAnnotation,
    ELNComponentEnum,
    SectionProperties,
)
from nomad.metainfo import Category, Quantity, SchemaPackage, Section

m_package = SchemaPackage()


class InferenceCategory(EntryDataCategory):
    """Category for inference workflows."""

    m_def = Category(
        label='Inference Workflows',
        categories=[EntryDataCategory],
    )


class WorkflowSection(ArchiveSection):
    """Abstract section to run inference workflows"""

    trigger_run_inference = Quantity(
        type=bool,
        default=False,
        description='Trigger to run the inference workflow.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.ActionEditQuantity, label='Run Inference'
        ),
    )

    @abstractmethod
    def run_workflow(self, archive, logger=None):
        """Run the inference workflow with the provided archive."""
        raise NotImplementedError('This method should be implemented in subclasses.')

    def normalize(self, archive, logger=None):
        """Normalize the section to ensure it is ready for processing."""
        super().normalize(archive, logger)
        if self.trigger_run_inference:
            self.run_workflow(archive, logger)
            self.trigger_run_inference = False

        # Ensure that the model path and URL are set
        if not self.model_path or not self.model_url:
            raise ValueError('Model path and URL must be specified to run inference.')

        # Additional normalization logic can be added here


class CrystaLLMInferenceSettings(ArchiveSection):
    """Settings for CrystaLLM inference workflows."""

    model_path = Quantity(
        type=str,
        default='models/crystallm_v1_small/ckpt.pt',
        description='Path to the model file.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.StringEditQuantity,
        ),
    )
    model_url = Quantity(
        type=str,
        default='https://zenodo.org/records/10642388/files/crystallm_v1_small.tar.gz',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.StringEditQuantity,
        ),
    )
    num_samples = Quantity(
        type=int,
        default=2,
        description='Number of samples to draw during inference.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    max_new_tokens = Quantity(
        type=int,
        default=3000,
        description='Maximum number of tokens to generate in each sample.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    temperature = Quantity(
        type=float,
        default=0.8,
        description='Controls the randomness of predictions. Lower values make the '
        'model more deterministic, while higher values increase randomness.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
            minValue=0.0,
        ),
    )
    top_k = Quantity(
        type=int,
        default=10,
        description='Retain only the top_k most likely tokens, clamp others to have 0 probability.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    seed = Quantity(
        type=int,
        default=1337,
        description='Random seed for reproducibility.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    dtype = Quantity(
        type=str,
        default='bfloat16',
        description='Data type for the model (e.g., "float32", "bfloat16", "float16").',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.StringEditQuantity,
        ),
    )
    compile = Quantity(
        type=bool,
        default=False,
        description='Whether to compile the model for faster inference.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.BoolEditQuantity,
        ),
    )


class CrystaLLMInferenceResult(ArchiveSection):
    """Result of a CrystaLLM inference workflow."""

    workflow_id = Quantity(
        type=str,
        description='ID of the workflow that generated this result.',
    )
    prompt = Quantity(
        type=str,
        description='Prompt used for the inference.',
    )
    status = Quantity(
        type=str,
        default='pending',
        description='Status of the inference result.',
    )
    cif_file = Quantity(
        type=str,
        default='',
        description='Path to the CIF file generated from the inference result.',
    )
    system = Quantity(
        type=System,
        description='Reference to the system normalized from the CIF file.',
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
    prompts = Quantity(
        type=str,
        description='Prompt to be used for inference.',
        repeat=True,
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity),
    )
    inference_settings = Quantity(
        type=CrystaLLMInferenceSettings,
        description='Settings for the CrystaLLM inference workflow.',
    )
    results = Quantity(
        type=CrystaLLMInferenceResult,
        description='Results of the inference workflow.',
        repeat=True,
    )

    def run_workflow(self, archive, logger=None):
        """
        Run the CrystaLLM inference workflow with the provided archive.
        This method should be implemented to perform the actual inference logic.
        """
        # Placeholder for actual inference logic
        logger.info('Running CrystaLLM inference workflow...')
        # Implement the inference logic here
        pass


m_package.__init_metainfo__()
