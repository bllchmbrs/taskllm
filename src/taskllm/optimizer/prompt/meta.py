import asyncio
import hashlib
import random
from abc import abstractmethod
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Type,
    TypeVar,
    cast,
)

import jinja2
from loguru import logger
from pydantic import BaseModel

from ...ai import (
    DEFAULT_LLM_CONFIG,
    LLMConfig,
    simple_llm_call,
    struct_llm_call,
)
from .ai import (
    generate_context_prompt_part,
    generate_instructions_prompt_part,
    generate_role_part_role,
    generate_simple_prompt,
    vary_content,
)

T = TypeVar("T", bound=BaseModel | bool | str)


class ModelsEnum(str, Enum):
    CLAUDE_3_HAIKU = "anthropic/claude-3-haiku-20240307"
    GPT_4_1_NANO = "openai/gpt-4.1-nano-2025-04-14"
    GPT_4_1_MINI = "openai/gpt-4.1-mini-2025-04-14"
    GEMMA_2_9B_INST = "groq/gemma2-9b-it"
    LLAMA_3_8B = "groq/llama3-8b-8192"


class InstructionsLocationEnum(Enum):
    START = "start"
    END = "end"
    BOTH = "both"


class PromptMode(Enum):
    SIMPLE = "simple"
    ADVANCED = "advanced"


def generate_random_modifier() -> str:
    possible_variations = [
        "Make the prompt you write as simple as possible",
        "make the prompt you write as short as possible",
        "Make the prompt you write a single sentence",
        "Make the prompt you write a single paragraph",
        "make the prompt you write thorough and detailed",
        "Use plain language in the prompt you write",
        "Use technical language in the prompt you write",
    ]
    return random.choice(possible_variations)


class MetaPromptSpecBase(BaseModel):
    input_user_task_goal: str
    input_variable_keys: List[str]
    input_expected_output_type: Type[BaseModel] | bool | str
    mode: PromptMode
    model: ModelsEnum
    prompt_execution_model: ModelsEnum = ModelsEnum.GPT_4_1_MINI

    @classmethod
    @abstractmethod
    async def generate_spec(
        cls,
        guidance: str,
        keys: List[str],
        expected_output_type: Type[BaseModel] | bool | str,
    ) -> Optional["MetaPromptSpecBase"]:
        pass

    def __hash__(self) -> int:
        return int(hashlib.sha256(self.model_dump_json().encode()).hexdigest(), 16)

    @abstractmethod
    async def variation_types(self) -> Set[str]:
        pass

    async def variation_weights(self) -> List[float]:
        return [
            1 / len(await self.variation_types()) for _ in await self.variation_types()
        ]

    @abstractmethod
    async def vary(self, *args: Any, **kwargs: Any) -> Optional["MetaPromptSpecBase"]:
        pass

    @abstractmethod
    def get_content(self) -> str:
        pass


MetaPromptSpecT = TypeVar("MetaPromptSpecT", bound=MetaPromptSpecBase)


class SimpleMetaPromptSpec(MetaPromptSpecBase):
    instructions_and_context: str
    mode: PromptMode = PromptMode.SIMPLE

    async def variation_types(self) -> Set[str]:
        return set(["instructions_and_context"])

    @classmethod
    async def generate_spec(
        cls,
        guidance: str,
        keys: List[str],
        expected_output_type: Type[BaseModel] | bool | str,
    ) -> Optional["SimpleMetaPromptSpec"]:
        """Generates a simpler prompt content with minimal structure."""
        guidance = guidance.strip()
        logger.info(f"Generating simple prompt content for: {guidance[:100]}...")

        simple_prompt = await generate_simple_prompt(
            guidance, keys, expected_output_type
        )
        if simple_prompt is None:
            return None

        # Dummy model selection for now
        model = random.choice(list(ModelsEnum))

        return SimpleMetaPromptSpec(
            input_user_task_goal=guidance,
            input_variable_keys=keys,
            input_expected_output_type=expected_output_type,
            instructions_and_context=simple_prompt,
            model=model,
        )

    async def vary(
        self, variation_type: Set[Literal["instructions_and_context"]]
    ) -> Optional["SimpleMetaPromptSpec"]:
        varied_instructions_and_context = await vary_content(
            self.instructions_and_context, generate_random_modifier()
        )
        varied_model = random.choice(list(ModelsEnum))
        coin_flip = random.random()
        if coin_flip < 0.5:
            varied_model = self.model
        if varied_instructions_and_context is None:
            return None
        return SimpleMetaPromptSpec(
            input_user_task_goal=self.input_user_task_goal,
            input_variable_keys=self.input_variable_keys,
            input_expected_output_type=self.input_expected_output_type,
            instructions_and_context=varied_instructions_and_context,
            model=varied_model,
        )

    def get_content(self) -> str:
        return self.instructions_and_context


class AdvancedMetaPromptSpec(MetaPromptSpecBase):
    role: str
    instructions: str
    context: str
    add_cot: bool
    instructions_location: InstructionsLocationEnum
    include_output_info: bool
    mode: PromptMode = PromptMode.ADVANCED

    async def variation_types(self) -> Set[str]:
        return set(["role", "instructions", "context", "structure"])

    async def variation_weights(self) -> List[float]:
        return [0.3, 0.3, 0.3, 0.1]

    def debug_prompt_string(self) -> str:
        return "\n\n".join([self.role, self.instructions, self.context])

    async def vary_role(self) -> Optional["AdvancedMetaPromptSpec"]:
        instructions = generate_random_modifier()
        varied_role = await vary_content(self.role, instructions)
        if varied_role is None:
            return None
        new_spec = AdvancedMetaPromptSpec(
            input_user_task_goal=self.input_user_task_goal,
            input_variable_keys=self.input_variable_keys,
            input_expected_output_type=self.input_expected_output_type,
            role=varied_role,
            instructions=self.instructions,
            context=self.context,
            add_cot=self.add_cot,
            instructions_location=self.instructions_location,
            include_output_info=self.include_output_info,
            model=self.model,
        )
        return new_spec

    async def vary_instructions(self) -> Optional["AdvancedMetaPromptSpec"]:
        instructions = generate_random_modifier()
        varied_instructions = await vary_content(self.instructions, instructions)
        if varied_instructions is None:
            return None
        new_spec = AdvancedMetaPromptSpec(
            input_user_task_goal=self.input_user_task_goal,
            input_variable_keys=self.input_variable_keys,
            input_expected_output_type=self.input_expected_output_type,
            role=self.role,
            instructions=varied_instructions,
            context=self.context,
            add_cot=self.add_cot,
            instructions_location=self.instructions_location,
            include_output_info=self.include_output_info,
            model=self.model,
        )
        return new_spec

    async def vary_context(self) -> Optional["AdvancedMetaPromptSpec"]:
        instructions = generate_random_modifier()
        varied_context = await vary_content(self.context, instructions)
        if varied_context is None:
            return None
        new_spec = AdvancedMetaPromptSpec(
            input_user_task_goal=self.input_user_task_goal,
            input_variable_keys=self.input_variable_keys,
            input_expected_output_type=self.input_expected_output_type,
            role=self.role,
            instructions=self.instructions,
            context=varied_context,
            add_cot=self.add_cot,
            instructions_location=self.instructions_location,
            include_output_info=self.include_output_info,
            model=self.model,
        )
        return new_spec

    async def vary(
        self,
        variation_type: Set[Literal["role", "instructions", "context", "structure"]],
    ) -> "AdvancedMetaPromptSpec":
        """Returns a random variation of the PromptStructureSpecification.
        Chooses which parts of the prompt to vary based on the `variation_type` set.
        """
        new_spec = self.model_copy()  # Start with a copy of the current spec

        if "role" in variation_type:
            varied_role = await vary_content(self.role, generate_random_modifier())
            new_spec.role = varied_role if varied_role else self.role

        if "instructions" in variation_type:
            varied_instructions = await vary_content(
                self.instructions, generate_random_modifier()
            )
            new_spec.instructions = (
                varied_instructions if varied_instructions else self.instructions
            )

        if "context" in variation_type:
            varied_context = await vary_content(
                self.context, generate_random_modifier()
            )
            new_spec.context = varied_context if varied_context else self.context

        if "structure" in variation_type:
            new_spec.add_cot = random.choice([True, False])
            new_spec.instructions_location = random.choice(
                list(InstructionsLocationEnum)
            )
            new_spec.include_output_info = random.choice([True, False])

        return new_spec

    @classmethod
    async def generate_spec(
        cls,
        guidance: str,
        keys: List[str],
        expected_output_type: Type[BaseModel] | bool | str,
    ) -> Optional["AdvancedMetaPromptSpec"]:
        logger.info(f"Generating prompt content for: {guidance}")
        role, instructions, context = await asyncio.gather(
            generate_role_part_role(guidance, keys, expected_output_type),
            generate_instructions_prompt_part(guidance, keys, expected_output_type),
            generate_context_prompt_part(guidance, keys, expected_output_type),
        )
        if role is None or instructions is None or context is None:
            return None

        # Dummy settings for now
        add_cot = random.choice([True, False])
        instructions_location = random.choice(list(InstructionsLocationEnum))
        include_output_info = random.choice([True, False])
        model = random.choice(list(ModelsEnum))

        new_spec = AdvancedMetaPromptSpec(
            input_user_task_goal=guidance,
            input_variable_keys=keys,
            input_expected_output_type=expected_output_type,
            role=role,
            instructions=instructions,
            context=context,
            add_cot=add_cot,
            instructions_location=instructions_location,
            include_output_info=include_output_info,
            model=model,
        )
        return new_spec

    def get_content(self) -> str:
        """Return a string representation of the prompt."""
        prompt_as_list = [
            self.role,
            self.context,
        ]
        output_info = ""
        if self.include_output_info:
            if self.input_expected_output_type is bool:
                output_info = "The output should be a boolean value (true or false)."
            elif self.input_expected_output_type is str:
                output_info = "The output should be a string."
            elif isinstance(self.input_expected_output_type, type) and issubclass(
                self.input_expected_output_type, BaseModel
            ):
                output_info = "The output should be a JSON object."
        if self.instructions_location == InstructionsLocationEnum.START:
            prompt_as_list.insert(0, self.instructions)
        elif self.instructions_location == InstructionsLocationEnum.END:
            prompt_as_list.append(self.instructions)
        elif self.instructions_location == InstructionsLocationEnum.BOTH:
            prompt_as_list.insert(0, self.instructions)
            prompt_as_list.append(self.instructions)
        prompt_as_list.append(output_info)
        return "\n\n".join(prompt_as_list)


class MetaPrompt(BaseModel, Generic[MetaPromptSpecT]):
    spec: MetaPromptSpecT
    config: LLMConfig
    expected_output_type: Type[BaseModel] | bool | str

    def get_user_message_content(self) -> str:
        return self.spec.get_content()

    def get_prompt_template(self) -> jinja2.Template:
        return jinja2.Template(self.get_user_message_content())

    def render(self, variables: Dict[str, Any]) -> str:
        return self.get_prompt_template().render(**variables)

    def generate_messages(self, variables: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = self.render(variables)
        return [{"role": "user", "content": prompt}]

    async def execute(self, variables: Dict[str, Any], config: LLMConfig) -> T | None:
        logger.trace("Executing prompt")
        if self.expected_output_type is bool:
            logger.trace("Executing simple prompt")
            res = await simple_llm_call(
                self.generate_messages(variables),
                config,
            )
            if res.lower().startswith("true"):
                return cast(T, True)
            elif res.lower().startswith("false"):
                return cast(T, False)
            else:
                logger.error(f"Invalid boolean response: {res}")
                return None
        elif self.expected_output_type is str:
            return cast(
                T,
                await simple_llm_call(
                    self.generate_messages(variables),
                    config,
                ),
            )
        elif isinstance(self.expected_output_type, type) and issubclass(
            self.expected_output_type, BaseModel
        ):
            logger.trace("Executing struct prompt")
            return cast(
                T,
                await struct_llm_call(
                    self.generate_messages(variables),
                    config,
                    self.expected_output_type,
                ),
            )
        else:
            raise ValueError(
                f"Invalid expected output type: {self.expected_output_type}"
            )


def generate_all_prompt_structures(
    mode: PromptMode = PromptMode.ADVANCED,
) -> Iterable[MetaPromptSpecBase]:
    models = list(ModelsEnum)

    if mode == PromptMode.SIMPLE:
        # Simple mode uses fixed settings
        for selected_model in models:
            yield SimpleMetaPromptSpec(
                input_user_task_goal="",  # Dummy values, will be overwritten
                input_variable_keys=[],
                input_expected_output_type="str",
                model=selected_model,
                instructions_and_context="",
            )
    else:  # ADVANCED mode
        add_cot = [True, False]
        instructions_location = list(InstructionsLocationEnum)
        include_output_info = [True, False]

        for selected_model in models:
            for selected_add_cot in add_cot:
                for selected_instructions_location in instructions_location:
                    for selected_include_output_info in include_output_info:
                        yield AdvancedMetaPromptSpec(
                            input_user_task_goal="",  # Dummy values, will be overwritten
                            input_variable_keys=[],
                            input_expected_output_type="str",
                            model=selected_model,
                            role="",
                            instructions="",
                            context="",
                            add_cot=selected_add_cot,
                            instructions_location=selected_instructions_location,
                            include_output_info=selected_include_output_info,
                        )


async def generate_prompts(
    guidance: str,
    keys: List[str],
    expected_output_type: Type[BaseModel] | bool | str,
    num_prompts: int,
    mode: PromptMode = PromptMode.ADVANCED,
) -> List[MetaPrompt[MetaPromptSpecBase]]:
    # Choose the appropriate spec generator based on mode
    spec_generator: Callable[
        [str, List[str], Type[BaseModel] | bool | str],
        Awaitable[Optional[MetaPromptSpecBase]],
    ]

    if mode == PromptMode.SIMPLE:
        spec_generator = SimpleMetaPromptSpec.generate_spec
    else:
        spec_generator = AdvancedMetaPromptSpec.generate_spec

    specs = await asyncio.gather(
        *[
            spec_generator(
                guidance + "\n\n" + generate_random_modifier(),
                keys,
                expected_output_type,
            )
            for _ in range(num_prompts)
        ]
    )

    return [
        MetaPrompt(
            spec=spec,
            expected_output_type=expected_output_type,
            config=DEFAULT_LLM_CONFIG,
        )
        for spec in specs
        if spec is not None
    ]
