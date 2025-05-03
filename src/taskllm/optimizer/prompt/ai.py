from textwrap import dedent
from typing import List, Type

from loguru import logger
from pydantic import BaseModel, Field

from ...ai import DEFAULT_LLM_CONFIG, struct_llm_call
from ...utils import get_cache

cache = get_cache("prompt_generation")


class RoleOutput(BaseModel):
    role: str


class InstructionsOutput(BaseModel):
    instructions: str


class ContextOutput(BaseModel):
    context: str


class VaryContentOutput(BaseModel):
    content: str


class PromptOutput(BaseModel):
    task_prompt: str = Field(
        ...,
        description="The task level prompt. Use the guidance provided to create a task level prompt.",
    )


async def generate_role_part_role(
    guidance: str, keys: List[str], expected_output_type: Type[BaseModel] | bool | str
) -> str | None:
    key_list_string = "\n".join([f"- {key}" for key in keys])
    output_info = ""
    if isinstance(expected_output_type, bool):
        output_info = "Provide instructions that the model should follow to determine if the output should be true or false."
    system_prompt = dedent(
        """You are a specialized role architect designed to create precise professional personas for language models. Your task is to craft a concise, highly specific role description that empowers a language model to perform optimally on user-defined tasks.
When presented with a task and variables, you will analyze what expertise would be ideal for that particular challenge and define a corresponding role that includes:

The primary professional identity (e.g., "You are a financial forecasting specialist with expertise in emerging markets")
Relevant qualifications or background that establish authority
Key mindsets or approaches that would enhance performance on the task

Your role descriptions should be tailored to the specific context rather than generic. For example, instead of "You are an expert programmer," create "You are a security-focused Python developer specializing in financial API integration with 10+ years of experience in PCI compliance environments."
This role description will be inserted into a larger system prompt framework alongside specific instructions, so focus exclusively on defining the ideal persona without including task instructions.
The final implementation will follow this workflow:

Generate the specialized role description (your output)
Combine with task-specific instructions in a separate prompt
Merge these components into a complete system prompt
Apply user variables to customize the final prompt
Deploy to the language model for response generation

Keep your output concise but impactful, focusing on the credentials, perspective, and expertise that would most benefit the specific task context.
"""
        + "\n"
        + output_info
    )
    user_prompt = dedent(f"""
<task>
{guidance}
</task>

<variables>
{key_list_string}
</variables>

Please ensure that the role you define is highly specific and relevant to the task and keys provided. The role should reflect expertise that is directly applicable to the task.
""")

    try:
        response = await struct_llm_call(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            config=DEFAULT_LLM_CONFIG,
            response_model=RoleOutput,
            use_cache=False,
        )
        return response.role
    except Exception as e:
        logger.error("Error generating role prompt", error=e)
        return None


# @cache.memoize(expire=60)
async def generate_instructions_prompt_part(
    guidance: str, keys: List[str], expected_output_type: Type[BaseModel] | bool | str
) -> str | None:
    key_list_string = "\n".join([f"- {key}" for key in keys])
    output_info = ""
    if isinstance(expected_output_type, bool):
        output_info = "Provide instructions that the model should follow to determine if the output should be true or false."

    system_prompt = dedent(
        """You are an instruction architect specializing in crafting precise, actionable task directives for language models. Your expertise lies in converting complex requirements into clear, structured instructions that maximize performance.

When presented with a task and its variables, you will generate comprehensive instructions that:

Outline the exact steps, methodology, or framework the language model should follow

Your instructions must be:

- Specific rather than generic
- Process-oriented with explicit guidance on how to approach the task
- Free of unnecessary explanation about what the instructions are for

This focused instruction set will be combined with a role prompt in a larger system framework, so concentrate exclusively on the step-by-step guidance without redefining the role or explaining meta-processes.

Ensure your instructions address the unique requirements of the specific task rather than providing generic guidance that could apply to any situation.
"""
        + "\n"
        + output_info
    )

    user_prompt = dedent(f"""
<task>
{guidance}
</task>

<variables>
{key_list_string}
</variables>

Please ensure that the instructions you provide are highly specific and relevant to the task and keys provided. The instructions should reflect expertise that is directly applicable to the task. Do not just make this a formula.
""")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        response = await struct_llm_call(
            messages=messages,
            config=DEFAULT_LLM_CONFIG,
            response_model=InstructionsOutput,
            use_cache=False,
        )
        return response.instructions
    except Exception as e:
        logger.error("Error generating instructions prompt", error=e)
        return None


# @cache.memoize(expire=60)
async def generate_context_prompt_part(
    guidance: str, keys: List[str], expected_output_type: Type[BaseModel] | bool | str
) -> str | None:
    key_list_string = "\n".join(["{{{{" + key + "}}}}" for key in keys])
    output_info = ""
    if isinstance(expected_output_type, bool):
        output_info = "Your output should be a boolean value (true or false)."
    system_prompt = dedent("""You are a template formatting specialist who excels at structuring variable inputs using Jinja2 syntax. Your expertise is in creating well-organized templates that properly incorporate variables for optimal readability and processing.
When presented with a task goal and a list of variables, you will:
Create a structured template that properly incorporates all input variables using the correct Jinja2 syntax (e.g., {{ variable_name }}). The user will provide you with a list of required variables. Your template must:

Format each variable using proper double curly braces: {{ variable_name }}
Structure the template based on the expected variable content and length:

- Short variables: Incorporate inline within text
- Medium variables: Place within appropriate semantic tags
- Long-form content: Use dedicated container tags with clear labels

Your template formatting must:

- Ensure consistent syntax across all variable references
- Maintain proper spacing and indentation for readability
- Use semantic XML tags when appropriate to organize content logically
- Avoid nested Jinja2 expressions at all costs
- Include all required variables from the provided list, with no omissions and no extra variables

Focus exclusively on creating the structural template with proper Jinja2 variable syntax based on the task goal. This template will be combined with role and task instructions to create a complete prompt.
Your output should be ready for direct implementation without requiring further formatting adjustments. Do not include context about the output format in your response.
<context_example_format_1>
Here's the context on the customer: (variables where: customer_name, account_id, subscription_level, renewal_date, payment_status)

Customer name: {{ customer_name }}
Account ID: {{ account_id }}
Subscription tier: {{ subscription_level }}
Renewal date: {{ renewal_date }}
Payment history: {{ payment_status }}
                           

</context_example_format_1>

<context_example_format_2>
Here is the context for what you're evaluating. (variables where: paper_content, author_credentials, previous_research)

<research_paper>
{{ paper_content }}
</research_paper>

<author_information>
{{ author_credentials }}
</author_information>

<related_studies>
{{ previous_research }}
</related_studies>

</context_example_format_2>

""")

    user_prompt = dedent(f"""
<task>
{guidance}
</task>
Here are the variables that you must include in the template.
<required_variables>
{key_list_string}
</required_variables>

{output_info}

Some final tips:
- Remember to format these variables using the correct Jinja2 syntax (e.g., {{{{ variable_name }}}}). DO NOT JUST MAKE THIS A FORMULA.
- DO NOT ADD ANY OTHER VARIABLES TO THE TEMPLATE.
- DO NOT USE ANY OTHER SYNTAX THAN JINJA2.
- DO NOT INCLUDE AN OUTPUT VARIABLE IN THE TEMPLATE.
- Ensure that the instructions you provide are highly specific and relevant to the task and keys provided. The instructions should reflect expertise that is directly applicable to the task.

Now take a deep breath and begin.
""")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        response = await struct_llm_call(
            messages=messages,
            config=DEFAULT_LLM_CONFIG,
            response_model=ContextOutput,
            use_cache=False,
        )
        return response.context
    except Exception as e:
        logger.error("Error generating context prompt", error=e)
        return None


async def generate_simple_prompt(
    guidance: str, keys: List[str], expected_output_type: Type[BaseModel] | bool | str
) -> str | None:
    key_list_string = "\n".join([f"- {key}" for key in keys])
    output_info = ""
    if isinstance(expected_output_type, bool):
        output_info = "Provide instructions that the model should follow to determine if the output should be true or false."
    elif isinstance(expected_output_type, BaseModel):
        json_schema = expected_output_type.model_json_schema()
        output_info = f"Provide simplified instructions that we want the following output structure <structure>\n{json_schema}\n</structure>"

    system_prompt = dedent(
        """You are a prompt engineer specializing in crafting complete, effective prompts for language models. Your expertise lies in combining the necessary elements into a single, coherent prompt that maximizes performance.

When presented with a task and its variables, you will generate a prompt that includes:

- A clear and specific role for the language model
- Actionable instructions that outline the steps to be taken
- Contextual in the form of jinja2 template variables

Your prompt must be:

- Specific rather than generic
- Action-oriented with explicit guidance
- Concise and free of unnecessary information

Ensure your prompt addresses the unique requirements of the specific task and variables provided by the user. When in doubt, keep things simple.
"""
        + "\n"
        + output_info
    )

    user_prompt = dedent(f"""
<task>
{guidance}
</task>

<variables>
{key_list_string}
</variables>

Please generate a complete prompt that includes a role, instructions, and context, tailored to the task and variables provided. Do not just make this a formula.
Some final tips:
- Remember to format these variables using the correct Jinja2 syntax (e.g., {{{{ variable_name }}}}). DO NOT JUST MAKE THIS A FORMULA.
- DO NOT ADD ANY OTHER VARIABLES TO THE TEMPLATE.
- DO NOT USE ANY OTHER SYNTAX THAN JINJA2.
- DO NOT INCLUDE AN OUTPUT VARIABLE IN THE TEMPLATE.
- Ensure that the instructions you provide are highly specific and relevant to the task and keys provided. The instructions should reflect expertise that is directly applicable to the task.

Now take a deep breath and begin.
""")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        response = await struct_llm_call(
            messages=messages,
            config=DEFAULT_LLM_CONFIG,
            response_model=PromptOutput,
            use_cache=False,
        )
        print(response.task_prompt)
        return response.task_prompt
    except Exception as e:
        logger.error("Error generating complete prompt", error=e)
        return None


async def vary_content(content: str, variation_instructions: str) -> str | None:
    user_prompt = dedent(f"""Please provide a variation of the following content:
                         
                         <content>
                         {content}
                         </content>
                         
                         <variation_instructions>
                         {variation_instructions}
                         </variation_instructions>
                         
                         Do not include any other text than the variation. If the content includes jinja2 variables, be sure to include the variables in the variation.
                         
    """)

    try:
        response = await struct_llm_call(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt},
            ],
            config=DEFAULT_LLM_CONFIG,
            response_model=VaryContentOutput,
            use_cache=False,
        )

        return response.content
    except Exception as e:
        logger.error("Error varying content", error=e)
        return None
