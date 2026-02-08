from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
import messages
from autogen_core import TRACE_LOGGER_NAME
import importlib
import logging
from autogen_core import AgentId
from dotenv import load_dotenv
import re
import ast

load_dotenv(override=True)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(TRACE_LOGGER_NAME)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

MAX_GENERATION_ATTEMPTS = 3


def clean_generated_code(code: str) -> str:
    # Remove fenced code blocks completely
    code = re.sub(r"```.*?```", "", code, flags=re.DOTALL)

    # Remove common LLM preambles
    code = re.sub(
        r"^(Here is.*?\n|Sure.*?\n|Below is.*?\n)",
        "",
        code,
        flags=re.IGNORECASE,
    )

    code = code.strip()

    # Hard syntax validation
    ast.parse(code)
    return code


class Creator(RoutedAgent):
    system_message = """
    You are an Agent that is able to create new AI Agents.
    You receive a template in the form of Python code that creates an Agent using Autogen Core and Autogen Agentchat.
    You should use this template to create a new Agent with a unique system message that is different from the template,
    and reflects their unique characteristics, interests and goals.
    You can choose to keep their overall goal the same, or change it.
    You can choose to take this Agent in a completely different direction.
    The only requirement is that the class must be named Agent,
    and it must inherit from RoutedAgent and have an __init__ method that takes a name parameter.
    Do not use OpenAIChatCompletionClient anywhere.
    Also avoid environmental interests - try to mix up the business verticals so that every agent is different.
    Respond only with valid Python code. No explanations. No markdown.
    """

    def __init__(self, name) -> None:
        super().__init__(name)

        # ↓↓↓ LOWER TEMPERATURE = STABLE CODEGEN ↓↓↓
        model_client = OllamaChatCompletionClient(
            model="llama3:latest",
            temperature=1.0,
        )

        self._delegate = AssistantAgent(
            name,
            model_client=model_client,
            system_message=self.system_message,
        )

    def get_user_prompt(self):
        prompt = (
            "Generate a new Agent based strictly on this template.\n"
            "Respond with VALID PYTHON CODE ONLY.\n"
            "Be creative about taking the agent in a new direction, but don't change method signatures.\n"
            "No explanations. No markdown. No comments outside code.\n\n"
            "Rules:\n"
            "- Use OllamaChatCompletionClient(model='llama3:latest') only\n"
            "- Class must be named Agent and inherit RoutedAgent\n"
            "- Must implement __init__(self, name) and handle_message\n"
            "- No syntax errors\n"
            "- Avoid environmental themes; pick new business verticals\n\n"
            "Template:\n\n"
        )
        with open("agent.py", "r", encoding="utf-8") as f:
            template = f.read()
        return prompt + template

    @message_handler
    async def handle_my_message_type(
        self, message: messages.Message, ctx: MessageContext
    ) -> messages.Message:
        filename = message.content
        agent_name = filename.split(".")[0]

        last_error = None

        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            logger.debug(f"Generating {agent_name}, attempt {attempt}")

            response = await self._delegate.on_messages(
                [TextMessage(content=self.get_user_prompt(), source="user")],
                ctx.cancellation_token,
            )

            try:
                cleaned_code = clean_generated_code(response.chat_message.content)
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt} failed for {agent_name}: {e}")
        else:
            logger.error(f"All attempts failed for {agent_name}: {last_error}")
            return messages.Message(
                content=f"Failed to generate valid agent: {agent_name}"
            )

        # Write validated code
        with open(filename, "w", encoding="utf-8") as f:
            f.write(cleaned_code)

        print(
            f"** Creator has created python code for agent {agent_name} - about to register with Runtime"
        )

        # Import & register
        try:
            module = importlib.import_module(agent_name)
            await module.Agent.register(
                self.runtime,
                agent_name,
                lambda: module.Agent(agent_name),
            )
            logger.info(f"** Agent {agent_name} is live **")
        except Exception as e:
            logger.error(f"Failed to register agent {agent_name}: {e}")
            return messages.Message(content=f"Failed to register agent: {agent_name}")

        # Kickstart agent
        result = await self.send_message(
            messages.Message(content="Give me an idea"),
            AgentId(agent_name, "default"),
        )
        return messages.Message(content=result.content)
