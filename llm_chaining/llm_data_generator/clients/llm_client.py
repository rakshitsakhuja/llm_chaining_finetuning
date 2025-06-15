from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from llm_data_generator.config.config import LLMConfig

class LLMClient:
    """Client for interacting with different LLM APIs"""
    def __init__(self, config: LLMConfig):
        self.config = config
        self.openai_client = AsyncOpenAI(api_key=config.openai_api_key)
        self.anthropic_client = AsyncAnthropic(api_key=config.anthropic_api_key)

    async def call_openai(self, prompt: str, system_prompt: str = "") -> str:
        """Make an async call to OpenAI's API"""
        response = await self.openai_client.chat.completions.create(
            model=self.config.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.temperature
        )
        return response.choices[0].message.content

    async def call_claude(self, prompt: str, system_prompt: str = "") -> str:
        """Make an async call to Anthropic's API"""
        response = await self.anthropic_client.messages.create(
            model=self.config.anthropic_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text 