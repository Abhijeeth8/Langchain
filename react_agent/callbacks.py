from typing import Any

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult


class ReactAgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
            self,
            serialized: dict[str, Any],
            prompts: list[str],
            **kwargs: Any,
    ) -> Any:
        print("---------------------")
        print(f"Input to the llm is \n{prompts[0]}")
        print("---------------------")

    def on_llm_end(
            self,
            response: LLMResult,
            **kwargs: Any,
    ) -> Any:
        print("---------------------")
        print(f"Response from the llm is \n{response.generations[0][0].text}")
        print("---------------------")
