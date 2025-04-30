from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict

from langchain_core.outputs import LLMResult


class RAGCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> Any:
        print(
            f"-----------Hitting LLM with prompt \n{prompts[0]}\n---------------------"
        )

    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any,
    ) -> Any:
        print(f"-----------Response recieved was \n{response}\n---------------------")
