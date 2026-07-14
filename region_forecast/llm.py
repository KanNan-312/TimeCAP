import os
import random
import time


class LLMClient:
    """
    Thin wrapper around an OpenAI-compatible chat endpoint, pointed at
    OpenRouter by default. Supports a --dry-run mode that fabricates
    deterministic responses so the rest of the pipeline (splitting,
    prompting, parsing, metrics, resumability) can be exercised without
    spending API calls or requiring an API key.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._client = None
        if not cfg.dry_run:
            api_key = os.environ.get(cfg.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Set the {cfg.api_key_env} environment variable (or pass --dry-run to test "
                    f"the pipeline without making API calls)."
                )
            from openai import OpenAI
            self._client = OpenAI(base_url=cfg.api_base, api_key=api_key)

    def chat(self, system_prompt, user_prompt, *, expect_numeric=False, n_values=None, max_retries=5):
        if self.cfg.dry_run:
            return self._mock_response(expect_numeric=expect_numeric, n_values=n_values)

        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt},
                    ],
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                )
                return resp.choices[0].message.content
            except Exception as e:  # noqa: BLE001 - broad on purpose, retried below
                last_err = e
                time.sleep(min(60.0, 2 ** attempt) + random.random())
        raise RuntimeError(f'LLM call failed after {max_retries} attempts: {last_err}')

    @staticmethod
    def _mock_response(expect_numeric, n_values):
        if expect_numeric and n_values:
            base = 350000.0
            return '|'.join(f'{base + 1000 * i:.2f}' for i in range(n_values))
        return (
            'Mock summary: home prices in this region have shown a moderate trend over the review '
            'period, with inventory and demand indicators pointing to a broadly stable market.'
        )
