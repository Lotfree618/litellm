from litellm.utils import get_model_info


def extract_provider_cost_per_request(
    *,
    model: str,
    custom_llm_provider: str | None = None,
) -> tuple[float, float]:
    model_info = get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    return float(model_info.get("input_cost_per_query") or 0.0), 0.0
