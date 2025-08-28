import asyncio

from custom_logging import init_logging
from feedoscope import config
from feedoscope import infer_time_sensitivity
from feedoscope import llm_infer


async def main() -> None:
    time_sensitivities = await infer_time_sensitivity.main()
    relevance_scores = await llm_infer.main(write_scores_to_db=False)


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
