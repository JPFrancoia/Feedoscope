import asyncio

from custom_logging import init_logging
from feedoscope import config


async def main() -> None:
    pass


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
