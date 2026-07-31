# Architecture

FastAPI controllers call application services, which call EnhancementAgent, tool registry, model adapters, and inference backends. SQLite stores metadata only.
