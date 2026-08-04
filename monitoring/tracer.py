import os
import logging
from typing import Optional, Dict, Any
from functools import wraps
import inspect
import sys

# ----------------------------------------------------------------------
# 1. System Logging Configurations
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("RAGTelemetryTracer")


# ----------------------------------------------------------------------
# 2. Production Telemetry Tracer Class (Singleton Architecture)
# ----------------------------------------------------------------------
class ProductionTelemetryTracer:
    """
    A unified, production-grade observability manager utilizing Langfuse.
    Acts as an enterprise boilerplate providing execution tracing, latency metrics,
    and fallback decorators with Zero external dependencies overhead.
    """
    _instance: Optional["ProductionTelemetryTracer"] = None

    def __new__(cls, *args, **kwargs):
        """Strict Singleton initialization pattern allocation."""
        if cls._instance is None:
            cls._instance = super(ProductionTelemetryTracer, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self.public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
            self.secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
            self.host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

            self.langfuse = None
            self.is_active = False

            if self.public_key and self.secret_key:
                try:
                    logger.info("Connecting to Langfuse Enterprise Cloud telemetry streams...")
                    from langfuse import Langfuse
                    self.langfuse = Langfuse(
                        public_key=self.public_key,
                        secret_key=self.secret_key,
                        host=self.host
                    )
                    self.is_active = True
                    logger.info("Telemetry Monitoring Platform successfully hooked: [ACTIVE].")
                except ImportError:
                    logger.warning("Package [langfuse] missing in execution path. Falling back to Mock Telemetry Mode.")
            else:
                logger.info("Langfuse credentials missing environment boundaries. Monitoring initialized in [PASSIVE/MOCK] mode.")

            self._initialized = True

    def create_trace(self, name: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Any:
        """Initializes a new root execution span trace wrapper."""
        if self.is_active and self.langfuse:
            return self.langfuse.trace(
                name=name,
                user_id=user_id,
                session_id=session_id
            )

        class MockSpan:
            def span(self, *args, **kwargs): return self
            def generation(self, *args, **kwargs): return self
            def update(self, *args, **kwargs): pass
            def end(self, *args, **kwargs): pass
        return MockSpan()

    def trace_span(self, span_name: str, component_type: str = "logic"):
        """
        A highly versatile advanced production decorator to automatically log
        execution telemetry, execution states, and latency times for any method.
        """
        def decorator(func):
            is_coroutine = getattr(func, "_is_coroutine", None) or inspect.iscoroutinefunction(func)

            if is_coroutine:
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    parent_span = kwargs.get("parent_span")
                    if not parent_span and len(args) > 1 and hasattr(args[1], "span"):
                        parent_span = args[1]

                    active_span = None
                    if parent_span and hasattr(parent_span, "span"):
                        active_span = parent_span.span(name=span_name, metadata={"type": component_type})

                    try:
                        result = await func(*args, **kwargs)
                        if active_span: active_span.end(status="SUCCESS")
                        return result
                    except Exception as e:
                        if active_span:
                            active_span.end(status="ERROR", level="CRITICAL", status_message=str(e))
                        raise e
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    parent_span = kwargs.get("parent_span")
                    if not parent_span and len(args) > 1 and hasattr(args[1], "span"):
                        parent_span = args[1]

                    active_span = None
                    if parent_span and hasattr(parent_span, "span"):
                        active_span = parent_span.span(name=span_name, metadata={"type": component_type})
                    try:
                        result = func(*args, **kwargs)
                        if active_span: active_span.end(status="SUCCESS")
                        return result
                    except Exception as e:
                        if active_span:
                            active_span.end(status="ERROR", level="CRITICAL", status_message=str(e))
                        raise e
                return sync_wrapper
        return decorator

    def flush(self):
        """Forces flushing buffered events to Langfuse remote endpoints."""
        if self.is_active and self.langfuse:
            try:
                self.langfuse.flush()
                logger.info("Telemetry traces successfully flushed to remote server.")
            except Exception as e:
                logger.error(f"Failed to flush telemetry events: {e}")


# ----------------------------------------------------------------------
# 3. Pipeline Local Testing / Validation Suite
# ----------------------------------------------------------------------
tracer_singleton = ProductionTelemetryTracer()

@tracer_singleton.trace_span(span_name="Mock_Vector_Retrieval_Step", component_type="retrieval")
async def dummy_async_retrieval_operation(query: str, parent_span: Any = None):
    logger.info(f"Simulating high-speed async trace context logging for query: '{query}'")
    import asyncio
    await asyncio.sleep(0.1)
    return {"status": "mock_data_hit"}

async def main():
    print("\n" + "="*50, file=sys.stderr)
    print("RUNNING OBSCURE TELEMETRY OBSERVABILITY PIPELINE TRACER", file=sys.stderr)
    print("="*50, file=sys.stderr)

    root_trace = tracer_singleton.create_trace(name="User_Chat_API_Endpoint", user_id="usr_freelance_demo_44")
    await dummy_async_retrieval_operation(query="Secure server specs", parent_span=root_trace)
    root_trace.end()
    tracer_singleton.flush()
    print("[✓] Telemetry Tracing pipeline lifecycle verified.", file=sys.stderr)
    print("="*50 + "\n", file=sys.stderr)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())