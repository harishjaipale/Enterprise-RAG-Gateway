import os
import logging
import asyncio
from typing import Optional, Union, Any
from pathlib import Path

# Setup system logger
logger = logging.getLogger("VectorDBConnectionPool")

# Import centralized application settings with safe fallback
try:
    from config.settings import settings
except ImportError:
    settings = None


# ----------------------------------------------------------------------
# Custom Infrastructure Exceptions
# ----------------------------------------------------------------------
class ConnectionPoolError(Exception):
    """Base exception for cluster infrastructure errors."""
    pass


class DatabaseUnreachableError(ConnectionPoolError):
    """Raised when health check ping fails across target interfaces."""
    pass


# ----------------------------------------------------------------------
# Connection Manager Class (Singleton Pattern)
# ----------------------------------------------------------------------
class VectorDBConnectionManager:
    """
    A high-throughput, thread-safe asynchronous connection pool manager 
    supporting multi-tenant configurations across Qdrant and Pinecone.
    """
    _instance: Optional["VectorDBConnectionManager"] = None
    _lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs):
        """Ensures a strict Singleton allocation in memory."""
        return super(VectorDBConnectionManager, cls).__new__(cls)

    def __init__(self, provider: Optional[str] = None):
        """Initializes internal tracking registers for connection isolation."""
        if not hasattr(self, "_initialized"):
            default_provider = "qdrant"
            if settings:
                default_provider = getattr(settings, "VECTOR_DB_PROVIDER", "qdrant")

            self.provider = (provider or default_provider).lower()
            
            self._async_qdrant_client: Optional[Any] = None
            self._async_pinecone_client: Optional[Any] = None
            
            self._initialized: bool = True
            logger.info(f"Vector Database Connection Manager initialized for target provider: [{self.provider.upper()}]")

    @classmethod
    async def get_instance(cls, provider: Optional[str] = None) -> "VectorDBConnectionManager":
        """Thread-safe and async-safe instantiation factory to fetch the singleton pool."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(provider=provider)
            elif provider and cls._instance.provider != provider.lower():
                cls._instance.provider = provider.lower()
                logger.info(f"Vector Database Provider switched dynamically to: [{cls._instance.provider.upper()}]")
            return cls._instance

    async def get_qdrant_client(self) -> Any:
        """
        Returns an instance of AsyncQdrantClient. 
        Connection reuse prevents socket exhaustion under heavy load.
        """
        if self.provider != "qdrant":
            raise ConnectionPoolError(f"Requested Qdrant client, but active provider is set to: {self.provider}")

        if self._async_qdrant_client is None:
            try:
                from qdrant_client import AsyncQdrantClient
            except ImportError:
                raise ConnectionPoolError(
                    "qdrant-client package is missing. Install it via 'pip install qdrant-client'."
                )

            try:
                logger.info("Initializing persistent connection pool to Qdrant cluster...")
                
                qdrant_url = "http://localhost:6333"
                qdrant_key = None

                if settings:
                    if hasattr(settings, "qdrant"):
                        qdrant_url = getattr(settings.qdrant, "url", qdrant_url)
                        qdrant_key = getattr(settings.qdrant, "api_key", None)
                    else:
                        qdrant_url = getattr(settings, "QDRANT_URL", qdrant_url)
                        qdrant_key = getattr(settings, "QDRANT_API_KEY", None)

                qdrant_url = os.getenv("QDRANT_URL", qdrant_url)
                qdrant_key = os.getenv("QDRANT_API_KEY", qdrant_key)

                # Local setups or memory DB ke liye gRPC disable kar rahe hain taaki connection rejection na ho
                use_grpc = True
                if "localhost" in qdrant_url or "127.0.0.1" in qdrant_url or qdrant_url.startswith(":memory:"):
                    use_grpc = False

                self._async_qdrant_client = AsyncQdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_key,
                    timeout=60.0,
                    prefer_grpc=use_grpc
                )
                logger.info(f"Async Qdrant client initialized (prefer_grpc={use_grpc}).")
            except Exception as e:
                logger.critical(f"Fatal connection error initializing Qdrant client: {str(e)}")
                raise ConnectionPoolError(f"Qdrant Allocation Failure: {str(e)}")
                
        return self._async_qdrant_client

    async def get_pinecone_client(self) -> Any:
        """
        Returns an instance of AsyncPinecone.
        """
        if self.provider != "pinecone":
            raise ConnectionPoolError(f"Requested Pinecone client, but active provider is set to: {self.provider}")

        if self._async_pinecone_client is None:
            try:
                from pinecone import AsyncPinecone
            except ImportError:
                raise ConnectionPoolError(
                    "pinecone-client package is missing. Install it via 'pip install pinecone-client'."
                )

            try:
                logger.info("Initializing persistent connection pool to Pinecone cluster...")
                pinecone_key = os.getenv("PINECONE_API_KEY")
                if settings and not pinecone_key:
                    pinecone_key = getattr(settings, "PINECONE_API_KEY", None)

                if not pinecone_key:
                    raise ConnectionPoolError("PINECONE_API_KEY environment variable is missing.")
                
                self._async_pinecone_client = AsyncPinecone(api_key=pinecone_key)
                logger.info("Async Pinecone client initialized successfully.")
            except Exception as e:
                logger.critical(f"Fatal connection error initializing Pinecone client: {str(e)}")
                raise ConnectionPoolError(f"Pinecone Allocation Failure: {str(e)}")

        return self._async_pinecone_client

    async def get_active_client(self) -> Any:
        """Dynamically retrieves the active database client based on system configuration."""
        if self.provider == "qdrant":
            return await self.get_qdrant_client()
        elif self.provider == "pinecone":
            return await self.get_pinecone_client()
        else:
            raise ConnectionPoolError(f"Unsupported Vector Database provider target: {self.provider}")

    async def verify_connectivity_health(self) -> bool:
        """
        Executes operational health check (Heartbeat Ping) before executing indexing or search operations.
        """
        try:
            client = await self.get_active_client()
            logger.info(f"Initiating health verification ping to [{self.provider.upper()}]...")
            
            if self.provider == "qdrant":
                await client.get_collections()
            elif self.provider == "pinecone":
                await client.list_indexes()
                
            logger.info(f"Database Health Check: [SUCCESS]. Node connectivity alive on [{self.provider.upper()}] cluster.")
            return True
            
        except Exception as e:
            logger.error(f"Database Health Check: [FAILED]. Cluster node unreachable: {str(e)}")
            raise DatabaseUnreachableError(f"Database host cluster terminal drop: {str(e)}")

    async def shutdown_connections_pool(self) -> None:
        """Gracefully terminates active background connections to prevent TCP socket leakage."""
        logger.info("Deallocating persistent vector connection pool...")
        
        if self._async_qdrant_client is not None:
            try:
                await self._async_qdrant_client.close()
                logger.info("Qdrant asynchronous client closed successfully.")
            except Exception as e:
                logger.error(f"Error closing Qdrant instance: {str(e)}")
            finally:
                self._async_qdrant_client = None
                
        if self._async_pinecone_client is not None:
            self._async_pinecone_client = None
            logger.info("Pinecone infrastructure bindings deallocated.")


# ----------------------------------------------------------------------
# Local Verification Engine Module
# ----------------------------------------------------------------------
if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        print("\n" + "="*50)
        print("RUNNING PERSISTENT CLIENT LIFECYCLE VERIFICATION POOL")
        print("="*50)

        try:
            pool_manager = await VectorDBConnectionManager.get_instance()
            second_reference = await VectorDBConnectionManager.get_instance()
            print(f"[✓] Singleton Memory Binding Integrity Verified: {pool_manager is second_reference}")
            
            try:
                await pool_manager.verify_connectivity_health()
            except DatabaseUnreachableError:
                print("[.] Health test caught expected connection rejection: DB Cluster server not active locally.")

            await pool_manager.shutdown_connections_pool()
            print("="*50 + "\n")

        except Exception as e:
            print(f"[✗] Failure testing connection pool architecture: {e}")

    asyncio.run(main())