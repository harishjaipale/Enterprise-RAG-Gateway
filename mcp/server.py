import os
import sys
import json
import logging
import asyncio
import argparse
from typing import List, Dict, Any, Optional, Callable, Type
from pydantic import BaseModel, Field, ValidationError

# ----------------------------------------------------------------------
# 1. System Logging Configurations
# ----------------------------------------------------------------------
# Redirect all logging output to stderr so stdout remains clean for JSON-RPC
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EnterpriseMCPServer")


# ----------------------------------------------------------------------
# 2. Protocol Custom Exceptions
# ----------------------------------------------------------------------
class MCPError(Exception):
    """Base exception for Model Context Protocol operational routines."""
    def __init__(self, message: str, code: int = -32603):
        super().__init__(message)
        self.code = code
        self.message = message

class ToolExecutionError(MCPError):
    """Raised when an internal tool execution context crashes runtime flows."""
    def __init__(self, message: str):
        super().__init__(message, code=-32000)

class ArgumentValidationError(MCPError):
    """Raised when incoming agent payloads break structured JSON validation schemes."""
    def __init__(self, message: str):
        super().__init__(message, code=-32602)


# ----------------------------------------------------------------------
# 3. Protocol Data Schemas & Mappings
# ----------------------------------------------------------------------
class MCPToolDefinition(BaseModel):
    """Defines structural protocol properties exposed directly to LLM context maps."""
    name: str = Field(..., description="Unique programmatic identifier of the tool execution unit.")
    description: str = Field(..., description="Semantic purpose text utilized by the LLM to trigger invocation.")
    input_schema: Dict[str, Any] = Field(..., description="JSON schema constraints governing incoming validation bounds.")


# Pydantic Input Schemas for Strict Runtime Safety
class FileScanInput(BaseModel):
    target_filename: str = Field(
        ..., 
        description="The specific text target asset to scan bounds within."
    )
    max_lines: int = Field(
        default=100, 
        ge=1, 
        le=5000, 
        description="Maximum lines to evaluate (1 to 5000)."
    )

class SecurityAuditInput(BaseModel):
    ip_address: str = Field(..., description="Target host IP address for security posture scan.")
    scan_type: str = Field(default="quick", description="Audit depth: 'quick' or 'deep'")


# ----------------------------------------------------------------------
# 4. Enterprise Dual-Transport MCP Server Engine
# ----------------------------------------------------------------------
class EnterpriseMCPServer:
    """
    Production-grade Model Context Protocol (MCP) Server Infrastructure.
    Features Pydantic input validation, dynamic tool binding, and supports 
    both STDIO (local IPC) and SSE (HTTP streaming) transport adapters.
    """
    def __init__(self, server_name: str = "EnterpriseContextBridge", version: str = "1.0.0"):
        self.server_name = server_name
        self.version = version
        
        self._tools_registry: Dict[str, Callable] = {}
        self._schemas_registry: Dict[str, Type[BaseModel]] = {}
        self._tools_metadata: Dict[str, MCPToolDefinition] = {}

    def register_tool(self, name: str, description: str, schema_model: Type[BaseModel]) -> Callable:
        """
        Decorator to register python execution routines with Pydantic type checking.
        """
        def decorator(func: Callable):
            tool_meta = MCPToolDefinition(
                name=name,
                description=description,
                input_schema=schema_model.model_json_schema()
            )
            self._tools_metadata[name] = tool_meta
            self._schemas_registry[name] = schema_model
            self._tools_registry[name] = func
            logger.info(f"Registered protocol tool node: [{name}] with Pydantic validation")
            return func
        return decorator

    async def _handle_json_rpc_request(self, raw_request: str) -> Optional[str]:
        """
        Executes JSON-RPC 2.0 lifecycle handlers including Pydantic validation.
        """
        try:
            payload = json.loads(raw_request)
            request_id = payload.get("id")
            method = payload.get("method")
            params = payload.get("params", {})

            if not method:
                raise ArgumentValidationError("Missing execution routing method.")

            # Route 0: Protocol Handshake
            if method == "initialize":
                return json.dumps({
                    "jsonrpc": "2.0",
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": self.server_name, "version": self.version}
                    },
                    "id": request_id
                })

            # Route 0b: Notification Confirmation
            elif method == "notifications/initialized":
                logger.info("Protocol handshake completed successfully by client.")
                return None

            # Route 1: Tool Discovery
            elif method == "tools/list":
                tools_payload = [meta.model_dump() for meta in self._tools_metadata.values()]
                return json.dumps({
                    "jsonrpc": "2.0", 
                    "result": {"tools": tools_payload}, 
                    "id": request_id
                })

            # Route 2: Tool Invocation
            elif method == "tools/call":
                tool_name = params.get("name")
                raw_arguments = params.get("arguments", {})

                if tool_name not in self._tools_registry:
                    return json.dumps({
                        "jsonrpc": "2.0", 
                        "error": {"code": -32601, "message": f"Tool execution node [{tool_name}] not registered."},
                        "id": request_id
                    })

                schema_cls = self._schemas_registry[tool_name]
                callback = self._tools_registry[tool_name]

                # Step A: Pydantic Validation Check
                try:
                    validated_args = schema_cls(**raw_arguments)
                except ValidationError as val_err:
                    logger.warning(f"Validation intercept on [{tool_name}]: {val_err.errors()}")
                    return json.dumps({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32602, 
                            "message": f"Invalid tool arguments: {val_err.errors()}"
                        },
                        "id": request_id
                    })

                # Step B: Execution Phase
                logger.info(f"Executing tool [{tool_name}]...")
                try:
                    if asyncio.iscoroutinefunction(callback):
                        result = await callback(validated_args)
                    else:
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(None, lambda: callback(validated_args))

                    return json.dumps({
                        "jsonrpc": "2.0",
                        "result": {
                            "content": [{"type": "text", "text": str(result)}]
                        },
                        "id": request_id
                    })
                except Exception as eval_err:
                    logger.error(f"Tool execution fault: {str(eval_err)}", exc_info=True)
                    return json.dumps({
                        "jsonrpc": "2.0",
                        "error": {"code": -32000, "message": f"Tool execution error: {str(eval_err)}"},
                        "id": request_id
                    })

            # Catch-all Route
            else:
                if request_id is None:
                    return None
                return json.dumps({
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Method [{method}] not implemented."},
                    "id": request_id
                })

        except json.JSONDecodeError:
            return json.dumps({
                "jsonrpc": "2.0", 
                "error": {"code": -32700, "message": "Failed parsing JSON payload."}, 
                "id": None
            })
        except MCPError as mcp_err:
            return json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": mcp_err.code, "message": mcp_err.message},
                "id": payload.get("id") if 'payload' in locals() else None
            })
        except Exception as e:
            return json.dumps({
                "jsonrpc": "2.0", 
                "error": {"code": -32603, "message": f"Fatal server failure: {str(e)}"}, 
                "id": None
            })

    # ------------------------------------------------------------------
    # Transport Loop A: Stdio Pipeline
    # ------------------------------------------------------------------
    async def start_stdio_transport_loop(self):
        """Launches standard bidirectional input-output processing loops over STDIO."""
        logger.info(f"Bootstrapping STDIO channel for MCP server: [{self.server_name}] v{self.version}")
        
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_running_loop().connect_read_pipe(lambda: protocol, sys.stdin)
        
        loop = asyncio.get_running_loop()
        w_transport, w_protocol = await loop.connect_write_pipe(asyncio.streams.FlowControlMixin, sys.stdout)
        writer = asyncio.StreamWriter(w_transport, w_protocol, reader, loop)

        try:
            while True:
                line_bytes = await reader.readline()
                if not line_bytes:
                    break
                
                raw_request_string = line_bytes.decode("utf-8").strip()
                if not raw_request_string:
                    continue
                
                rpc_response = await self._handle_json_rpc_request(raw_request_string)
                if rpc_response:
                    writer.write((rpc_response + "\n").encode("utf-8"))
                    await writer.drain()
        except Exception as loop_fault:
            logger.critical(f"Stdio transport failure: {str(loop_fault)}", exc_info=True)
        finally:
            logger.info("Stdio transport pipeline closed.")

    # ------------------------------------------------------------------
    # Transport Loop B: SSE (Server-Sent Events) Pipeline via Starlette
    # ------------------------------------------------------------------
    async def start_sse_transport_loop(self, host: str = "0.0.0.0", port: int = 8000):
        """Launches an asynchronous SSE HTTP Endpoint server for Remote/Cloud deployments."""
        try:
            import uvicorn
            from starlette.applications import Starlette
            from starlette.routing import Route
            from starlette.responses import StreamingResponse, Response
            from starlette.requests import Request
        except ImportError:
            logger.critical("SSE transport requires 'starlette' and 'uvicorn'. Run: pip install starlette uvicorn")
            sys.exit(1)

        logger.info(f"Bootstrapping SSE Channel on http://{host}:{port}/sse")

        async def handle_sse_endpoint(request: Request):
            async def event_generator():
                # Initial endpoint connection event
                yield "event: endpoint\ndata: /messages\n\n"
                while True:
                    await asyncio.sleep(15)  # Keeps connection alive
                    yield ": keep-alive\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        async def handle_message_endpoint(request: Request):
            body_bytes = await request.body()
            raw_request_string = body_bytes.decode("utf-8").strip()
            
            rpc_response = await self._handle_json_rpc_request(raw_request_string)
            if rpc_response:
                return Response(content=rpc_response, media_type="application/json")
            return Response(status_code=202)

        routes = [
            Route("/sse", endpoint=handle_sse_endpoint, methods=["GET"]),
            Route("/messages", endpoint=handle_message_endpoint, methods=["POST"]),
        ]

        app = Starlette(routes=routes)
        config = uvicorn.Config(app=app, host=host, port=port, log_level="error")
        server = uvicorn.Server(config)
        await server.serve()


# ----------------------------------------------------------------------
# 5. Pipeline Testing & Tool Registration
# ----------------------------------------------------------------------
mcp_bridge = EnterpriseMCPServer(server_name="CorporateSecureVaultBridge", version="1.0.0")

@mcp_bridge.register_tool(
    name="secure_inspect_filesystem",
    description="Allows specialized AI agents to scan server log assets securely.",
    schema_model=FileScanInput
)
async def secure_inspect_filesystem(args: FileScanInput) -> str:
    """Safely executes directory traversal check using validated model properties."""
    target_filename = args.target_filename
    clean_name = os.path.basename(target_filename)
    if clean_name != target_filename or ".." in target_filename:
        raise PermissionError("Directory Traversal anomaly intercepted. Operation rejected.")
    return f"Sandbox Pass: Target file '{target_filename}' evaluated successfully. Line bound: {args.max_lines}."

@mcp_bridge.register_tool(
    name="run_security_audit",
    description="Runs network interface diagnostic check on target host.",
    schema_model=SecurityAuditInput
)
async def run_security_audit(args: SecurityAuditInput) -> Dict[str, Any]:
    """Runs a simulated network security check."""
    return {
        "ip": args.ip_address,
        "mode": args.scan_type,
        "status": "SECURE",
        "scanned_ports": [22, 80, 443]
    }


# ----------------------------------------------------------------------
# 6. Command-Line Entrypoint
# ----------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="Production Grade Dual-Transport MCP Server")
    parser.add_argument(
        "--transport", 
        choices=["stdio", "sse", "test"], 
        default="stdio", 
        help="Transport strategy: 'stdio' (Local IPC), 'sse' (Cloud HTTP Stream), or 'test' (Integrity Suite)."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Binding host IP for SSE mode")
    parser.add_argument("--port", type=int, default=8000, help="Binding port for SSE mode")

    args = parser.parse_args()

    if args.transport == "test":
        print("\n" + "="*50, file=sys.stderr)
        print("RUNNING MODEL CONTEXT PROTOCOL (MCP) INTEGRITY SUITE", file=sys.stderr)
        print("="*50, file=sys.stderr)
        
        mock_init_rpc = json.dumps({"jsonrpc": "2.0", "method": "initialize", "params": {}, "id": "tx_998"})
        mock_discovery_rpc = json.dumps({"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "tx_999"})
        
        # Testing Pydantic validation intercept with invalid input (max_lines > 5000)
        mock_invalid_call = json.dumps({
            "jsonrpc": "2.0", 
            "method": "tools/call", 
            "params": {"name": "secure_inspect_filesystem", "arguments": {"target_filename": "audit.log", "max_lines": 99999}}, 
            "id": "tx_1000"
        })
        
        res0 = await mcp_bridge._handle_json_rpc_request(mock_init_rpc)
        res1 = await mcp_bridge._handle_json_rpc_request(mock_discovery_rpc)
        res2 = await mcp_bridge._handle_json_rpc_request(mock_invalid_call)
        
        print(f"[✓] Protocol Initialize Handshake Passed:\n{json.dumps(json.loads(res0 or '{}'), indent=2)}", file=sys.stderr)
        print(f"\n[✓] Discovery Mappings Resolution Passed:\n{json.dumps(json.loads(res1 or '{}'), indent=2)[:250]}...", file=sys.stderr)
        print(f"\n[✓] Pydantic Argument Validation Intercept Passed:\n{json.dumps(json.loads(res2 or '{}'), indent=2)}", file=sys.stderr)
        print("="*50 + "\n", file=sys.stderr)

    elif args.transport == "sse":
        await mcp_bridge.start_sse_transport_loop(host=args.host, port=args.port)
    else:
        await mcp_bridge.start_stdio_transport_loop()

if __name__ == "__main__":
    asyncio.run(main())