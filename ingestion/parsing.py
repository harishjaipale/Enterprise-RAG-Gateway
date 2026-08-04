import os
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv


# Config Layer Integration
try:
    from config.settings import settings
except ImportError:
    settings = None

# Logger setup
logger = logging.getLogger("ProductionDocumentParser")


class ProductionDocumentParser:
    """
    Enterprise-grade, resilient document parser.
    Uses LlamaParse for high-fidelity markdown layout extraction, 
    with a graceful local fallback mechanism (PyPDF/Text) if APIs are unreachable.
    """
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        result_type: str = "markdown",
        language: str = "en",
        **kwargs  # Captures extra parameters like tier, api_version safely
    ):
        # Fetch API key via Config settings -> Explicit Arg -> ENV
        if api_key:
            self.api_key = api_key
        elif settings and hasattr(settings, 'parser') and settings.parser.llama_cloud_api_key:
            self.api_key = settings.parser.llama_cloud_api_key
        else:
            self.api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        

        self.result_type = result_type
        self.language = language
        self.extra_kwargs = kwargs

        if not self.api_key:
            logger.warning(
                "LLAMA_CLOUD_API_KEY missing! LlamaParse engine disabled. "
                "Parser will operate in Local Fallback Extraction Mode."
            )

    async def _fallback_local_parse(self, file_path_obj: Path) -> str:
        """
        Local fallback parser using PyPDF or standard text extraction 
        when LlamaParse cloud service is offline or unconfigured.
        """
        logger.info(f"Executing Local Fallback Parsing for: {file_path_obj.name}")
        ext = file_path_obj.suffix.lower()

        if ext == ".pdf":
            def read_pdf():
                try:
                    import pypdf
                    reader = pypdf.PdfReader(str(file_path_obj))
                    text_pages = [page.extract_text() for page in reader.pages if page.extract_text()]
                    return "\n\n".join(text_pages)
                except ImportError:
                    logger.error("pypdf package not installed for local PDF parsing fallback.")
                    return f"# {file_path_obj.stem}\n\n[PDF text extraction unavailable: Install pypdf]"

            return await asyncio.to_thread(read_pdf)
        else:
            # Standard plain-text / markdown reader
            def read_text():
                with open(file_path_obj, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()

            return await asyncio.to_thread(read_text)

    async def parse(
        self, 
        file_path: str, 
        output_dir: str = "./data/output",
        input_options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Parses document asynchronously with strict zero-crash error boundaries.
        """
        file_path_obj = Path(file_path).resolve()
        output_dir_obj = Path(output_dir).resolve()
        
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Target document missing at specified path: {file_path_obj}")
            
        output_dir_obj.mkdir(parents=True, exist_ok=True)
        
        parsed_content = ""

        # Attempt Primary Engine: LlamaParse Cloud
        if self.api_key:
            try:
                from llama_parse import LlamaParse

                logger.info(f"Triggering LlamaParse Cloud Engine for: {file_path_obj.name}")
                
                parse_params = {
                    "api_key": self.api_key,
                    "result_type": self.result_type,
                    "language": self.language
                }
                if input_options:
                    parse_params.update(input_options)

                parser = LlamaParse(**parse_params)
                documents = await parser.aload_data(str(file_path_obj))
                parsed_content = "\n\n".join([doc.text for doc in documents])

            except Exception as llama_err:
                logger.error(f"LlamaParse extraction failed ({llama_err}). Switching to Local Fallback Engine.")
                parsed_content = await self._fallback_local_parse(file_path_obj)
        else:
            parsed_content = await self._fallback_local_parse(file_path_obj)

        # Non-blocking async disk write
        output_file = output_dir_obj / f"{file_path_obj.stem}_parsed.md"
        
        def save_file():
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(parsed_content)

        await asyncio.to_thread(save_file)
        logger.info(f"[✓] Document parsing & archival complete: {output_file}")
        
        return parsed_content