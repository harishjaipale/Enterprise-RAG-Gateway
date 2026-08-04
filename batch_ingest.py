import os
import asyncio
from ingest_pdf import ProductionIngestionPipeline

async def ingest_all():
    # Target files
    target_files = [
        "Isoniazid_Hepatotoxicity_Report.pdf",
        "Medical_Clinical_Trial_Report.pdf",
        "Q4_Financial_Performance_Statement.pdf",
        "System_Architecture_and_API_Spec.pdf"
    ]

    # Check potential directories where files might reside
    possible_dirs = [".", "data", "database", "./data"]
    
    pipeline = ProductionIngestionPipeline(
        collection_name="enterprise_rag_vector_index",
        embedding_model="BAAI/bge-small-en-v1.5",
        dimensions=384,
        recreate_collection=True # Cleans old incomplete vector state on first file
    )

    processed_count = 0

    for filename in target_files:
        resolved_path = None
        
        # Resolve full path dynamically
        for folder in possible_dirs:
            test_path = os.path.join(folder, filename)
            if os.path.exists(test_path):
                resolved_path = test_path
                break
        
        if not resolved_path:
            print(f"[!] File missing across directory locations: {filename}")
            continue

        # Prevent wiping database after the first file is ingested
        if processed_count > 0:
            pipeline.recreate_collection = False 

        print(f"\n==================================================")
        print(f" PROCESSING [{processed_count + 1}]: {resolved_path}")
        print(f"==================================================")
        
        success = await pipeline.execute_pipeline(resolved_path)
        if success:
            processed_count += 1
            print(f"[✓] Successfully Indexed: {filename}")
        else:
            print(f"[✗] Failed to Index: {filename}")

    print(f"\n[✓] Batch Execution Complete! Total indexed documents: {processed_count}")

if __name__ == "__main__":
    asyncio.run(ingest_all())