import multiprocessing
import uvicorn
import time
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Orchestrator")


def start_e5():
    """Starts the E5 Embedding Service on Port 5000"""
    try:
        uvicorn.run("e5_server:app", host="127.0.0.1", port=5000, reload=True)
    except Exception as e:
        logger.error(f"E5 Server failed: {e}")


def start_retriever():
    """Starts the Retriever Service on Port 8000"""
    try:
        uvicorn.run("retriever:app", host="127.0.0.1", port=8000, reload=True)
    except Exception as e:
        logger.error(f"Retriever Server failed: {e}")


def start_generator():
    """Starts the Generator (LLM) Service on Port 9000"""
    try:
        uvicorn.run("generator:app", host="127.0.0.1", port=9000, reload=True)
    except Exception as e:
        logger.error(f"Generator Server failed: {e}")


def main():
    logger.info("=== Launching Yelp RAG Microservices ===")

    p_e5 = multiprocessing.Process(target=start_e5, name="E5_Service")
    p_retriever = multiprocessing.Process(
        target=start_retriever, name="Retriever_Service"
    )
    p_generator = multiprocessing.Process(
        target=start_generator, name="Generator_Service"
    )

    try:
        # 1. Starting Embedding Service
        logger.info("Starting E5 Embedding Service (Port 5000)...")
        p_e5.start()
        time.sleep(2)

        # 2. Starting Retriever
        logger.info("Starting Retriever Service (Port 8000)...")
        p_retriever.start()
        time.sleep(2)

        # 3. Starting Generator
        logger.info("Starting Generator Service (Port 9000)...")
        p_generator.start()

        logger.info("All services are running.")
        logger.info("Generator Swagger: http://127.0.0.1:9000/docs")
        logger.info("Press Ctrl+C to stop all servers.")

        p_e5.join()
        p_retriever.join()
        p_generator.join()

    except KeyboardInterrupt:
        logger.info("\n🔻 Stopping services...")
        p_generator.terminate()
        p_retriever.terminate()
        p_e5.terminate()

        p_generator.join()
        p_retriever.join()
        p_e5.join()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
