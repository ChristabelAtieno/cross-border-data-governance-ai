import os
from opensearchpy import OpenSearch

def get_client():
    host = os.getenv("OPENSEARCH_HOST", "127.0.0.1")
    port = int(os.getenv("OPENSEARCH_PORT", "9200"))
    username = os.getenv("OPENSEARCH_USERNAME")
    password = os.getenv("OPENSEARCH_PASSWORD")

    #use_ssl = os.getenv("OPENSEARCH_USE_SSL", "false").strip().lower() in {"1", "true", "yes", "y"}
    #verify_certs = os.getenv("OPENSEARCH_VERIFY_CERTS", "false").strip().lower() in {"1", "true", "yes", "y"}
    timeout = int(os.getenv("OPENSEARCH_TIMEOUT", "100"))

    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=(username, password) if username and password else None,
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        ssl_assert_hostname=False,
        timeout=timeout,
    )
    try:
        pipeline_body = {
            "description": "Pipeline for hybrid search combining BM25 and vector search",
            "phase_results_processors": [
                {
                    "normalization-processor": {
                        "normalization": {"technique": "min_max"},
                        "combination": {
                            "technique": "arithmetic_mean",
                            "parameters": {"weights": [0.3, 0.7]}
                        }
                    }
                }
            ]
        }
        pipeline_id = os.getenv("OPENSEARCH_SEARCH_PIPELINE", "legal-hybrid-pipeline")
        client.search_pipeline.put(id=pipeline_id, body=pipeline_body)
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        pass
    return client


