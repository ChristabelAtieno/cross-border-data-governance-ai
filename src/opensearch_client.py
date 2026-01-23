from opensearchpy import OpenSearch

def get_client():
    client = OpenSearch(
        hosts=[{'host': '127.0.0.1', 'port': 9200}],
        http_auth=('admin', 'L3g4l_#XyZ_2026_Stronk_!!'),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        timeout=100
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
        client.search_pipeline.put(id="legal-hybrid-pipeline",body=pipeline_body)
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        pass
    return client


