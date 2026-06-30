from vectordb_bench.backend.clients import DB, IndexType
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.openGauss.config import openGaussHNSWRabitQConfig


def test_opengauss_rabitq_index_and_session_params():
    config = openGaussHNSWRabitQConfig(
        metric_type=MetricType.L2,
        m=16,
        ef_construction=64,
        ef_search=128,
        rbq_query_bits=8,
        rbq_refinek=10,
        rabitq_refine_type="FP32",
    )

    index_param = config.index_param()
    index_options = {
        item["option_name"]: item["val"]
        for item in index_param["index_creation_with_options"]
    }
    session_options = {
        item["parameter"]["setting_name"]: item["parameter"]["val"]
        for item in config.session_param()["session_options"]
    }

    assert (
        DB.openGauss.case_config_cls(IndexType.HNSW_RABITQ)
        is openGaussHNSWRabitQConfig
    )
    assert index_param["index_type"] == "hnsw"
    assert index_param["metric"] == "vector_l2_ops"
    assert index_options == {
        "m": "16",
        "ef_construction": "64",
        "enable_rabitq": "on",
        "rabitq_refine_type": "FP32",
    }
    assert session_options == {
        "hnsw_ef_search": "128",
        "rbq_query_bits": "8",
        "rbq_refinek": "10",
    }
