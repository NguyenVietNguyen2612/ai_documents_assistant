import os
import sys
import time
from typing import List, Tuple

# Đảm bảo import được app từ thư mục backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_suite(module_name: str, test_names: List[str]) -> Tuple[int, int, float]:
    print("\n" + "=" * 76)
    print(f"  ▶ ĐANG THỰC THI BỘ KIỂM THỬ: {module_name}")
    print("=" * 76)

    t0 = time.perf_counter()
    passed = 0
    failed = 0

    try:
        mod = __import__(module_name, fromlist=test_names)
    except Exception as e:
        print(f"❌ LỖI KHÔNG THỂ IMPORT MODULE {module_name}: {e}")
        return 0, len(test_names), 0.0

    for test_fn_name in test_names:
        fn = getattr(mod, test_fn_name, None)
        if not fn:
            print(f"❌ Không tìm thấy hàm {test_fn_name} trong {module_name}")
            failed += 1
            continue

        try:
            fn()
            passed += 1
        except AssertionError as ae:
            print(f"❌ ASSERTION FAILED trong {test_fn_name}: {ae}")
            failed += 1
        except Exception as ex:
            print(f"❌ EXCEPTION trong {test_fn_name}: {ex}")
            failed += 1

    elapsed = time.perf_counter() - t0
    return passed, failed, elapsed


def main():
    print("""
============================================================================
       AI DOCUMENTS ASSISTANT - BỘ KIỂM THỬ TOÀN DIỆN (AUTOMATED TEST SUITE)
============================================================================
""")

    suites = [
        (
            "auto_tests.test_unit_document",
            [
                "test_chunk_text_splits_correctly",
                "test_structural_markdown_chunking_injects_header_metadata",
                "test_chunk_pages_preserves_page_numbers",
            ],
            "Document Service & Structural Chunking",
        ),
        (
            "auto_tests.test_unit_laya",
            [
                "test_laya_decision_service_with_agent_mock",
                "test_laya_decision_service_answer_choice",
                "test_laya_fast_conversational_heuristic",
                "test_laya_low_confidence_answer_fallback_to_retrieve",
                "test_laya_decision_fallback_llm",
                "test_multi_tools_and_corrective_rag",
            ],
            "Laya System-1 Decision Router",
        ),
        (
            "auto_tests.test_unit_retriever_bm25",
            [
                "test_bm25_service_index_and_search",
                "test_bm25_service_add_and_remove",
                "test_reranker_service_selects_top_3",
                "test_retriever_pipeline_with_reranker_top_3",
                "test_retriever_ablation_modes_vector_vs_hybrid_vs_rerank",
                "test_small_to_big_window_expansion_merges_adjacent_chunks",
                "test_context_builder_small_to_big_and_corrective_rag_warning",
            ],
            "Hybrid Search (Milvus + BM25 RRF + BGE Reranker)",
        ),
        (
            "auto_tests.test_unit_context_and_crag",
            [
                "test_small_to_big_window_boundary_edge_cases",
                "test_corrective_rag_three_confidence_tiers",
                "test_calculator_tool_ast_safety_and_math",
                "test_document_overview_tool_hierarchy",
            ],
            "Small-to-Big Expansion & Corrective RAG (CRAG)",
        ),
        (
            "auto_tests.test_unit_agent_service",
            [
                "test_count_current_turn_tool_calls_enforces_limit_3",
                "test_sanitize_messages_removes_old_think_tags",
                "test_rescue_text_tool_calls_from_llm_output",
            ],
            "LangGraph ReAct Multi-Tool Agent Engine",
        ),
        (
            "auto_tests.test_integration_api",
            [
                "test_health_check_api",
                "test_documents_list_api",
                "test_chat_sessions_api",
                "test_document_upload_rejects_unsupported_extensions",
                "test_async_background_upload_and_status",
            ],
            "FastAPI REST Endpoints Integration",
        ),
        (
            "auto_tests.test_unit_semantic_cache_and_pruning",
            [
                "test_semantic_cache_store_and_lookup_hit",
                "test_semantic_cache_miss_for_unrelated_query",
                "test_semantic_cache_scoped_document_isolation",
                "test_tool_message_pruning_and_sliding_window",
            ],
            "Semantic Cache & Context Window Pruning",
        ),
    ]

    total_passed = 0
    total_failed = 0
    total_time = 0.0
    summary_table = []

    for mod_path, test_list, suite_desc in suites:
        p, f, dur = run_suite(mod_path, test_list)
        total_passed += p
        total_failed += f
        total_time += dur
        status_str = "✅ PASS" if f == 0 else f"❌ {f} FAIL"
        summary_table.append((suite_desc, f"{p}/{p+f}", f"{dur:.2f}s", status_str))

    print("\n" + "=" * 76)
    print("                  BẢNG TỔNG KẾT KẾT QUẢ KIỂM THỬ (TEST SCORECARD)")
    print("=" * 76)
    print(f"{'Hạng mục kiểm thử':<42} | {'Tỉ lệ':<8} | {'Thời gian':<10} | {'Trạng thái'}")
    print("-" * 76)
    for desc, ratio, dur_str, status in summary_table:
        print(f"{desc:<42} | {ratio:<8} | {dur_str:<10} | {status}")
    print("-" * 76)
    print(f"{'TỔNG CỘNG TOÀN HỆ THỐNG':<42} | {f'{total_passed}/{total_passed+total_failed}':<8} | {f'{total_time:.2f}s':<10} | {'✅ 100% PASS' if total_failed == 0 else '❌ CÓ LỖI'}")
    print("=" * 76 + "\n")

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
