"""Tests for the StateGraph engine (app.services.context.graph_rag)."""

import pytest

from app.services.context.graph_rag import (
    ChatState,
    StateGraph,
    extract_entities_node,
    llm_generation_node,
    router_node,
)


class TestChatState:
    def test_defaults(self):
        s = ChatState(session_id="s1", user_message="hi")
        assert s.extracted_entities == {}
        assert s.graph_context == []
        assert s.llm_response is None
        assert s.should_stop is False
        assert s.next_node is None


class TestStateGraph:
    @pytest.mark.asyncio
    async def test_single_node_execution(self):
        async def node(state: ChatState) -> ChatState:
            state.llm_response = "done"
            state.should_stop = True
            return state

        graph = StateGraph()
        graph.add_node("start", node)
        state = ChatState(session_id="t", user_message="x")
        result = await graph.execute("start", state)
        assert result.llm_response == "done"

    @pytest.mark.asyncio
    async def test_edge_traversal(self):
        async def a(state: ChatState) -> ChatState:
            state.graph_context.append("a")
            return state

        async def b(state: ChatState) -> ChatState:
            state.graph_context.append("b")
            state.should_stop = True
            return state

        graph = StateGraph()
        graph.add_node("a", a)
        graph.add_node("b", b)
        graph.add_edge("a", "b")

        state = ChatState(session_id="t", user_message="x")
        result = await graph.execute("a", state)
        assert result.graph_context == ["a", "b"]

    @pytest.mark.asyncio
    async def test_conditional_edge(self):
        async def check(state: ChatState) -> ChatState:
            return state

        async def yes(state: ChatState) -> ChatState:
            state.llm_response = "yes"
            state.should_stop = True
            return state

        async def no(state: ChatState) -> ChatState:
            state.llm_response = "no"
            state.should_stop = True
            return state

        graph = StateGraph()
        graph.add_node("check", check)
        graph.add_node("yes", yes)
        graph.add_node("no", no)
        graph.add_conditional_edge(
            "check", lambda s: "yes" if "go" in s.user_message else "no"
        )

        s1 = await graph.execute("check", ChatState(session_id="t", user_message="go"))
        assert s1.llm_response == "yes"

        s2 = await graph.execute(
            "check", ChatState(session_id="t", user_message="stop")
        )
        assert s2.llm_response == "no"

    @pytest.mark.asyncio
    async def test_missing_node_raises(self):
        graph = StateGraph()
        state = ChatState(session_id="t", user_message="x")
        with pytest.raises(ValueError, match="not found"):
            await graph.execute("missing", state)

    @pytest.mark.asyncio
    async def test_next_node_field_overrides_edges(self):
        async def router(state: ChatState) -> ChatState:
            state.next_node = "target"
            return state

        async def fallback(state: ChatState) -> ChatState:
            state.llm_response = "fallback"
            state.should_stop = True
            return state

        async def target(state: ChatState) -> ChatState:
            state.llm_response = "target"
            state.should_stop = True
            return state

        graph = StateGraph()
        graph.add_node("router", router)
        graph.add_node("fallback", fallback)
        graph.add_node("target", target)
        graph.add_edge("router", "fallback")  # should be overridden

        result = await graph.execute(
            "router", ChatState(session_id="t", user_message="x")
        )
        assert result.llm_response == "target"


class TestBuiltInNodes:
    @pytest.mark.asyncio
    async def test_router_directs_to_extraction(self):
        state = ChatState(session_id="t", user_message="who is Alice?")
        result = await router_node(state)
        assert result.next_node == "extract_entities"

    @pytest.mark.asyncio
    async def test_router_directs_to_llm(self):
        state = ChatState(session_id="t", user_message="tell me a joke")
        result = await router_node(state)
        assert result.next_node == "llm_generation"

    @pytest.mark.asyncio
    async def test_extract_entities_finds_alice(self):
        state = ChatState(session_id="t", user_message="alice is great")
        result = await extract_entities_node(state)
        assert result.extracted_entities.get("person_name") == "Alice"

    @pytest.mark.asyncio
    async def test_llm_generation_with_context(self):
        state = ChatState(
            session_id="t",
            user_message="x",
            graph_context=["Alice knows Bob"],
        )
        result = await llm_generation_node(state)
        assert result.should_stop is True
        assert "Knowledge Graph" in result.llm_response

    @pytest.mark.asyncio
    async def test_llm_generation_without_context(self):
        state = ChatState(session_id="t", user_message="x")
        result = await llm_generation_node(state)
        assert result.should_stop is True
        assert "don't need the graph" in result.llm_response
