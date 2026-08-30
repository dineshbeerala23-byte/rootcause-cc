import pytest

from app.services.graph.provider import StaticTopologyProvider


@pytest.fixture
def provider() -> StaticTopologyProvider:
    return StaticTopologyProvider()


@pytest.mark.parametrize("system", ["online-boutique", "sock-shop", "train-ticket"])
def test_get_graph_returns_correct_system(provider, system):
    graph = provider.get_graph(system)
    assert graph.system == system


@pytest.mark.parametrize("system", ["online-boutique", "sock-shop", "train-ticket"])
def test_all_edges_reference_real_nodes(provider, system):
    graph = provider.get_graph(system)
    node_names = {n.name for n in graph.nodes}
    for edge in graph.edges:
        assert edge.source in node_names
        assert edge.target in node_names


def test_online_boutique_main_has_no_outgoing_edge_and_is_flagged(provider):
    graph = provider.get_graph("online-boutique")
    assert any(n.name == "main" for n in graph.nodes)
    assert not any(e.source == "main" for e in graph.edges)
    assert any("main" in note for note in graph.notes)


def test_sock_shop_excludes_rabbitmq_exporter(provider):
    graph = provider.get_graph("sock-shop")
    assert not any(n.name == "rabbitmq-exporter" for n in graph.nodes)


def test_sock_shop_primary_edges_present(provider):
    graph = provider.get_graph("sock-shop")
    edge_pairs = {(e.source, e.target) for e in graph.edges}
    assert ("orders", "payment") in edge_pairs
    assert ("orders", "shipping") in edge_pairs
    assert ("orders", "orders-db") in edge_pairs


def test_train_ticket_has_no_edges_and_says_so(provider):
    graph = provider.get_graph("train-ticket")
    assert graph.edges == []
    assert any("not performed" in note.lower() for note in graph.notes)


def test_train_ticket_nodes_split_into_application_and_database(provider):
    graph = provider.get_graph("train-ticket")
    kinds = {n.kind for n in graph.nodes}
    assert kinds == {"application", "database"}