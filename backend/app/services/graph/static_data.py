from __future__ import annotations

from app.schemas.graph import ServiceEdge, ServiceGraph, ServiceNode

# ============================================================
# ONLINE BOUTIQUE
# Nodes/edges source-verified from live GoogleCloudPlatform/microservices-demo
# main.go / recommendation_server.py (see conversation research report).
# "main" node: RCAEval telemetry token with no matching pod name in any
# official Online Boutique manifest. Strong circumstantial evidence it
# corresponds to "loadgenerator" (exact 1-to-1 set match against the
# real 12-pod list), but NOT confirmed by an explicit RCAEval statement.
# The one edge that would depend on that identity (main -> frontend) is
# deliberately OMITTED, not guessed.
# ============================================================

_OB_NODE_DEFS = [
    ("frontend", "application"),
    ("cartservice", "application"),
    ("productcatalogservice", "application"),
    ("currencyservice", "application"),
    ("paymentservice", "application"),
    ("shippingservice", "application"),
    ("emailservice", "application"),
    ("checkoutservice", "application"),
    ("recommendationservice", "application"),
    ("adservice", "application"),
    ("redis", "database"),
    ("main", "application"),  # unresolved identity -- see notes
]

_OB_EDGES = [
    ("frontend", "productcatalogservice"),
    ("frontend", "currencyservice"),
    ("frontend", "cartservice"),
    ("frontend", "recommendationservice"),
    ("frontend", "checkoutservice"),
    ("frontend", "shippingservice"),
    ("frontend", "adservice"),
    ("checkoutservice", "shippingservice"),
    ("checkoutservice", "productcatalogservice"),
    ("checkoutservice", "cartservice"),
    ("checkoutservice", "currencyservice"),
    ("checkoutservice", "emailservice"),
    ("checkoutservice", "paymentservice"),
    ("recommendationservice", "productcatalogservice"),
    ("cartservice", "redis"),
]

_OB_NOTES = [
    "'main' telemetry token has no matching pod name in any official "
    "Online Boutique manifest checked. Strongly suspected to be "
    "'loadgenerator' (exact 1-to-1 set match against the real 12-pod "
    "list) but NOT confirmed by an explicit RCAEval source. The "
    "main -> frontend edge that would follow from that identity is "
    "deliberately omitted, not guessed.",
]

ONLINE_BOUTIQUE_GRAPH = ServiceGraph(
    system="online-boutique",
    nodes=[ServiceNode(name=n, system="online-boutique", kind=k) for n, k in _OB_NODE_DEFS],
    edges=[ServiceEdge(source=s, target=t) for s, t in _OB_EDGES],
    notes=_OB_NOTES,
)

# ============================================================
# SOCK SHOP
# 3 edges primary-verified from live microservices-demo/orders
# OrdersController.java. Remaining edges are secondary-source
# (workflow walkthroughs / repo descriptions), not source-grepped.
# rabbitmq-exporter excluded (Prometheus sidecar, not app logic).
# ============================================================

_SS_NODE_DEFS = [
    ("front-end", "application"),
    ("carts", "application"),
    ("carts-db", "database"),
    ("catalogue", "application"),
    ("catalogue-db", "database"),
    ("orders", "application"),
    ("orders-db", "database"),
    ("payment", "application"),
    ("queue-master", "application"),
    ("rabbitmq", "infrastructure"),
    ("session-db", "database"),
    ("shipping", "application"),
    ("user", "application"),
    ("user-db", "database"),
]

_SS_EDGES_PRIMARY = [
    ("orders", "payment"),
    ("orders", "shipping"),
    ("orders", "orders-db"),
]

_SS_EDGES_SECONDARY = [
    ("front-end", "catalogue"),
    ("front-end", "carts"),
    ("front-end", "orders"),
    ("front-end", "user"),
    ("catalogue", "catalogue-db"),
    ("carts", "carts-db"),
    ("user", "user-db"),
    ("user", "session-db"),
    ("queue-master", "rabbitmq"),
]

_SS_NOTES = [
    "Primary-verified edges (from live microservices-demo/orders "
    "OrdersController.java): orders->payment, orders->shipping, "
    "orders->orders-db.",
    "Secondary-source-only edges (workflow walkthroughs / repo "
    "descriptions, NOT individually source-grepped): front-end->{catalogue,"
    "carts,orders,user}, catalogue->catalogue-db, carts->carts-db, "
    "user->{user-db,session-db}, queue-master->rabbitmq.",
    "rabbitmq-exporter excluded as a Prometheus sidecar, not application logic.",
]

SOCK_SHOP_GRAPH = ServiceGraph(
    system="sock-shop",
    nodes=[ServiceNode(name=n, system="sock-shop", kind=k) for n, k in _SS_NODE_DEFS],
    edges=[ServiceEdge(source=s, target=t) for s, t in _SS_EDGES_PRIMARY + _SS_EDGES_SECONDARY],
    notes=_SS_NOTES,
)

# ============================================================
# TRAIN TICKET
# Nodes only, taken directly from the real RE1 metrics.parquet column
# list (re1tt_ts-auth-service_cpu_1), split by suffix: "-mongo"/"-mysql"
# => database, else => application. Edge research was NOT performed --
# edges list is intentionally empty. Node count (62 here) does not
# cleanly reconcile with RCAEval's stated 64 or FudanSELab's stated 41;
# not resolved, flagged in notes.
# ============================================================

_TT_APPLICATION = [
    "ts-admin-basic-info-service", "ts-admin-order-service", "ts-admin-route-service",
    "ts-admin-travel-service", "ts-admin-user-service", "ts-assurance-service",
    "ts-auth-service", "ts-avatar-service", "ts-basic-service", "ts-cancel-service",
    "ts-config-service", "ts-consign-price-service", "ts-consign-service",
    "ts-contacts-service", "ts-execute-service", "ts-food-map-service", "ts-food-service",
    "ts-inside-payment-service", "ts-news-service", "ts-notification-service",
    "ts-order-other-service", "ts-order-service", "ts-preserve-other-service",
    "ts-preserve-service", "ts-price-service", "ts-rebook-service", "ts-route-plan-service",
    "ts-route-service", "ts-seat-service", "ts-security-service", "ts-station-service",
    "ts-ticket-office-service", "ts-ticketinfo-service", "ts-train-service",
    "ts-travel-plan-service", "ts-travel-service", "ts-travel2-service", "ts-ui-dashboard",
    "ts-user-service", "ts-verification-code-service", "ts-voucher-service",
]

_TT_DATABASE = [
    "ts-assurance-mongo", "ts-auth-mongo", "ts-config-mongo", "ts-consign-mongo",
    "ts-consign-price-mongo", "ts-contacts-mongo", "ts-food-map-mongo", "ts-food-mongo",
    "ts-inside-payment-mongo", "ts-order-mongo", "ts-order-other-mongo", "ts-payment-mongo",
    "ts-price-mongo", "ts-route-mongo", "ts-security-mongo", "ts-station-mongo",
    "ts-ticket-office-mongo", "ts-train-mongo", "ts-travel-mongo", "ts-travel2-mongo",
    "ts-user-mongo", "ts-voucher-mysql",
]

_TT_NOTES = [
    "Edge research NOT performed -- edges intentionally empty. Do not "
    "add edges to this graph without a dedicated, separate research pass "
    "(41-64 service system; too large for the confidence bar used on "
    "the other two systems in this pass).",
    "Node count here (62) does not cleanly reconcile with RCAEval's "
    "stated 64 services or FudanSELab's stated 41 microservices; not "
    "resolved. Application/database split is a direct read of the "
    "'-service' vs '-mongo'/'-mysql' naming suffix in the real telemetry "
    "columns, not independently verified against source.",
]

TRAIN_TICKET_GRAPH = ServiceGraph(
    system="train-ticket",
    nodes=(
        [ServiceNode(name=n, system="train-ticket", kind="application") for n in _TT_APPLICATION]
        + [ServiceNode(name=n, system="train-ticket", kind="database") for n in _TT_DATABASE]
    ),
    edges=[],
    notes=_TT_NOTES,
)

_GRAPHS_BY_SYSTEM = {
    "online-boutique": ONLINE_BOUTIQUE_GRAPH,
    "sock-shop": SOCK_SHOP_GRAPH,
    "train-ticket": TRAIN_TICKET_GRAPH,
}