## Time-series graph representation: `_TimeseriesCoreGraph`, `TimeseriesDAG`, `TimeseriesPDAG`

Contributors: [@Thavinash02](https://github.com/Thavinash02)

Related: pgmpy#3575

### Introduction

pgmpy has no class that can hold a stationary time-series causal graph. The only time-aware model is
`DynamicBayesianNetwork`, which does not fit:

1. It stores `(variable, time_slice)` nodes, and `add_edge` materialises the slice-1 copy of every lag-0 edge
   rather than deriving it, so one link occupies two stored edges.
2. It is limited to lag 1. There is no `tau_max`.
3. It carries CPDs and a `check_model` contract. Causal discovery produces structure only.
4. It has no way to represent an unoriented link. PCMCI+ returns contemporaneous links it cannot orient, and the
   latent-confounder variants return circle and bidirected endpoints.
5. There is no way to read in the output format these algorithms use: tigramite's `(N, N, tau_max + 1)` array,
   where `graph[i, j, tau]` describes the link between `var_i` at lag `tau` and `var_j` in the present.


### Proposed Solution

A private base class `_TimeseriesCoreGraph(_CoreGraph)` holds the lag handling: `tau_max`, node and edge
normalisation, and the adjacency conversion. Two public classes inherit it and declare which edge types they
allow.

<img width="693" height="556" alt="image" src="https://github.com/user-attachments/assets/00fc5e02-dd3f-4420-b263-4d26e0532edd" />



`_CoreGraph`'s subclasses differ in which edge types they allow. The time-series classes differ in what a node
is. Those are independent, which is why `_TimeseriesCoreGraph` can inherit `_CoreGraph` instead of being a
standalone class. `_CoreGraph` is not modified.

**Storage.** The graph is stationary, so only the difference between an edge's endpoints is stored:

```
add_edge(u, v, edge_type)  ->  lag = v_time - u_time,  stored as ((u_var, -lag), (v_var, 0))
```

The caller's time convention does not matter. `(("A", -1), ("B", 0), "->")` and `(("A", 0), ("B", 1), "->")` are
the same link and reach the same stored edge. An edge written backwards in time is stored with its endpoints
swapped and its `edge_type` inverted, using the markers `_CoreGraph` already maintains, so an unoriented link
keeps its endpoint marks. Shifted copies follow from stationarity and are not stored.

This is tigramite's layout, so `graph[i, j, tau]` maps directly onto one stored edge.

#### Why inherit `_CoreGraph`

- `_CoreGraph` stores orientation as per-endpoint markers (`-`, `>`, `o`) on an `nx.MultiGraph`, not as directed
  edges. That covers what time-series discovery returns: `->` oriented, `--` unoriented, and `o>` / `oo` / `<>`
  for the latent-confounder algorithms. None of it assumes contemporaneity.
- Subclasses restrict themselves with `SUPPORTED_EDGE_TYPES`, as `PDAG`, `ADMG` and `MAG` already do.
  `TimeseriesDAG` and `TimeseriesPDAG` differ only in that frozenset and in a mark table, so both are under 80
  lines.
- `from_adjacency` accepts a tigramite mark only if `_MARK_TO_EDGE_TYPE` translates it and the translation is in
  `SUPPORTED_EDGE_TYPES`. `TimeseriesDAG` rejects `o-o` and `x-x` with no code of its own, and `TimeseriesPDAG`
  accepts `o-o` by adding one dictionary entry.
- Acyclicity needs no new code. A lagged edge cannot close a directed cycle: its tail is at a strictly earlier
  offset and stored edges always end at offset 0, so no lagged node has an incoming edge. The inherited cycle
  guard only fires within the present slice.
- `get_parents`, `get_children`, `get_ancestors`, `get_descendants`, `get_skeleton`, `get_marker` / `set_marker`,
  `replace_edge`, roles and plotting are all inherited. Adding a `TimeseriesADMG` would take six lines.

Adding an optional lag to `_CoreGraph` itself would put a temporal concept into `PDAG`, `ADMG` and `MAG`, where
it has no meaning. Keeping it in a subclass leaves those classes unchanged.

### Alternative Solutions

**1. `TimeseriesDAG(DAG)`** — the original plan in pgmpy#3575. 
- `DAG` is an `nx.DiGraph` and `_CoreGraph` is an `nx.MultiGraph`, so a re-parented DBN would change its
  underlying networkx class and lose `moralize` and `active_trail_nodes`.
- `DynamicBayesianNetwork.add_edge` requires `(variable, time_slice)` nodes and normalises slice indices, so
  it accepts less than its parent would.

   Explicit conversion methods are the alternative to inheritance.

### Details of proposed solution

Class purpose and method signatures only.

#### `pgmpy/base/_timeseries_base.py`

```python
class _TimeseriesCoreGraph(_CoreGraph):
    """
    Base class for time-series graph types. Every endpoint is a (variable, time) pair and only the lag between
    an edge's endpoints is stored. Subclasses declare `SUPPORTED_EDGE_TYPES` as the static classes do; the time
    handling here does not depend on which edge types they choose.
    """

    _time_origin = 0            # the time the present slice is reported as when edges are read back
    _STATIC_CLASS = None        # the static class `unroll` expands into
    _MARK_TO_EDGE_TYPE = {...}  # tigramite mark -> pgmpy edge_type; marks not listed are rejected

    def __init__(self, edge_list=None, tau_max=None, exposures=None, outcomes=None, latents=None, roles=None):
        """Builds from (u, v, edge_type) triples. Sets `tau_max` before `super().__init__`, because the inherited constructor adds edges through the overridden `add_edge`, which reads it."""

    # tau_max

    def _set_tau_max(self, tau_max):
        """Stores `tau_max` and whether it was declared. A declared value is enforced; an undeclared one starts at 0 and grows. An (N, N, tau_max + 1) array declares a maximum lag even when no edge reaches it, so the distinction has to survive a round trip."""

    @property
    def tau_max(self):
        """The maximum lag the graph accepts."""

    def _check_lag(self, lag):
        """Enforces a declared `tau_max`, or grows an undeclared one."""

    # normalisation

    @staticmethod
    def _normalize_node(node):
        """Coerces a node into a (variable, time) pair. A bare name is read as time 0; objects with `to_tuple`, such as `DynamicNode`, are unwrapped."""

    def _normalize_edge(self, u, v, edge_type):
        """Reduces an edge to its lag, head at offset 0 and tail at -lag. A negative lag is stored with endpoints swapped and `edge_type` inverted."""

    # mutation

    def add_node(self, node, **attr):
        """Adds a variable at offset 0. An existing node keeps its stored offset, because inherited code such as `with_role` calls `add_node` to attach attributes."""

    def add_nodes_from(self, nodes, **attr): ...

    def add_edge(self, u, v, edge_type):
        """Adds a link. The lag is checked against `tau_max`; the absolute times are not."""

    def remove_edge(self, u, v, edge_type): ...

    def has_edge(self, u, v, edge_type=None):
        """Membership test, in any time frame."""

    # accessors

    @property
    def variables(self):
        """The distinct variable names, ignoring offsets."""

    def _render_node(self, node) / _derender_node(self, node):
        """Converts a stored offset to a caller-facing time and back, shifting by `_time_origin`."""

    def _stored_edges(self, edge_types=None):
        """Edges as stored, with offsets. Everything that reasons about lags uses this, so `_time_origin` only affects what is reported."""

    def get_edges(self, data=True, edge_types=None):
        """The edges, with endpoints in the caller's time frame."""

    def get_links(self):
        """Every edge as (tail_variable, head_variable, lag, edge_type). Maps one-to-one onto a cell of the adjacency array."""

    def get_lag(self, u, v):
        """The lag of a link."""

    def get_intra_edges(self) / get_inter_edges(self):
        """Links at lag 0, and links at lag >= 1."""

    def get_parents_by_lag(self, variable):
        """The parents of a variable, grouped by lag: {1: ["A"], 2: ["A", "C"]}. Conditioning sets need the lag along with the name."""

    # adjacency arrays

    @classmethod
    def from_adjacency(cls, graph, variables=None):
        """Builds the graph from a tigramite-style (N, N, tau_max + 1) array. Accepts string marks and the raw 0/1 array PCMCI returns before conversion."""

    def to_adjacency(self, variables=None):
        """The graph back as an array of string marks, in a given variable order."""

    # conversion

    def unroll(self, n_time_points):
        """Expands the structure over N time points into `_STATIC_CLASS`, with absolute (variable, t) nodes. Only links whose tail is inside the window are drawn, so the first `tau_max` points have incomplete parent sets."""

    @classmethod
    def _build_static(cls, edge_list):
        """Constructs the static counterpart. Defaults to the `_CoreGraph` constructor signature."""

    # identity

    def get_subgraph(self, nodes=None, edge_types=None):
        """As inherited, but carries `tau_max` onto the result. The inherited version rebuilds through `self.__class__()`, which would reset it. `copy` uses this method, so one override covers both."""

    def __eq__(self, other) / __hash__(self):
        """Equal when the inherited structure, roles and class match and `tau_max` agrees."""
```

#### `pgmpy/base/TimeseriesDAG.py`

```python
class TimeseriesDAG(_TimeseriesCoreGraph):
    """
    Directed acyclic graph over a stationary time series. Fully oriented structure only; an algorithm that
    leaves lag-0 links unoriented produces a `TimeseriesPDAG`.
    """

    SUPPORTED_EDGE_TYPES = frozenset(["->", "<-"])
    _STATIC_CLASS = DAG
    _MARK_TO_EDGE_TYPE = {"-->": "->", "<--": "<-"}

    @classmethod
    def _build_static(cls, edge_list):
        """`DAG` is not yet migrated to `_CoreGraph` (pgmpy#3585), so it takes two-tuples through `ebunch`. Deletable once it is."""
```

#### `pgmpy/base/TimeseriesPDAG.py`

```python
class TimeseriesPDAG(_TimeseriesCoreGraph):
    """
    Partially directed counterpart. Lag-0 links that could not be oriented are held as "--". Lagged links are
    always oriented, because time orders them.
    """

    SUPPORTED_EDGE_TYPES = frozenset(["->", "<-", "--"])
    _STATIC_CLASS = PDAG
    _MARK_TO_EDGE_TYPE = {"-->": "->", "<--": "<-", "o-o": "--"}
```

