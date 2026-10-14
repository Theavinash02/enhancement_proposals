## Time-series graph representation: `_TimeseriesCoreGraph`, `TimeseriesDAG`, `TimeseriesPDAG`

Contributors: [@Theavinash02](https://github.com/Theavinash02)

Related: pgmpy#3575

### Introduction

pgmpy has no class that can hold a stationary time-series causal graph. The only time-aware model is
`DynamicBayesianNetwork`, which does not fit:

1. It stores `(variable, time_slice)` nodes, and `add_edge` materialises the slice-1 copy of every lag-0 edge
   rather than deriving it, so one link occupies two stored edges.
2. It is limited to lag 1. There is no `tau_max`.
3. It carries CPDs and a `check_model` contract. Causal discovery produces structure only.
4. It has no way to represent an unoriented link.
5. There is no way to read in the output format these algorithms use: tigramite's `(N, N, tau_max + 1)` array,
   where `graph[i, j, tau]` describes the link between `var_i` at lag `tau` and `var_j` in the present.

### Proposed Solution

A private base class `_TimeseriesCoreGraph(_CoreGraph)` would hold the lag handling: `tau_max`, node and edge
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

A single `_time_origin` class attribute decides what "now" is called when edges are read back. It is 0 in the
base, so a lag-1 link reads as `("A", -1) -> ("B", 0)`. A subclass that counts slices forward sets it to 1 and
the same link reads as `("A", 0) -> ("B", 1)`. Storage does not change; only what the caller sees. Every method
that reports endpoints - `get_edges`, `get_parents`, `get_children` - renders through it, and every method that
reasons about lags works on the stored offsets, so the two can never disagree.

Where a method takes an optional edge type, the "any type" value is `None` rather than a literal edge type.
`"--"` would be the obvious sentinel but it is not in `TimeseriesDAG.SUPPORTED_EDGE_TYPES`, so a query would be
passing that class an edge type it does not accept. `None` is carried through an endpoint swap unchanged.

#### Adjacency: `"tigramite"` as a sixth encoding

`_CoreGraph` already defines `from_adjacency` / `to_adjacency`, with an `encoding` argument selecting the
format - `edge_type`, `binary`, `bnlearn`, `causal-learn`, `pcalg` - and documents the round-trip invariant
`cls.from_adjacency(g.to_adjacency(e), encoding=e) == g`.

A time-series graph needs a third dimension for the lag, which none of those encodings has. Overriding the
inherited methods with a different signature and return type would silently break that invariant for anyone
holding a `TimeseriesDAG`.

The proposal is to add `"tigramite"` as a sixth encoding rather than to override:

```python
from_adjacency(cls, adj, encoding="edge_type", variables=None)
to_adjacency(self, encoding="edge_type", nodelist=None, variables=None)
```

- The default stays `edge_type`, so nothing changes for a caller who does not ask for the new format.
- A 3-D input auto-selects `tigramite`, so an estimator handing over a `(N, N, tau_max + 1)` array needs no
  encoding argument.
- Every other encoding delegates to `super()`.
- The round-trip invariant holds for all six.

Naming precedent: three of the five existing encodings (`bnlearn`, `causal-learn`, `pcalg`) are already named
after third-party tools, so `tigramite` is consistent with how the argument is already used.

The table translating tigramite’s three-character marks to and from pgmpy’s two-character edge types is local
to `from_adjacency` and `to_adjacency`, exactly as `_CoreGraph` already keeps its `pcalg` and `causal-learn`
code tables inside the two methods that use them. It covers the whole tigramite vocabulary, and
`SUPPORTED_EDGE_TYPES` does the filtering: a mark is accepted only if the table translates it and the
translation is one the class admits. Mark rejection therefore needs no per-class code and no per-class table -
`TimeseriesDAG` turns away `o-o` because `--`is not in its frozenset, through the same validation path as any
other edge. tigramite’s `x-x` has no pgmpy edge type and so is in no table at all.

#### Why inherit `_CoreGraph`

- `_CoreGraph` stores orientation as per-endpoint markers (`-`, `>`, `o`) on an `nx.MultiGraph`, not as directed
  edges. That covers what time-series discovery returns: `->` oriented, `--` unoriented, and `o>` / `oo` / `<>`
  for the latent-confounder algorithms. None of it assumes contemporaneity.
- Subclasses restrict themselves with `SUPPORTED_EDGE_TYPES`, as `PDAG`, `ADMG` and `MAG` already do.
  `TimeseriesDAG` and `TimeseriesPDAG` differ only in that frozenset.

#### What happens to `DynamicBayesianNetwork`

A DBN is the `tau_max = 1` case of a time-series graph, which makes re-parenting it onto `TimeseriesDAG` look
natural. Under `_TimeseriesCoreGraph(_CoreGraph)` the cost is concrete: `TimeseriesDAG` is not a `DAG`, so a
re-parented DBN would lose `moralize` and `active_trail_nodes`, which `_CoreGraph` does not define, along with
its `nx.DiGraph` substrate.

### Alternative Solutions

**1. `TimeseriesDAG(DAG)`** - the original direction in pgmpy#3575. `DAG` still inherits `nx.DiGraph` rather
than `_CoreGraph` (pgmpy#3585 is pending), so it does not store endpoint markers: `TimeseriesPDAG` could not
share a base with it, and unoriented links would have nowhere to go. Every lag-handling method would also have
to be written against a second storage model rather than reusing the one `PDAG`, `ADMG` and `MAG` already use.

### Details of proposed solution

Class purpose and method signatures only.

#### `pgmpy/base/_timeseries_base.py`

```python
class _TimeseriesCoreGraph(_CoreGraph):
    """
    Base class for time-series graph types. A node is a (variable, time) pair, and what gets stored for an
    edge is the gap between its two times, not the times themselves. Subclasses say which edge types they
    accept, the same way the non-time-series classes do; none of the time handling here depends on that
    choice.
    """

    _time_origin = 0        # what "now" is called when an edge is handed back to the caller

    def __init__(self, edge_list=None, tau_max=None, exposures=None, outcomes=None, latents=None, roles=None):
        """
        Builds the graph from (u, v, edge_type) triples.

        `tau_max` has to be set before calling the parent constructor, because the parent adds the edges by
        calling `add_edge`, and this class's `add_edge` checks the lag against `tau_max`.
        """

    # tau_max

    def _set_tau_max(self, tau_max):
        """
        Records the maximum lag, and whether the caller asked for one.

        Those are two different situations. If a maximum was given, exceeding it is an error. If none was
        given, it starts at 0 and rises to fit whatever is added. The difference has to be remembered because
        an (N, N, tau_max + 1) array states a maximum even when no edge actually reaches it, and converting
        the graph back has to produce an array of the same shape.
        """

    @property
    def tau_max(self):
        """The largest lag this graph will accept."""

    def _check_lag(self, lag):
        """Rejects a lag above a maximum the caller set, or raises the maximum when the caller set none."""

    # normalisation

    @staticmethod
    def _normalize_node(node):
        """
        Turns whatever was passed into a (variable, time) pair.

        A bare name counts as time 0. A `DynamicNode`, or anything else with a `to_tuple` method, is unwrapped
        first.
        """

    def _normalize_edge(self, u, v, edge_type):
        """
        Works out the lag and puts the edge in one fixed position: head at time 0, tail at minus the lag.

        An edge written backwards in time is not refused. Its two ends are swapped and its edge type inverted,
        which gives the same link written the other way round. `edge_type=None` means "any type" and stays
        None through the swap.
        """

    # mutation

    def add_node(self, node, **attr):
        """
        Adds a variable, placed at time 0.

        A node already in the graph keeps the time it has. This matters because inherited code such as
        `with_role` calls `add_node` to attach an attribute to an existing node; moving it would attach the
        attribute to a different node.
        """

    def add_nodes_from(self, nodes, **attr):
        """Several at once, each following the rule above."""

    def add_edge(self, u, v, edge_type):
        """
        Adds a link.

        The lag is checked against `tau_max`. The times themselves are not checked, so any time frame works.
        """

    def remove_edge(self, u, v, edge_type):
        """
        Removes a link, written in any time frame.

        The edge goes through the same normalising as `add_edge`, so a link can be removed using different
        times than the ones it was added with.
        """

    def has_edge(self, u, v, edge_type=None):
        """Whether a link is present, in any time frame. `None` means any edge type will do."""

    # accessors

    @property
    def variables(self):
        """
        The variable names, with times dropped and duplicates removed.

        This also fixes the row and column order of the tigramite array.
        """

    def _render_node(self, node) / _derender_node(self, node):
        """
        The one place a stored time becomes a time the caller sees, and the reverse.

        Both shift by `_time_origin`, which is 0 here, so by default they change nothing.
        """

    def _stored_edges(self, edge_types=None):
        """
        The edges as stored, with their own times rather than the caller's.

        Everything that calculates with lags uses this rather than `get_edges`, which is what keeps
        `_time_origin` from affecting any result.
        """

    def get_edges(self, data=True, edge_types=None):
        """The edges, with their times converted into the caller's frame."""

    def get_parents(self, nodes) / get_children(self, nodes):
        """
        As inherited, except the times are converted into the caller's frame.

        Without that they would disagree with `get_edges`, which reports the same edges.
        """

    def get_links(self):
        """
        Every edge as (tail_variable, head_variable, lag, edge_type), with no times at all. One tuple
        corresponds to one cell of the tigramite array.

        The earlier end is always put first. That is not automatic: for a symmetric edge type such as `"--"`
        the inherited edge view may return the two ends either way round, and the lag has to be measured from
        the earlier one.
        """

    def get_lag(self, u, v):
        """How far apart in time the two ends of a link are, whatever type of link it is."""

    def get_intra_edges(self) / get_inter_edges(self):
        """
        Links within one time point, and links across time points.

        Named to match the methods DBN already has.
        """

    def get_parents_by_lag(self, variable):
        """
        The parents of a variable, grouped by how far back they sit: {1: ["A"], 2: ["A", "C"]}.

        The lag has to come back alongside the name, because that pair is what a conditioning set is built
        from.
        """

    # adjacency

    @classmethod
    def from_adjacency(cls, adj, encoding="edge_type", variables=None):
        """
        Adds one more encoding, "tigramite", to the five the parent class already accepts. A tigramite array
        has three dimensions, (N, N, tau_max + 1), because of the lag. Both shapes tigramite produces work:
        text marks like "-->", and the plain 0/1 array it gives before conversion.

        Because none of the other five encodings is three-dimensional, a 3-D array picks this one on its own
        and the caller need not name it. Any other encoding is passed straight to the parent class.

        The mark table is defined here, covering the whole tigramite vocabulary; a mark is kept only if its
        translation is in the class’s SUPPORTED_EDGE_TYPES. A subclass declares nothing for this to work.
        """

    def to_adjacency(self, encoding="edge_type", nodelist=None, variables=None):
        """
        The reverse, with the same six encodings, and the inverse table defined here.

        A graph converted out and back comes back equal to itself, for every one of them.
        """

    # identity

    def get_subgraph(self, nodes=None, edge_types=None):
        """
        As inherited, but the result keeps the original `tau_max`.

        The inherited version builds the result by calling `self.__class__()` with no arguments, which would
        reset it to 0. `copy` is built on this method, so this one override fixes both.
        """

    def __eq__(self, other) / __hash__(self):
        """
        Two graphs are equal when the inherited comparison says so and their `tau_max` agrees.

        Same links but different maximum lags means different graphs, because they accept different things.
        """
```

#### `pgmpy/base/TimeseriesDAG.py`

```python
class TimeseriesDAG(_TimeseriesCoreGraph):
    """
    Directed acyclic graph over a stationary time series. Every link is oriented. An algorithm that leaves
    some same-time links unoriented produces a `TimeseriesPDAG` instead.
    """

    SUPPORTED_EDGE_TYPES = frozenset(["->", "<-"])
```

A DAG has only directed links, so why list both `"->"` and `"<-"`? Because an edge written backwards in time
is turned round rather than refused, and turning it round inverts its type, so the link arrives as `"<-"`. If
`"<-"` were not accepted, that rewrite would fail. `PDAG` lists both for the same reason.

#### `pgmpy/base/TimeseriesPDAG.py`

```python
class TimeseriesPDAG(_TimeseriesCoreGraph):
    """
    The partially directed counterpart. A same-time link the algorithm could not orient is held as "--".
    Lagged links are always oriented, because time decides the direction.
    """

    SUPPORTED_EDGE_TYPES = frozenset(["->", "<-", "--"])
```

`"--"` means the algorithm found a link but could not say which way it points. That can only happen at lag 0.
Across time there is nothing to decide: the earlier event is the cause, because the later one cannot be.

Neither subclass defines a method, and neither declares anything but its frozenset. That one line is also what
decides which tigramite marks the class will accept.

### User journeys with the solution
**1. Building a graph**
 
```python
from pgmpy.base import TimeseriesDAG
 
dag = TimeseriesDAG(
    edge_list=[
        (("A", -1), ("A", 0), "->"),
        (("A", -1), ("B", 0), "->"),
        (("B", 0), ("C", 0), "->"),
        (("C", -2), ("A", 0), "->"),
    ],
    tau_max=2,
)
dag.tau_max
```
 
```
2
```
 
A maximum the caller sets is kept even when no edge reaches it, so converting to an array produces the shape
they asked for. Left out, it starts at 0 and rises to fit whatever is added:
 
```python
g = TimeseriesDAG(edge_list=[(("A", -1), ("B", 0), "->")])
g.tau_max
```
 
```
1
```
 

### Open Questions
