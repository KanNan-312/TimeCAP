"""
Level-3 ablation: accumulate "experience" from the training pool and query
similar historical cases at test time to condition the forecast on
precedent - conceptually similar to TimeCAP's in-context retrieval, but
retrieved cases are meant to feed the *contextualization* prompt (as
precedent narrative) rather than being spliced directly into the predict
prompt like TimeCAP's P4 does with raw series+outcome pairs.

No retrieval method has been chosen yet, so this module only wires up the
extension point (fit/query) - `context_level=3` can already be turned on
end-to-end (the pipeline calls fit() once and query() per window) without
any pipeline changes once one of these is implemented:

  - Nearest neighbors in the level-1 feature space (momentum/trend/
    volatility vector) via cosine or Euclidean distance - cheap, reuses
    features already computed for level 1.
  - DTW / shape-based similarity directly on the lookback series.
  - Cross-region analogs: same idea as above, but drawn from other
    regions' training windows too, not just this region's own history -
    natural fit given the panel is already spatio-temporal.

Until one of these is implemented, `query()` returns [] so the "Similar
historical precedents" prompt section degrades to an explicit
"not yet available" note (see prompts.py / serialize.py) instead of
breaking the ablation.
"""


class ExperienceStore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cases = []
        self._fitted = False

    def fit(self, train_cases):
        """
        train_cases: list of dicts, one per training-pool (region, window),
        e.g. {'region_id', 'start', 'window', 'history', 'outcome'}.

        TODO: build a similarity index over `train_cases` (e.g. a feature
        vector or learned embedding per case) for `query` to search.
        """
        self.cases = list(train_cases)
        self._fitted = True
        return self

    def query(self, region_id, k):
        """
        TODO: return up to `k` precedent cases most similar to this
        region's current situation (dicts with at least a de-identified
        'summary' field - no zipcode/exact dates - for the prompt layer to
        render). Returns [] until implemented.
        """
        return []
