"""
region_forecast_ctx - an enriched-contextualization extension of the
region_forecast TimeCP baseline (region_forecast/, unmodified - kept as the
comparison baseline).

P1 (contextualize) is no longer an LLM call: a feature calculator computes
statistics from the data and serializes them directly into a compact text
block (prompts.build_context_text) that P3 uses as-is. Deterministic,
free, and reproducible - data -> prompt -> prediction, nothing in between.
Three escalating, independently-ablatable context levels feed it:

  context_level=1  target-series statistics, computed over the region's
                    *entire* causal history to date (not just the lookback
                    window): data overview, long-term trend & STL seasonal
                    structure, volatility & persistence (ACF, mean-reversion
                    crossings), correlation with other indicators. Only
                    short-term momentum (and "recent" volatility) look at a
                    short recent slice, by design - see
                    --momentum-short-months / --volatility-recent-months.
                    Each component is its own --no-<component> switch.
  context_level=2  + spatio-temporal context: haversine distance over
                    latitude/longitude locates the top-k geographic
                    neighbors; reports their trend agreement and how this
                    region's momentum/price level compares to them.
  context_level=3  + experience retrieval from the training pool: k-NN
                    search over training-pool windows pooled across *all*
                    regions, either in a scale-invariant feature space
                    (--experience-method features, default) or over
                    z-normalized lookback-series shape (--experience-method
                    shape, --experience-shape-metric euclidean|dtw). Reports
                    an aggregated, de-identified outcome summary (median/
                    range/direction-agreement across the top-k analogs),
                    never which region/period a match came from - see
                    features/experience.py.

De-identification: nothing sent to the forecasting LLM ever names a zipcode
or exact calendar date - geographic neighbors appear only as "Neighbor 1",
"Neighbor 2", ... ranked by distance. This is to prevent the model from
substituting memorized knowledge of a specific real ZIP/period for genuine
reasoning over the provided data. Local checkpoint files (features/*.json)
still carry region_id for our own bookkeeping - that's not sent to the LLM.

Example (dry run, sanity-check plumbing, no API calls):

  python -m region_forecast_ctx.run_pipeline --csv DC_House.csv \
      --context-level 3 --dry-run --max-regions 2

Example (real run via OpenRouter, level-1 ablation only):

  export OPENROUTER_API_KEY=sk-or-...
  python -m region_forecast_ctx.run_pipeline --csv DC_House.csv \
      --context-level 1 --model openai/gpt-4o-mini --workers 8
"""

import argparse

from dotenv import load_dotenv

from region_forecast.llm import LLMClient
from region_forecast_ctx import data as D
from region_forecast_ctx import pipeline
from region_forecast_ctx.config import Config
from region_forecast_ctx.features.experience import ExperienceStore

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(
        description='region_forecast_ctx: contextualization-enriched region-level price forecasting',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--csv', required=True, help='path to the region-level CSV (zipcode, city, metro, price, date, latitude, longitude, ...)')
    p.add_argument('--results-dir', default='region_forecast_ctx/results')
    p.add_argument('--mode', default='timecp_ctx', choices=['timeseries', 'timecp_ctx'])
    p.add_argument('--stage', default='all', choices=['contextualize', 'experience', 'predict', 'evaluate', 'all'])

    p.add_argument('--lookback', type=int, default=12)
    p.add_argument('--horizon', type=int, default=12)
    p.add_argument('--train-frac', type=float, default=0.7)
    p.add_argument('--val-frac', type=float, default=0.1)
    p.add_argument('--test-frac', type=float, default=0.2)
    p.add_argument('--test-stride', type=int, default=1)
    p.add_argument('--train-stride', type=int, default=1)

    # --- ablation: context level + per-component toggles -----------------
    p.add_argument('--context-level', type=int, default=1, choices=[1, 2, 3],
                    help='1=target-series stats, 2=+spatial neighbors, 3=+experience retrieval (placeholder)')

    p.add_argument('--no-data-overview', dest='ctx_data_overview', action='store_false')
    p.add_argument('--no-momentum', dest='ctx_momentum', action='store_false')
    p.add_argument('--no-trend-seasonal', dest='ctx_trend_seasonal', action='store_false')
    p.add_argument('--no-volatility-persistence', dest='ctx_volatility_persistence', action='store_false')
    p.add_argument('--no-correlation', dest='ctx_correlation', action='store_false')
    p.set_defaults(ctx_data_overview=True, ctx_momentum=True, ctx_trend_seasonal=True,
                    ctx_volatility_persistence=True, ctx_correlation=True)

    p.add_argument('--momentum-short-months', type=int, default=3)
    p.add_argument('--momentum-compare-months', type=int, default=3)
    p.add_argument('--stl-period', type=int, default=12)
    p.add_argument('--volatility-recent-months', type=int, default=6)
    p.add_argument('--correlation-top-n', type=int, default=3)
    p.add_argument('--correlation-min-abs', type=float, default=0.3)

    p.add_argument('--no-spatial', dest='ctx_spatial', action='store_false')
    p.set_defaults(ctx_spatial=True)
    p.add_argument('--spatial-k', type=int, default=5)
    p.add_argument('--spatial-max-km', type=float, default=None)
    p.add_argument('--latitude-column', default='latitude')
    p.add_argument('--longitude-column', default='longitude')

    p.add_argument('--no-experience', dest='ctx_experience', action='store_false')
    p.set_defaults(ctx_experience=True)
    p.add_argument('--experience-k', type=int, default=3)
    p.add_argument('--experience-method', default='features', choices=['features', 'shape'],
                    help="'features'=k-NN on scale-invariant momentum/trend/volatility stats; "
                         "'shape'=distance between z-normalized lookback series")
    p.add_argument('--experience-shape-metric', default='euclidean', choices=['euclidean', 'dtw'],
                    help='only used when --experience-method=shape')
    p.add_argument('--experience-min-gap-months', type=int, default=None,
                    help='min start-index gap between two retrieved analogs from the same region '
                         '(default: = --lookback, i.e. no overlapping analogs)')

    p.add_argument('--model', default='openai/gpt-4o-mini')
    p.add_argument('--api-base', default='https://openrouter.ai/api/v1')
    p.add_argument('--api-key-env', default='OPENROUTER_API_KEY')
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--max-tokens', type=int, default=1024)

    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--dry-run', action='store_true')

    p.add_argument('--zipcodes', default=None)
    p.add_argument('--max-regions', type=int, default=None)

    p.add_argument('--region-column', default='zipcode')
    p.add_argument('--city-column', default='city')
    p.add_argument('--metro-column', default='metro')
    p.add_argument('--date-column', default='date')
    p.add_argument('--target-column', default='price')
    p.add_argument('--indicator-columns', default=None)

    return p.parse_args()


def build_config(args):
    return Config(
        csv_path=args.csv,
        results_dir=args.results_dir,
        mode=args.mode,
        lookback=args.lookback,
        horizon=args.horizon,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        test_stride=args.test_stride,
        train_stride=args.train_stride,
        context_level=args.context_level,
        ctx_data_overview=args.ctx_data_overview,
        ctx_momentum=args.ctx_momentum,
        ctx_trend_seasonal=args.ctx_trend_seasonal,
        ctx_volatility_persistence=args.ctx_volatility_persistence,
        ctx_correlation=args.ctx_correlation,
        momentum_short_months=args.momentum_short_months,
        momentum_compare_months=args.momentum_compare_months,
        stl_period=args.stl_period,
        volatility_recent_months=args.volatility_recent_months,
        correlation_top_n=args.correlation_top_n,
        correlation_min_abs=args.correlation_min_abs,
        ctx_spatial=args.ctx_spatial,
        spatial_k=args.spatial_k,
        spatial_max_km=args.spatial_max_km,
        latitude_column=args.latitude_column,
        longitude_column=args.longitude_column,
        ctx_experience=args.ctx_experience,
        experience_k=args.experience_k,
        experience_method=args.experience_method,
        experience_shape_metric=args.experience_shape_metric,
        experience_min_gap_months=args.experience_min_gap_months,
        model=args.model,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        workers=args.workers,
        dry_run=args.dry_run,
        zipcodes=args.zipcodes.split(',') if args.zipcodes else None,
        max_regions=args.max_regions,
        region_column=args.region_column,
        city_column=args.city_column,
        metro_column=args.metro_column,
        date_column=args.date_column,
        target_column=args.target_column,
        indicator_columns=args.indicator_columns.split(',') if args.indicator_columns else None,
    )


def main():
    args = parse_args()
    cfg = build_config(args)
    cfg.validate()

    df = D.load_dataset(cfg)
    plans, indicator_cols = pipeline.plan_regions(cfg, df)
    n_ok = sum(1 for p in plans.values() if p['status'] == 'ok')
    print(f'[plan] {n_ok}/{len(plans)} region(s) usable for mode={cfg.mode} ablation={cfg.ablation_tag} '
          f'(lookback={cfg.lookback}, horizon={cfg.horizon})')
    print(f'[plan] indicator columns: {indicator_cols}')

    spatial_index = pipeline.build_spatial_index(cfg, df, plans)
    experience_store = ExperienceStore(cfg)

    # `experience` must run before `contextualize` - contextualize *queries*
    # the store that `experience` fits, so on a combined `--stage all` run
    # the store needs to already be fitted (or loaded from a prior run's
    # cache) by the time contextualize's workers start querying it.
    #
    # Contextualization is deterministic (no LLM), so LLMClient - which
    # requires an API key unless --dry-run - is only constructed for the
    # stage that actually needs it.
    if args.stage in ('experience', 'all'):
        pipeline.run_experience_stage(cfg, df, plans, indicator_cols, experience_store)
    if args.stage in ('contextualize', 'all'):
        pipeline.run_contextualize_stage(cfg, df, plans, indicator_cols, spatial_index, experience_store)
    if args.stage in ('predict', 'all'):
        llm = LLMClient(cfg)
        pipeline.run_predict_stage(cfg, df, plans, indicator_cols, llm)
    if args.stage in ('evaluate', 'all'):
        pipeline.run_evaluate_stage(cfg, plans)


if __name__ == '__main__':
    main()
