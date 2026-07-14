"""
Region-level home price forecasting pipeline - a TimeCAP baseline extension
for spatio-temporal forecasting.

Modes (mirroring the paper's P2/P3/P4 prediction stages, adapted from
single-step classification to `--horizon`-step numeric forecasting):
  timeseries  - forecast directly from the raw lookback time series (P2).
  timecp      - contextualize the lookback window into text (P1), then
                forecast from that text summary (P3).
  timecap     - contextualize (P1), retrieve the k most similar train-window
                summaries via sentence-embedding cosine similarity, and
                forecast using the summary + retrieved in-context examples
                with their actual outcomes (P4).

Every stage writes one checkpoint file per (region, window) before moving
on, and skips work whose checkpoint already exists - so killing the process
and rerunning the same command resumes automatically. Use --stage to run
(or re-run) a single stage explicitly, e.g. after fixing an API key issue:

  python -m region_forecast.run_pipeline --csv DC_House.csv --mode timecap --stage predict

Example (dry run, no API key / network calls, to sanity-check the plumbing):

  python -m region_forecast.run_pipeline --csv DC_House.csv --mode timecap \
      --dry-run --max-regions 2

Example (real run via OpenRouter):

  export OPENROUTER_API_KEY=sk-or-...
  python -m region_forecast.run_pipeline --csv DC_House.csv --mode timecap \
      --model openai/gpt-4o-mini --workers 8
"""

import argparse

from region_forecast import data as D
from region_forecast import pipeline
from region_forecast.config import Config
from region_forecast.llm import LLMClient


def parse_args():
    p = argparse.ArgumentParser(
        description='TimeCAP region-level price forecasting pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--csv', required=True, help='path to the region-level CSV (zipcode, city, metro, price, date, ...)')
    p.add_argument('--results-dir', default='region_forecast/results')
    p.add_argument('--mode', required=True, choices=['timeseries', 'timecp', 'timecap'])
    p.add_argument('--stage', default='all', choices=['contextualize', 'embed', 'predict', 'evaluate', 'all'])

    p.add_argument('--lookback', type=int, default=12)
    p.add_argument('--horizon', type=int, default=12)
    p.add_argument('--train-frac', type=float, default=0.7)
    p.add_argument('--val-frac', type=float, default=0.1)
    p.add_argument('--test-frac', type=float, default=0.2)

    p.add_argument('--test-stride', type=int, default=1, help='evaluate every Nth valid test window (cost control)')
    p.add_argument('--train-stride', type=int, default=1, help='step between train-pool windows used for TimeCAP retrieval')
    p.add_argument('--k-examples', type=int, default=5, help='number of in-context examples for TimeCAP')

    p.add_argument('--model', default='openai/gpt-4o-mini', help='OpenRouter model id')
    p.add_argument('--embedding-model', default='all-MiniLM-L6-v2', help='sentence-transformers model for TimeCAP retrieval')
    p.add_argument('--api-base', default='https://openrouter.ai/api/v1')
    p.add_argument('--api-key-env', default='OPENROUTER_API_KEY')
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--max-tokens', type=int, default=1024)

    p.add_argument('--workers', type=int, default=4, help='concurrent LLM calls')
    p.add_argument('--dry-run', action='store_true', help='use a deterministic mock LLM, no API calls')

    p.add_argument('--zipcodes', default=None, help='comma-separated list of region ids to restrict to')
    p.add_argument('--max-regions', type=int, default=None)

    p.add_argument('--region-column', default='zipcode')
    p.add_argument('--city-column', default='city')
    p.add_argument('--metro-column', default='metro')
    p.add_argument('--date-column', default='date')
    p.add_argument('--target-column', default='price')
    p.add_argument('--indicator-columns', default=None, help='comma-separated override; default = auto-detect numeric columns')

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
        k_examples=args.k_examples,
        model=args.model,
        embedding_model=args.embedding_model,
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
    print(f'[plan] {n_ok}/{len(plans)} region(s) usable for mode={cfg.mode} '
          f'(lookback={cfg.lookback}, horizon={cfg.horizon})')
    print(f'[plan] indicator columns: {indicator_cols}')

    llm = LLMClient(cfg)

    if args.stage in ('contextualize', 'all'):
        pipeline.run_contextualize_stage(cfg, df, plans, indicator_cols, llm)
    if args.stage in ('embed', 'all'):
        pipeline.run_embed_stage(cfg, plans)
    if args.stage in ('predict', 'all'):
        pipeline.run_predict_stage(cfg, df, plans, indicator_cols, llm)
    if args.stage in ('evaluate', 'all'):
        pipeline.run_evaluate_stage(cfg, plans)


if __name__ == '__main__':
    main()
