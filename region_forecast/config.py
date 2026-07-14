from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    csv_path: str
    results_dir: str
    mode: str  # 'timeseries' | 'timecp' | 'timecap'

    # forecasting setup
    lookback: int = 12
    horizon: int = 12
    train_frac: float = 0.7
    val_frac: float = 0.1
    test_frac: float = 0.2

    # cost / density controls
    test_stride: int = 1
    train_stride: int = 1
    k_examples: int = 5

    # LLM
    model: str = 'openai/gpt-4o-mini'
    api_base: str = 'https://openrouter.ai/api/v1'
    api_key_env: str = 'OPENROUTER_API_KEY'
    temperature: float = 0.7
    max_tokens: int = 1024

    # retrieval embeddings (TimeCAP mode only)
    embedding_model: str = 'all-MiniLM-L6-v2'

    # execution
    workers: int = 4
    dry_run: bool = False

    # region selection
    zipcodes: Optional[List[str]] = None
    max_regions: Optional[int] = None

    # schema
    region_column: str = 'zipcode'
    city_column: str = 'city'
    metro_column: str = 'metro'
    date_column: str = 'date'
    target_column: str = 'price'
    indicator_columns: Optional[List[str]] = None  # None => auto-detect

    def validate(self):
        frac_sum = self.train_frac + self.val_frac + self.test_frac
        if abs(frac_sum - 1.0) > 1e-6:
            raise ValueError(f'train/val/test fractions must sum to 1.0, got {frac_sum}')
        if self.mode not in ('timeseries', 'timecp', 'timecap'):
            raise ValueError(f'unknown mode: {self.mode}')
        if self.lookback <= 0 or self.horizon <= 0:
            raise ValueError('lookback and horizon must be positive')
