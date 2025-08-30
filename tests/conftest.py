import pandas as pd
import pytest
from pathlib import Path

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture
def tmpdir(tmp_path: Path):
    return tmp_path


@pytest.fixture
def sample_hpi_csv(tmp_path: Path) -> Path:
    txt = (
        "DATAFLOW;LAST UPDATE;freq;purchase;unit;geo;TIME_PERIOD;OBS_VALUE;OBS_FLAG;CONF_STATUS\n"
        "ESTAT:PRC_HPI_A;2025-07-04;Annual;Existing;Annual average index, 2010=100;Austria;2015;128.63;;\n"
        "ESTAT:PRC_HPI_A;2025-07-04;Annual;Existing;Annual average index, 2010=100;Austria;2016;136.27;;\n"
    )
    p = tmp_path / "house_price_index.csv"
    p.write_text(txt, encoding="utf-8")
    return p


@pytest.fixture
def sample_earn_csv(tmp_path: Path) -> Path:
    txt = (
        "DATAFLOW,LAST UPDATE,freq,currency,estruct,ecase,geo,TIME_PERIOD,OBS_VALUE,OBS_FLAG,CONF_STATUS\n"
        "ESTAT:EARN_NT_NET,2025-04-29,Annual,Euro,Net earning,Case,Austria,2015,34414.48,,\n"
        "ESTAT:EARN_NT_NET,2025-04-29,Annual,Euro,Net earning,Case,Austria,2016,36707.82,,\n"
    )
    p = tmp_path / "net_earnings.csv"
    p.write_text(txt, encoding="utf-8")
    return p