# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pytest
from app.app_utils.plot_tools import plot_time_trend, plot_runs_comparison

# We will use temporary output paths for saving files during tests
@pytest.fixture
def temp_output_trend(tmp_path):
    return str(tmp_path / "time_trend.png")


@pytest.fixture
def temp_output_comparison(tmp_path):
    return str(tmp_path / "comparison.png")


def test_plot_time_trend_success(temp_output_trend):
    # RunNo 470 exists in merged_data_L303.csv
    result = plot_time_trend(run_id=470, output_filename=temp_output_trend)
    
    assert "Plot saved as" in result
    assert temp_output_trend in result
    assert os.path.exists(temp_output_trend)
    assert os.path.getsize(temp_output_trend) > 0


def test_plot_time_trend_missing_run(temp_output_trend):
    # RunNo 999999 should not exist in the dataset
    result = plot_time_trend(run_id=999999, output_filename=temp_output_trend)
    
    assert "No data found for RunNo 999999" in result
    assert not os.path.exists(temp_output_trend)


def test_plot_runs_comparison_success(temp_output_comparison):
    # RunNos 470 is in the dataset
    result = plot_runs_comparison(run_ids=[470], output_filename=temp_output_comparison)
    
    assert "Plot saved as" in result
    assert temp_output_comparison in result
    assert os.path.exists(temp_output_comparison)
    assert os.path.getsize(temp_output_comparison) > 0


def test_plot_runs_comparison_missing_runs(temp_output_comparison):
    # RunNos 999999, 888888 should not exist
    result = plot_runs_comparison(run_ids=[999999, 888888], output_filename=temp_output_comparison)
    
    assert "No data found for the specified runs" in result
    assert not os.path.exists(temp_output_comparison)
