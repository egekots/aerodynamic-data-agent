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
from app.app_utils.data_tools import extract_run_data
from app.app_utils.advanced_aero_tools import plot_pressure_distribution, compare_integrated_vs_reported


@pytest.fixture
def temp_output_dist(tmp_path):
    return str(tmp_path / "pressure_dist.png")


@pytest.fixture
def temp_output_compare(tmp_path):
    return str(tmp_path / "compare.png")


def test_extract_run_data_success():
    # RunNo 470 exists in merged_data_L303.csv
    result = extract_run_data(run_id=470)
    assert "Summary for Run 470:" in result
    assert "Number of Samples:" in result
    assert "Angle of Attack (AOA) Range:" in result
    assert "Average Lift Coefficient (Cl):" in result


def test_extract_run_data_missing_run():
    # RunNo 999999 should not exist
    result = extract_run_data(run_id=999999)
    assert "Error: No data found for RunNo 999999" in result


def test_plot_pressure_distribution_success(temp_output_dist):
    # Run 470 exists
    result = plot_pressure_distribution(run_ids=[470], target_alpha=7.6, output_filename=temp_output_dist)
    assert "Plot saved as" in result
    assert temp_output_dist in result
    assert os.path.exists(temp_output_dist)
    assert os.path.getsize(temp_output_dist) > 0


def test_plot_pressure_distribution_missing_runs(temp_output_dist):
    # Run 999999 does not exist
    result = plot_pressure_distribution(run_ids=[999999], target_alpha=7.6, output_filename=temp_output_dist)
    assert "Error: No data found for the specified runs" in result
    assert not os.path.exists(temp_output_dist)


def test_compare_integrated_vs_reported_success(temp_output_compare):
    # Run 470 exists
    result = compare_integrated_vs_reported(run_id=470, output_filename=temp_output_compare)
    assert "Plot saved as" in result
    assert temp_output_compare in result
    assert os.path.exists(temp_output_compare)
    assert os.path.getsize(temp_output_compare) > 0


def test_compare_integrated_vs_reported_missing_run(temp_output_compare):
    # Run 999999 does not exist
    result = compare_integrated_vs_reported(run_id=999999, output_filename=temp_output_compare)
    assert "Error: No data found for RunNo 999999" in result
    assert not os.path.exists(temp_output_compare)
