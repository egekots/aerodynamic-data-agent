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

import pytest
from app.app_utils.data_tools import get_run_metadata


def test_get_run_metadata_success():
    # RunNo 470 exists in merged_data_L303.csv
    result = get_run_metadata(run_id=470)
    
    assert "Metadata for Run 470:" in result
    assert "Profile: l303" in result
    assert "Type: Clean" in result
    assert "Degree: 20 degree mean angle" in result
    assert "Temperature: 63.80 K" in result
    assert "Ambient Pressure: 14.3194 Pa" in result
    assert "Airspeed: 108.60 m/s" in result
    assert "Reynolds Number (Rey): 0.98" in result
    assert "Frequency: 1.79 Hz" in result


def test_get_run_metadata_missing_run():
    # RunNo 999999 should not exist in the dataset
    result = get_run_metadata(run_id=999999)
    
    assert "Error: No data found for RunNo 999999" in result
