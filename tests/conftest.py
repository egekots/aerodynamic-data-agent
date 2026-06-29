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

import sys
from unittest.mock import MagicMock
import google.auth

class MockCredentials:
    def __init__(self):
        self.quota_project_id = "mock-project-id"
        self.project_id = "mock-project-id"
        self.token = "mock-token"
        self.service_account_email = "mock@mock.com"
        self.expired = False
    def refresh(self, request):
        pass
    def apply(self, headers):
        headers["Authorization"] = "Bearer mock-token"
    def before_request(self, request, method, url, headers):
        headers["Authorization"] = "Bearer mock-token"

# Mock google.auth.default to prevent DefaultCredentialsError during test collection and run
google.auth.default = MagicMock(return_value=(MockCredentials(), "mock-project-id"))
