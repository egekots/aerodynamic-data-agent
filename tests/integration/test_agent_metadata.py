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

from unittest.mock import MagicMock, patch
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import root_agent


def test_agent_metadata_routing() -> None:
    """
    Integration test to verify that the agent correctly routes metadata/header requests
    to the get_run_metadata tool and returns the expected information.
    """
    # Create a mock response object
    mock_response = MagicMock()
    mock_response.text = '{"tool_name": "get_run_metadata", "run_id": 470}'

    # We patch google.genai.Client to return a mock client whose models.generate_content returns our mock response
    with patch("google.genai.Client") as MockClient:
        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = mock_response
        MockClient.return_value = mock_client_instance

        session_service = InMemorySessionService()
        session = session_service.create_session_sync(user_id="test_user", app_name="test")
        runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

        message = types.Content(
            role="user", parts=[types.Part.from_text(text="What are the test conditions for run 470?")]
        )

        events = list(
            runner.run(
                new_message=message,
                user_id="test_user",
                session_id=session.id,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            )
        )
    
    assert len(events) > 0, "Expected at least one event from the runner execution"

    response_text = ""
    for event in events:
        if (
            event.content
            and event.content.parts
        ):
            for part in event.content.parts:
                if part.text:
                    response_text += part.text

    assert "Metadata for Run 470:" in response_text
    assert "Profile: l303" in response_text
    assert "Type: Clean" in response_text
    assert "Degree: 20 degree mean angle" in response_text
    assert "Temperature: 63.80 K" in response_text
    assert "Ambient Pressure: 14.3194 Pa" in response_text
    assert "Airspeed: 108.60 m/s" in response_text
    assert "Reynolds Number" in response_text
    assert "Frequency" in response_text
