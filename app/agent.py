# ruff: noqa
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
from typing import List, Optional, Literal, Any
from pydantic import BaseModel, Field

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.agents.context import Context
from google.adk.workflow import Workflow, node, Edge, START
from google.adk.events import Event
from google.genai import types

# Import aerodynamic tools
from app.app_utils.data_tools import extract_run_data, get_run_metadata
from app.app_utils.plot_tools import plot_time_trend, plot_runs_comparison
from app.app_utils.advanced_aero_tools import plot_pressure_distribution, compare_integrated_vs_reported


# Define Workflow state schema
class AeroState(BaseModel):
    run_id: Optional[int] = None
    run_ids: Optional[List[int]] = None
    target_alpha: Optional[float] = None
    output_filename: Optional[str] = None
    result: Optional[str] = None


# Define routing decision schema
class AeroRoutingDecision(BaseModel):
    tool_name: Literal[
        "extract_run_data",
        "get_run_metadata",
        "plot_time_trend",
        "plot_runs_comparison",
        "plot_pressure_distribution",
        "compare_integrated_vs_reported",
        "unknown"
    ] = Field(description="The name of the aerodynamic tool to run based on the user request.")
    run_id: Optional[int] = Field(default=None, description="The RunNo (single integer) extracted from the request if the tool expects a single run_id.")
    run_ids: Optional[List[int]] = Field(default=None, description="The list of RunNo integers extracted from the request if the tool expects multiple runs.")
    target_alpha: Optional[float] = Field(default=None, description="The target angle of attack (AOA or Alpha) in degrees, if applicable.")
    output_filename: Optional[str] = Field(default=None, description="The output file path to save the generated plot to, if applicable.")


def extract_text(node_input: Any) -> str:
    """Helper to extract text content from various types of inputs."""
    if not node_input:
        return ""
    if isinstance(node_input, str):
        return node_input
    if hasattr(node_input, "parts") and node_input.parts:
        parts_text = []
        for part in node_input.parts:
            if hasattr(part, "text") and part.text:
                parts_text.append(part.text)
        return "".join(parts_text)
    if isinstance(node_input, dict):
        parts = node_input.get("parts", [])
        if parts:
            parts_text = []
            for part in parts:
                if isinstance(part, dict) and "text" in part:
                    parts_text.append(part["text"])
                elif hasattr(part, "text") and part.text:
                    parts_text.append(part.text)
            return "".join(parts_text)
    return str(node_input)


# --- Workflow Graph Nodes ---

@node
async def router_node(ctx: Context, node_input: Any = None):
    """Router node that acts as the intent classifier and extracts parameters."""
    prompt = extract_text(node_input)
    if not prompt:
        res = "Please provide a query for the Ohio Aerodynamic Agent."
        ctx.output = res
        yield Event(content=types.Content(role="model", parts=[types.Part.from_text(text=res)]))
        return

    try:
        from google.genai import Client
        api_key = os.environ.get("GEMINI_API_KEY") or "MOCK_API_KEY"
        client = Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AeroRoutingDecision,
                temperature=0.0,
                system_instruction=(
                    "You are the intent classifier and entity extractor for the Ohio Aerodynamic Agent.\n"
                    "Analyze the user's prompt and select the most appropriate tool to call.\n"
                    "If the user wants to:\n"
                    "- Extract, get, show, summary, statistics, or view data for a run: select 'extract_run_data'.\n"
                    "- Get header information, test conditions, or metadata about a specific run: select 'get_run_metadata'.\n"
                    "- Plot, show, visualize Cl/Cdp/Cm over time or time trend for a run: select 'plot_time_trend'.\n"
                    "- Compare multiple runs (AOA vs Cl/Cdp/Cm): select 'plot_runs_comparison'.\n"
                    "- Plot pressure distribution (Cp vs x/c) for one or more runs at a target alpha/AOA: select 'plot_pressure_distribution'.\n"
                    "- Compare integrated vs reported values: select 'compare_integrated_vs_reported'.\n"
                    "Ensure you extract all required fields: run_id (single integer), run_ids (list of integers), target_alpha (float), and output_filename (string) if specified."
                )
            )
        )
        import json
        decision_data = json.loads(response.text)
        decision = AeroRoutingDecision(**decision_data)
    except Exception as e:
        res = f"Error during intent classification: {e}"
        ctx.output = res
        yield Event(content=types.Content(role="model", parts=[types.Part.from_text(text=res)]))
        return

    # Update state variables (using dict-like bracket notation on State object)
    ctx.state["run_id"] = decision.run_id
    ctx.state["run_ids"] = decision.run_ids
    ctx.state["target_alpha"] = decision.target_alpha
    ctx.state["output_filename"] = decision.output_filename

    # Align run_id and run_ids if necessary
    if decision.tool_name in ("plot_runs_comparison", "plot_pressure_distribution"):
        if not ctx.state.get("run_ids") and ctx.state.get("run_id") is not None:
            ctx.state["run_ids"] = [ctx.state.get("run_id")]
    if decision.tool_name in ("extract_run_data", "get_run_metadata", "plot_time_trend", "compare_integrated_vs_reported"):
        if ctx.state.get("run_id") is None and ctx.state.get("run_ids"):
            ctx.state["run_id"] = ctx.state.get("run_ids")[0]

    # Generate default output filenames if not specified
    if not ctx.state.get("output_filename"):
        if decision.tool_name == "plot_time_trend" and ctx.state.get("run_id") is not None:
            ctx.state["output_filename"] = f"time_trend_run_{ctx.state.get('run_id')}.png"
        elif decision.tool_name == "plot_runs_comparison":
            ctx.state["output_filename"] = "runs_comparison.png"
        elif decision.tool_name == "plot_pressure_distribution":
            ctx.state["output_filename"] = "pressure_distribution.png"
        elif decision.tool_name == "compare_integrated_vs_reported" and ctx.state.get("run_id") is not None:
            ctx.state["output_filename"] = f"compare_integrated_vs_reported_{ctx.state.get('run_id')}.png"

    if decision.tool_name == "unknown":
        res = (
            "I could not determine the appropriate aerodynamic tool for your request. "
            "Please specify if you want to extract run data, plot time trend, plot runs comparison, "
            "plot pressure distribution, or compare integrated vs reported values."
        )
        ctx.output = res
        yield Event(content=types.Content(role="model", parts=[types.Part.from_text(text=res)]))
    else:
        ctx.route = decision.tool_name


@node
async def run_extract_run_data(ctx: Context, node_input: Any = None):
    run_id = ctx.state.get("run_id")
    if run_id is None:
        res = "Error: run_id was not provided in the request."
    else:
        res = extract_run_data(run_id)
    ctx.output = res
    yield Event(content=types.Content(role="model", parts=[types.Part.from_text(text=res)]))


@node
async def run_get_run_metadata(ctx: Context, node_input: Any = None):
    run_id = ctx.state.get("run_id")
    if run_id is None:
        res = "Error: run_id was not provided in the request."
    else:
        res = get_run_metadata(run_id)
    ctx.output = res
    yield Event(content=types.Content(role="model", parts=[types.Part.from_text(text=res)]))


@node
async def run_plot_time_trend(ctx: Context, node_input: Any = None):
    run_id = ctx.state.get("run_id")
    output_filename = ctx.state.get("output_filename")
    if run_id is None:
        res = "Error: run_id was not provided in the request."
    else:
        if not output_filename:
            output_filename = f"time_trend_run_{run_id}.png"
        res = plot_time_trend(run_id, output_filename)
    ctx.output = res
    yield Event(content=types.Content(role="model", parts=[types.Part.from_text(text=res)]))


@node
async def run_plot_runs_comparison(ctx: Context, node_input: Any = None):
    run_ids = ctx.state.get("run_ids")
    output_filename = ctx.state.get("output_filename")
    if not run_ids:
        res = "Error: run_ids list was not provided in the request."
    else:
        if not output_filename:
            output_filename = "runs_comparison.png"
        res = plot_runs_comparison(run_ids, output_filename)
    ctx.output = res
    yield Event(content=types.Content(role="model", parts=[types.Part.from_text(text=res)]))


@node
async def run_plot_pressure_distribution(ctx: Context, node_input: Any = None):
    run_ids = ctx.state.get("run_ids")
    target_alpha = ctx.state.get("target_alpha")
    output_filename = ctx.state.get("output_filename")
    if not run_ids:
        res = "Error: run_ids list was not provided in the request."
    elif target_alpha is None:
        res = "Error: target_alpha (target angle of attack) was not provided in the request."
    else:
        if not output_filename:
            output_filename = "pressure_distribution.png"
        res = plot_pressure_distribution(run_ids, target_alpha, output_filename)
    ctx.output = res
    yield Event(content=types.Content(role="model", parts=[types.Part.from_text(text=res)]))


@node
async def run_compare_integrated_vs_reported(ctx: Context, node_input: Any = None):
    run_id = ctx.state.get("run_id")
    output_filename = ctx.state.get("output_filename")
    if run_id is None:
        res = "Error: run_id was not provided in the request."
    else:
        if not output_filename:
            output_filename = f"compare_integrated_vs_reported_{run_id}.png"
        res = compare_integrated_vs_reported(run_id, output_filename)
    ctx.output = res
    yield Event(content=types.Content(role="model", parts=[types.Part.from_text(text=res)]))


# Build the ADK 2.0 Graph Workflow
root_agent = Workflow(
    name="root_agent",
    state_schema=AeroState,
    edges=[
        Edge(from_node=START, to_node=router_node),
        Edge(from_node=router_node, to_node=run_extract_run_data, route="extract_run_data"),
        Edge(from_node=router_node, to_node=run_get_run_metadata, route="get_run_metadata"),
        Edge(from_node=router_node, to_node=run_plot_time_trend, route="plot_time_trend"),
        Edge(from_node=router_node, to_node=run_plot_runs_comparison, route="plot_runs_comparison"),
        Edge(from_node=router_node, to_node=run_plot_pressure_distribution, route="plot_pressure_distribution"),
        Edge(from_node=router_node, to_node=run_compare_integrated_vs_reported, route="compare_integrated_vs_reported"),
    ]
)

app = App(
    root_agent=root_agent,
    name="app",
)
