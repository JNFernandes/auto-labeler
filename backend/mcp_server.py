"""MCP server: lets Claude Desktop or Claude Code run the auto-labeler as a tool.

    python mcp_server.py            # normally started by the MCP client, not by hand
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)   # data/ and runs/ are relative to backend/, whoever starts us

from fastapi import HTTPException
from PIL import Image

import api

try:                                         # mcp 2.x
    from mcp.server.mcpserver import MCPServer
except ImportError:                          # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

mcp = MCPServer("kitti-autolabel")


def summary(result: dict) -> dict:
    """A short answer for the chat: counts first, then the boxes that need a human."""
    return {"image": result["image"],
            "accepted": len(result["accepted"]), "for_human": len(result["for_human"]), "dropped": len(result["dropped"]),
            "claude_cost_usd": result["claude_cost"],
            "boxes_for_human": result["for_human"], "accepted_boxes": result["accepted"], "trace": result["trace"]}


def run(image: Image.Image, name: str, accept_above, drop_below, min_height, zoom, critic) -> dict:
    try:
        return summary(api.run_agent(image, name, accept_above, drop_below, min_height, zoom, critic))
    except HTTPException as error:           # HTTP errors mean nothing to an MCP client
        raise ValueError(error.detail)


@mcp.tool()
def list_images(limit: int = 20, offset: int = 0) -> dict:
    """List KITTI image ids that can be labelled."""
    ids = api.get_dataset().ids
    return {"total": len(ids), "ids": ids[offset: offset + limit]}


@mcp.tool()
def label_kitti_image(image_id: str, accept_above: float = 0.60, drop_below: float = 0.20,
                      min_height: float = 0.0, zoom: bool = False, use_claude_critic: bool = False) -> dict:
    """Auto-label a KITTI image (cars, pedestrians, cyclists). Boxes at or above accept_above are accepted,
    boxes below drop_below are discarded, the rest are listed for a human. use_claude_critic asks Claude to
    reject fake or wrong uncertain boxes; it costs about a tenth of a cent per box."""
    dataset = api.get_dataset()
    if image_id not in dataset.ids:
        raise ValueError(f"No KITTI image '{image_id}'. Use list_images to see valid ids.")
    return run(dataset.load_image(image_id), image_id, accept_above, drop_below, min_height, zoom, use_claude_critic)


@mcp.tool()
def label_photo(path: str, accept_above: float = 0.60, drop_below: float = 0.20,
                min_height: float = 0.0, zoom: bool = False, use_claude_critic: bool = False) -> dict:
    """Auto-label a photo from a file path on this computer."""
    file = Path(path)
    if not file.is_file():
        raise ValueError(f"No file at '{path}'.")
    try:
        image = Image.open(file).convert("RGB")
    except Exception:
        raise ValueError(f"'{path}' is not an image.")
    return run(image, file.stem, accept_above, drop_below, min_height, zoom, use_claude_critic)


if __name__ == "__main__":
    mcp.run()