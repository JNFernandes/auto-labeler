import asyncio

import pytest

import mcp_server
from test_api import client


def call(name, arguments):
    client()    # puts the fake dataset and fake models in place
    return asyncio.run(mcp_server.mcp.call_tool(name, arguments))


def test_tools_are_listed():
    names = [t.name for t in asyncio.run(mcp_server.mcp.list_tools())]
    assert names == ["list_images", "label_kitti_image", "label_photo"]


def test_label_kitti_image_summary():
    out = str(call("label_kitti_image", {"image_id": "000001"}))
    assert "boxes_for_human" in out and "000001" in out


def test_unknown_image_is_an_error():
    with pytest.raises(Exception):
        call("label_kitti_image", {"image_id": "nope"})