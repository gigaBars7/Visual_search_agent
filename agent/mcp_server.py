from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Settings


Settings.model_rebuild()

mcp = FastMCP("test-tools")


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Умножение двух целых чисел"""
    return a * b


if __name__ == "__main__":
    mcp.run(transport="stdio")
