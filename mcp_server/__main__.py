"""``python -m mcp_server`` — serve Recall over stdio.

stdio is the transport MCP clients (Claude Code, Cursor, VS Code) use to spawn a
local server, so this is the entry point the README's config snippets point at.
"""

from mcp_server.server import main

if __name__ == "__main__":
    main()
