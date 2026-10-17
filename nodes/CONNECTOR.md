# Connect a GPU across separate networks

The connector opens an outbound encrypted SSH tunnel through the existing public relay. No VPN account, inbound PC firewall rule, or public GPU API is needed. The node service must already run on the Windows PC at `127.0.0.1:18090` (including a Docker port published from WSL2). The relay carries traffic; experiments continue on the GPU node.

## Windows

1. Unzip the controller's private connection bundle into a private folder. Keep `connector_app.py`, `Connect-GPU.bat`, and `relay_ed25519` together. Never upload that key or bundle to Git.
2. Install Python 3 with its Tkinter component and enable Windows **OpenSSH Client** if they are missing.
3. Double-click **Connect-GPU.bat**, then click **Connect**. The included key is selected automatically.
4. Keep the window open and the PC awake. Tell the controller when it shows **Tunnel running**. The controller then verifies node authentication and actual GPU jobs before reporting readiness.

Click **Disconnect** or close the window to stop sharing this tunnel. Internet interruptions reconnect automatically. The application does not install itself at startup, install a driver, or change the node service. A tunnel can run while the local node API is down; GPU readiness is independently checked by the controller.

For a terminal, run `python connector_app.py --headless --key /private/path/relay_ed25519`; Ctrl+C disconnects. The GUI can also run on Linux with Python Tkinter and OpenSSH installed.

## Connection details

- Relay: `shardul@172.96.160.149:22`
- Pinned ED25519 fingerprint: `SHA256:cPm6ER/B86pNFzx4O76UgwJsGgzwMrs7i9g0CcHmFHo`
- Tunnel: relay **loopback** `127.0.0.1:18200` → this PC `127.0.0.1:18090`
- Controller: separately forwards its local port through its own authenticated relay connection.

The supplied dedicated relay key must be restricted server-side to remote forwarding on `127.0.0.1:18200`, without shell access. The app verifies the scanned host key against the pinned fingerprint before SSH, forces strict host checking, ignores user SSH configuration, and suppresses raw SSH output. An identity mismatch stops reconnection. The node's existing API key still authenticates every job; the relay key alone is not a node API credential.

If the connection repeatedly closes, check the relay key permissions, whether another connector already owns port 18200, and relay availability. Do not bypass host-key verification. This single bundle is for one node; additional nodes need their own restricted key and relay port.
